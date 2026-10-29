"""Task events and approval dossiers; no HTTP production bypass or compute."""

from typing import Literal

from fastapi import APIRouter, Query, Request, Response

from argus.dashboard.api_response import (
    ApiEnvelope, api_forbidden, api_not_found, api_unavailable,
)
from argus.dashboard.auth import require_permission
from argus.dashboard.routes.analysis_tasks import _service
from argus.dashboard.routes.task_platform import CommandKey, _actor, command, safe_document
from argus.dashboard.task_lifecycle_contracts import (
    TaskEventReviewRequest, TaskReleaseDossierRequest, TaskReleaseEvidenceRequest,
    TaskReleaseReviewRequest, TaskReleaseRevokeRequest, TaskTrainingCancelRequest,
    TaskTrainingCapabilitiesResponse, TaskTrainingExecutionRequest,
)
from argus.dashboard.task_platform_contracts import TaskDocumentResponse, TaskDocumentsResponse, TaskReviewImportRequest
from argus.storage.analysis_tasks import AnalysisTaskConflictError

router = APIRouter()


class _UnconfirmedLifecycleOutcome(Exception):
    """Unexpected runtime failures may follow a commit; never imply a 409."""


def _training_context_matches(request, service):
    """Do not splice startup trust and live observations from different owners."""
    state = request.app.state
    configuration = getattr(state, "task_training_config", None)
    matches = getattr(service, "configuration_matches", None)
    try:
        if not callable(matches) or matches(configuration) is not True:
            return False
        workflows = getattr(state, "task_workflows", None)
        if workflows is None or getattr(service, "workflows", None) is not workflows:
            return False
        current = getattr(state, "config", None)
        if current is not None and matches(getattr(current, "task_training", None)) is not True:
            return False
        runtime = getattr(state, "task_training_runtime", None)
        if runtime is not None and getattr(getattr(runtime, "coordinator", None), "service", None) is not service:
            return False
        admission = getattr(state, "task_training_admission", None)
        if admission is not None and matches(getattr(admission, "config", None)) is not True:
            return False
    except Exception:
        return False
    return True


def _invoke(request, task_id, name, action, *, permission="read_cameras", human=False,
            collection=False, status_code=200, response_model=None, scope_check=None):
    if not require_permission(request, permission):
        return api_forbidden()
    actor = _actor(request, permission, human=human)
    if human and actor is None:
        return api_forbidden("此操作需要已登录的个人身份")
    tasks, service = _service(request), getattr(request.app.state, name, None)
    if tasks is None or service is None:
        return api_unavailable("任务服务不可用")
    try:
        if tasks.get_task(task_id) is None:
            return api_not_found()
        if scope_check is not None:
            scope_check(service)
    except LookupError:
        return api_not_found()
    except Exception:
        return api_unavailable("任务状态不可用")
    if human and _actor(request, permission, human=True) != actor:
        return api_forbidden("个人身份或权限已变更，请重新登录后操作")
    if name == "task_training" and not _training_context_matches(request, service):
        return api_unavailable("训练服务配置或执行实例不一致")

    def call():
        try:
            return action(service, actor)
        except Exception as exc:
            # Event revision conflicts inherit ValueError; map CAS failures
            # before the generic domain adapter classifies validation errors.
            if type(exc).__name__.endswith("Conflict"):
                raise AnalysisTaskConflictError("TASK_LIFECYCLE_REVISION_CONFLICT") from exc
            if (isinstance(exc, RuntimeError)
                    and not isinstance(exc, AnalysisTaskConflictError)
                    and not (type(exc).__module__.startswith("argus.")
                             and type(exc).__name__.endswith("Blocked"))):
                raise _UnconfirmedLifecycleOutcome("lifecycle outcome requires reconciliation") from exc
            raise

    return command(call, model=response_model or (TaskDocumentsResponse if collection else TaskDocumentResponse),
                   transform=(lambda value: value) if response_model is not None else
                             (lambda rows: {"documents": safe_document(rows)}) if collection else None,
                   status_code=status_code)


def _release(service, task_id, release_id):
    row = service.get(release_id)
    if row is None or row["task_id"] != task_id:
        raise LookupError("release not found")
    return row


@router.get("/{task_id}/events", response_model=ApiEnvelope[TaskDocumentsResponse])
def events(request: Request, task_id: str, limit: int = Query(default=100, ge=1, le=100)):
    return _invoke(request, task_id, "task_events", lambda s, _: s.list(task_id, limit=limit), collection=True)


@router.get("/{task_id}/events/{event_id}", response_model=ApiEnvelope[TaskDocumentResponse])
def event(request: Request, task_id: str, event_id: str):
    def read(service, _):
        result = service.get(task_id, event_id)
        if result is None:
            raise LookupError("event not found")
        return result
    return _invoke(request, task_id, "task_events", read)


@router.post("/{task_id}/events/{event_id}/review", response_model=ApiEnvelope[TaskDocumentResponse])
def review_event(request: Request, task_id: str, event_id: str, body: TaskEventReviewRequest):
    return _invoke(request, task_id, "task_events", lambda s, a: s.review(
        task_id, event_id, actor=a, **body.model_dump()), permission="handle_alerts", human=True)


@router.get("/{task_id}/events/{event_id}/replay", response_model=ApiEnvelope[TaskDocumentResponse])
def event_replay(request: Request, task_id: str, event_id: str):
    return _invoke(request, task_id, "task_events", lambda s, _: s.replay_metadata(task_id, event_id),
                   permission="view_replay")


@router.get("/{task_id}/events/{event_id}/evidence")
def event_evidence(request: Request, task_id: str, event_id: str):
    if not require_permission(request, "view_replay"):
        return api_forbidden()
    tasks, service = _service(request), getattr(request.app.state, "task_events", None)
    if tasks is None or service is None:
        return api_unavailable()
    try:
        if tasks.get_task(task_id) is None:
            return api_not_found()
        payload, mime = service.load_evidence(task_id, event_id)
        return Response(payload, media_type=mime, headers={
            "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
        })
    except LookupError:
        return api_not_found()
    except Exception:
        return api_unavailable("事件证据不可用或摘要核验失败")


@router.get("/{task_id}/releases", response_model=ApiEnvelope[TaskDocumentsResponse])
def releases(request: Request, task_id: str):
    return _invoke(request, task_id, "task_releases", lambda s, _: s.list(task_id),
                   permission="read_models", collection=True)


@router.get("/{task_id}/releases/readiness", response_model=ApiEnvelope[TaskDocumentResponse])
def release_readiness(request: Request, task_id: str, release_id: str | None = None):
    return _invoke(request, task_id, "task_releases", lambda s, _: s.readiness(task_id, release_id),
                   permission="read_models")


@router.post("/{task_id}/releases", response_model=ApiEnvelope[TaskDocumentResponse])
def request_release(request: Request, task_id: str, body: TaskReleaseDossierRequest, idempotency_key: CommandKey):
    return _invoke(request, task_id, "task_releases", lambda s, a: s.request(
        task_id, actor=a, idempotency_key=idempotency_key, **body.model_dump()),
        permission="manage_models", human=True, status_code=202)


@router.get("/{task_id}/releases/{release_id}", response_model=ApiEnvelope[TaskDocumentResponse])
def release(request: Request, task_id: str, release_id: str):
    return _invoke(request, task_id, "task_releases", lambda s, _: _release(s, task_id, release_id),
                   permission="read_models")


def _change_release(request, task_id, release_id, action, body, idempotency_key):
    def write(service, actor):
        _release(service, task_id, release_id)
        return getattr(service, action)(release_id, actor=actor, idempotency_key=idempotency_key,
                                        **body.model_dump())
    return _invoke(request, task_id, "task_releases", write, permission="manage_models", human=True)


@router.post("/{task_id}/releases/{release_id}/evidence", response_model=ApiEnvelope[TaskDocumentResponse])
def revise_release(request: Request, task_id: str, release_id: str, body: TaskReleaseEvidenceRequest,
                   idempotency_key: CommandKey):
    return _change_release(request, task_id, release_id, "revise_evidence", body, idempotency_key)


@router.post("/{task_id}/releases/{release_id}/review", response_model=ApiEnvelope[TaskDocumentResponse])
def review_release(request: Request, task_id: str, release_id: str, body: TaskReleaseReviewRequest,
                   idempotency_key: CommandKey):
    return _change_release(request, task_id, release_id, "review", body, idempotency_key)


@router.post("/{task_id}/releases/{release_id}/revoke", response_model=ApiEnvelope[TaskDocumentResponse])
def revoke_release(request: Request, task_id: str, release_id: str, body: TaskReleaseRevokeRequest,
                   idempotency_key: CommandKey):
    return _change_release(request, task_id, release_id, "revoke", body, idempotency_key)


def _cached_training_snapshot(request: Request, name: str, *, resource=False):
    component = getattr(request.app.state, name, None)
    snapshot = getattr(component, "snapshot", None)
    if not callable(snapshot):
        return None
    try:
        result = snapshot()
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    # No raw diagnostic text (paths, subprocess errors or credentials) is public.
    return ({"state": "blocked", "reason_code": "TRAINING_RESOURCE_SNAPSHOT_UNAVAILABLE"} if resource else
            {"state": "error", "enabled": False, "error_code": "TRAINING_RUNTIME_SNAPSHOT_UNAVAILABLE"})


@router.get("/{task_id}/training-capabilities", response_model=ApiEnvelope[TaskTrainingCapabilitiesResponse])
def training_capabilities(request: Request, task_id: str):
    def read(service, _):
        return service.capabilities(
            runtime_snapshot=_cached_training_snapshot(request, "task_training_runtime"),
            resource_snapshot=_cached_training_snapshot(request, "task_training_admission", resource=True),
        )
    return _invoke(request, task_id, "task_training", read, permission="manage_training",
                   response_model=TaskTrainingCapabilitiesResponse)


@router.get("/{task_id}/training-executions", response_model=ApiEnvelope[TaskDocumentsResponse])
def training_jobs(request: Request, task_id: str):
    return _invoke(request, task_id, "task_training", lambda s, _: s.list(task_id=task_id),
                   permission="manage_training", collection=True)


@router.post("/{task_id}/workflows/{workflow_id}/training-executions", response_model=ApiEnvelope[TaskDocumentResponse])
def submit_training(request: Request, task_id: str, workflow_id: str, body: TaskTrainingExecutionRequest,
                    idempotency_key: CommandKey):
    def scope(_service):
        workflows = request.app.state.task_workflows
        if workflows.get(workflow_id)["task_id"] != task_id:
            raise LookupError("workflow not found")
    return _invoke(request, task_id, "task_training", lambda service, actor: service.submit(
        workflow_id=workflow_id, actor=actor, idempotency_key=idempotency_key, **body.model_dump()),
        permission="manage_training", human=True, status_code=202, scope_check=scope)


@router.get("/{task_id}/workflows/{workflow_id}/annotation-export", response_model=ApiEnvelope[TaskDocumentResponse])
def export_annotations(request: Request, task_id: str, workflow_id: str,
                       target: Literal["cvat", "fiftyone"] = "cvat"):
    def export(service, _):
        if service.get(workflow_id)["task_id"] != task_id:
            raise LookupError("workflow not found")
        return service.export_annotation_exchange(workflow_id, target=target)
    return _invoke(request, task_id, "task_workflows", export, permission="manage_training")


@router.post("/{task_id}/workflows/{workflow_id}/annotation-import", response_model=ApiEnvelope[TaskDocumentResponse])
def import_annotations(request: Request, task_id: str, workflow_id: str, body: TaskReviewImportRequest):
    def ingest(service, actor):
        if service.get(workflow_id)["task_id"] != task_id:
            raise LookupError("workflow not found")
        return service.import_annotation_exchange(workflow_id, actor=actor, **body.model_dump())
    # Third-party annotations remain unapproved suggestions even when an
    # authenticated automation token uploads them. Individual review is a
    # separate existing human-only endpoint.
    return _invoke(request, task_id, "task_workflows", ingest, permission="manage_training")


@router.post("/{task_id}/training-executions/{job_id}/cancel", response_model=ApiEnvelope[TaskDocumentResponse])
def cancel_training(request: Request, task_id: str, job_id: str, body: TaskTrainingCancelRequest):
    def scope(service):
        if service.get(job_id)["task_id"] != task_id:
            raise LookupError("training execution not found")
    return _invoke(request, task_id, "task_training", lambda service, actor: service.cancel(
        job_id, actor=actor, **body.model_dump()), permission="manage_training", human=True, scope_check=scope)
