"""Task-scoped data workflows and offline review/training exchanges."""

from fastapi import APIRouter, Request, Response

from argus.dashboard.api_response import (
    ApiEnvelope, api_forbidden, api_not_found, api_unavailable,
)
from argus.dashboard.auth import require_permission
from argus.dashboard.routes.analysis_tasks import _service
from argus.dashboard.routes.task_platform import CommandKey, _actor, command, safe_document
from argus.dashboard.task_platform_contracts import (
    TaskCandidateImportRequest, TaskCollectionRequest, TaskDocumentResponse,
    TaskDocumentsResponse, TaskReviewImportRequest, TaskReviewRequest,
    TaskRevisionRequest, TaskTrainingProposalRequest,
)

router = APIRouter()


def _invoke(request, task_id, workflow_id, action):
    service = getattr(request.app.state, "task_workflows", None)
    if service is None:
        raise RuntimeError("task workflow service is unavailable")
    if service.get(workflow_id)["task_id"] != task_id:
        raise KeyError("workflow does not belong to task")
    return action(service)


@router.get("/{task_id}/workflows", response_model=ApiEnvelope[TaskDocumentsResponse])
def workflows(request: Request, task_id: str):
    if not require_permission(request, "read_cameras"):
        return api_forbidden()
    service = getattr(request.app.state, "task_workflows", None)
    tasks = _service(request)
    if service is None or tasks is None:
        return api_unavailable()
    def list_existing():
        if tasks.get_task(task_id) is None:
            raise LookupError("task not found")
        return service.list(task_id=task_id)
    return command(list_existing, model=TaskDocumentsResponse,
                   transform=lambda rows: {"documents": safe_document(rows)})


@router.post("/{task_id}/workflows", response_model=ApiEnvelope[TaskDocumentResponse])
def create_workflow(request: Request, task_id: str, idempotency_key: CommandKey):
    actor = _actor(request, "manage_training")
    if actor is None:
        return api_forbidden()
    service = getattr(request.app.state, "task_workflows", None)
    if service is None:
        return api_unavailable()
    return command(lambda: service.create(task_id, actor=actor, idempotency_key=idempotency_key), status_code=201)


@router.post("/{task_id}/workflows/{workflow_id}/collect", response_model=ApiEnvelope[TaskDocumentResponse])
def collect(request: Request, task_id: str, workflow_id: str, body: TaskCollectionRequest):
    actor = _actor(request, "edit_config")
    if actor is None:
        return api_forbidden()
    manager = getattr(request.app.state, "camera_manager", None)
    if manager is None:
        return api_unavailable("相机采集源不可用")
    return command(lambda: _invoke(request, task_id, workflow_id, lambda service:
        service.start_collection(workflow_id, **body.model_dump(), manager=manager, actor=actor)))


@router.post("/{task_id}/workflows/{workflow_id}/stop", response_model=ApiEnvelope[TaskDocumentResponse])
def stop_collection(request: Request, task_id: str, workflow_id: str, body: TaskRevisionRequest):
    actor = _actor(request, "edit_config")
    if actor is None:
        return api_forbidden()
    return command(lambda: _invoke(request, task_id, workflow_id, lambda service:
        service.stop_collection(workflow_id, **body.model_dump(), actor=actor)))


@router.get("/{task_id}/workflows/{workflow_id}/samples/{sample_id}/image")
def sample_image(request: Request, task_id: str, workflow_id: str, sample_id: str):
    if not require_permission(request, "read_cameras"):
        return api_forbidden()
    try:
        image = _invoke(request, task_id, workflow_id,
                        lambda service: service.read_sample_image(workflow_id, sample_id))
        return Response(image, media_type="image/png", headers={
            "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
        })
    except LookupError:
        return api_not_found()
    except Exception:
        return api_unavailable("审核图片不可用或摘要校验失败")


@router.post("/{task_id}/workflows/{workflow_id}/review", response_model=ApiEnvelope[TaskDocumentResponse])
def review(request: Request, task_id: str, workflow_id: str, body: TaskReviewRequest):
    actor = _actor(request, "manage_training", human=True)
    if actor is None:
        return api_forbidden("审核需要已登录的人工身份")
    return command(lambda: _invoke(request, task_id, workflow_id, lambda service:
        service.review(workflow_id, actor=actor, **body.model_dump())))


@router.post("/{task_id}/workflows/{workflow_id}/freeze", response_model=ApiEnvelope[TaskDocumentResponse])
def freeze_dataset(request: Request, task_id: str, workflow_id: str, body: TaskRevisionRequest):
    actor = _actor(request, "manage_training", human=True)
    if actor is None:
        return api_forbidden()
    return command(lambda: _invoke(request, task_id, workflow_id, lambda service:
        service.freeze(workflow_id, actor=actor, **body.model_dump())))


@router.get("/{task_id}/workflows/{workflow_id}/review-export", response_model=ApiEnvelope[TaskDocumentResponse])
def export_review(request: Request, task_id: str, workflow_id: str, target: str = "cvat"):
    if not require_permission(request, "manage_training"):
        return api_forbidden()
    return command(lambda: _invoke(request, task_id, workflow_id, lambda service:
        service.export_review(workflow_id, target=target)))


@router.post("/{task_id}/workflows/{workflow_id}/review-import", response_model=ApiEnvelope[TaskDocumentResponse])
def import_review(request: Request, task_id: str, workflow_id: str, body: TaskReviewImportRequest):
    actor = _actor(request, "manage_training", human=True)
    if actor is None:
        return api_forbidden()
    return command(lambda: _invoke(request, task_id, workflow_id, lambda service:
        service.import_review(workflow_id, actor=actor, **body.model_dump())))


@router.post("/{task_id}/workflows/{workflow_id}/preannotations", response_model=ApiEnvelope[TaskDocumentResponse])
def preannotations(request: Request, task_id: str, workflow_id: str, body: TaskRevisionRequest):
    actor = _actor(request, "manage_training")
    if actor is None:
        return api_forbidden()
    executor = getattr(request.app.state, "task_execution", None)
    if executor is None:
        return api_unavailable("旁路任务结果不可用")

    def suggest(service):
        from argus.core.task_preannotations import propose_review_annotations

        envelope = service.export_review(workflow_id, target="cvat")
        if envelope["workflow_revision"] != body.expected_revision:
            raise RuntimeError("WORKFLOW_REVISION_CHANGED")
        payload = propose_review_annotations(envelope, executor.results(task_id, limit=20))
        if not payload["results"]:
            raise RuntimeError("EXACT_FRAME_OBSERVATIONS_UNAVAILABLE")
        return service.import_review(workflow_id, expected_revision=body.expected_revision, payload=payload, actor=actor)

    return command(lambda: _invoke(request, task_id, workflow_id, suggest))


@router.get("/{task_id}/training-recipes", response_model=ApiEnvelope[TaskDocumentsResponse])
def recipes(request: Request, task_id: str):
    if not require_permission(request, "manage_training"):
        return api_forbidden()
    service = getattr(request.app.state, "task_workflows", None)
    tasks = _service(request)
    if service is None or tasks is None:
        return api_unavailable()
    def compatible_recipes():
        task = tasks.get_task(task_id)
        if task is None:
            raise LookupError("task not found")
        if task.get("authority") != "managed" or task.get("archived"):
            return []
        return [recipe for recipe in service.recipes.list()
                if task["template_id"] in recipe.get("templates", [])]
    return command(compatible_recipes, model=TaskDocumentsResponse,
                   transform=lambda rows: {"documents": safe_document(rows)})


@router.post("/{task_id}/workflows/{workflow_id}/training", response_model=ApiEnvelope[TaskDocumentResponse])
def propose_training(request: Request, task_id: str, workflow_id: str, body: TaskTrainingProposalRequest):
    actor = _actor(request, "manage_training")
    if actor is None:
        return api_forbidden()
    return command(lambda: _invoke(request, task_id, workflow_id, lambda service:
        service.propose_training(workflow_id, actor=actor, **body.model_dump())))


@router.post("/{task_id}/workflows/{workflow_id}/training-approve", response_model=ApiEnvelope[TaskDocumentResponse])
def approve_training(request: Request, task_id: str, workflow_id: str, body: TaskRevisionRequest):
    actor = _actor(request, "manage_training", human=True)
    if actor is None:
        return api_forbidden()
    return command(lambda: _invoke(request, task_id, workflow_id, lambda service:
        service.approve_training(workflow_id, actor=actor, **body.model_dump())))


@router.get("/{task_id}/workflows/{workflow_id}/training-export", response_model=ApiEnvelope[TaskDocumentResponse])
def export_training(request: Request, task_id: str, workflow_id: str):
    if not require_permission(request, "manage_training"):
        return api_forbidden()
    return command(lambda: _invoke(request, task_id, workflow_id, lambda service:
        service.export_training_package(workflow_id)))


@router.post("/{task_id}/workflows/{workflow_id}/candidate", response_model=ApiEnvelope[TaskDocumentResponse])
def import_candidate(request: Request, task_id: str, workflow_id: str, body: TaskCandidateImportRequest):
    actor = _actor(request, "manage_models", human=True)
    if actor is None:
        return api_forbidden()
    return command(lambda: _invoke(request, task_id, workflow_id, lambda service:
        service.import_candidate(workflow_id, actor=actor, **body.model_dump())))
