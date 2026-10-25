"""Durable, task-scoped shadow deployment commands; never production release."""

from fastapi import APIRouter, Request

from argus.dashboard.api_response import (
    ApiEnvelope, api_forbidden, api_unavailable,
)
from argus.dashboard.auth import require_permission
from argus.dashboard.routes.task_platform import CommandKey, _actor, command, safe_document
from argus.dashboard.routes.analysis_tasks import _service
from argus.core.task_execution import TaskExecutionError
from argus.dashboard.task_platform_contracts import (
    TaskConfigurationRestoreRequest, TaskDeploymentRequest, TaskDocumentResponse, TaskDocumentsResponse,
)
from argus.storage.analysis_tasks import (
    AnalysisTaskConflictError, AnalysisTaskIntegrityError, AnalysisTaskValidationError,
)

router = APIRouter()


@router.post("/{task_id}/configuration-restores", response_model=ApiEnvelope[TaskDocumentResponse], status_code=201)
def restore_configuration(request: Request, task_id: str, body: TaskConfigurationRestoreRequest,
                          idempotency_key: CommandKey):
    actor = _actor(request, "manage_models", human=True)
    if actor is None:
        return api_forbidden("恢复配置需要人工管理身份")
    service = _service(request)
    executor = getattr(request.app.state, "task_execution", None)
    if service is None or executor is None:
        return api_unavailable()
    arguments = {**body.model_dump(), "requested_by": actor, "idempotency_key": idempotency_key}

    def restore():
        committed = service.get_bundle_configuration_restore_receipt(task_id, **arguments)
        if committed is not None:
            return committed
        try:
            return executor.mutate_if_stopped(task_id, lambda: service.restore_bundle_configuration(task_id, **arguments))
        except TaskExecutionError as exc:
            if exc.code == "BUSY":
                committed = service.get_bundle_configuration_restore_receipt(task_id, **arguments)
                if committed is not None:
                    return committed
            raise

    def confirmed_restore():
        try:
            return restore()
        except (LookupError, AnalysisTaskConflictError, AnalysisTaskIntegrityError,
                AnalysisTaskValidationError, TaskExecutionError):
            raise
        except Exception as exc:
            raise _UnconfirmedDeploymentRequest() from exc

    return command(confirmed_restore, status_code=201)


class _UnconfirmedDeploymentRequest(Exception):
    """A failed acknowledgement is not a definitely rejected domain command."""


def _services(request: Request):
    return (getattr(request.app.state, "task_deployments", None),
            getattr(request.app.state, "task_deployment_coordinator", None))


@router.post("/{task_id}/deployments", response_model=ApiEnvelope[TaskDocumentResponse], status_code=202)
def request_deployment(request: Request, task_id: str, body: TaskDeploymentRequest, idempotency_key: CommandKey):
    actor = _actor(request, "edit_config")
    if actor is None:
        return api_forbidden()
    service, coordinator = _services(request)
    if service is None or coordinator is None:
        return api_unavailable("持久化任务部署服务不可用")

    def submit():
        try:
            operation = service.request(task_id, **body.model_dump(), requested_by=actor,
                                        idempotency_key=idempotency_key)
        except (LookupError, AnalysisTaskConflictError, AnalysisTaskIntegrityError, AnalysisTaskValidationError):
            raise  # Explicit domain rejections are safe to render as 4xx.
        except Exception as exc:
            # The shared command mapper treats generic RuntimeError as 409.
            # A commit may already be durable when an ACK is lost, so retain a
            # 5xx/unknown result and make the client retry this exact key/body.
            raise _UnconfirmedDeploymentRequest() from exc
        # The durable intent has already committed. A wake-up failure must not
        # cause a new command: background reconciliation owns the same receipt.
        try:
            coordinator.submit(operation["operation_id"])
        except Exception:
            pass
        return operation

    return command(submit, status_code=202)


@router.get("/{task_id}/deployment", response_model=ApiEnvelope[TaskDocumentResponse])
def deployment(request: Request, task_id: str):
    if not require_permission(request, "read_cameras"):
        return api_forbidden()
    service, _ = _services(request)
    if service is None:
        return api_unavailable()
    return command(lambda: service.get(task_id))


@router.get("/{task_id}/deployment-operations", response_model=ApiEnvelope[TaskDocumentsResponse])
def operations(request: Request, task_id: str):
    if not require_permission(request, "read_cameras"):
        return api_forbidden()
    service, _ = _services(request)
    if service is None:
        return api_unavailable()
    return command(lambda: service.list_operations(task_id), model=TaskDocumentsResponse,
                   transform=lambda rows: {"documents": safe_document(rows)})


@router.get("/{task_id}/deployment-operations/{operation_id}", response_model=ApiEnvelope[TaskDocumentResponse])
def operation(request: Request, task_id: str, operation_id: str):
    if not require_permission(request, "read_cameras"):
        return api_forbidden()
    service, _ = _services(request)
    if service is None:
        return api_unavailable()

    def scoped():
        result = service.get_operation(operation_id)
        if result is None or result.get("task_id") != task_id:
            raise LookupError("operation does not belong to task")
        return result

    return command(scoped)


@router.post("/{task_id}/deployment/reconcile", response_model=ApiEnvelope[TaskDocumentResponse])
def reconcile(request: Request, task_id: str):
    actor = _actor(request, "edit_config")
    if actor is None:
        return api_forbidden()
    service, coordinator = _services(request)
    if service is None or coordinator is None:
        return api_unavailable()
    return command(lambda: coordinator.reconcile(task_id))
