import type { ApiResponse } from '../types/api'
import type {
  TaskDocumentResponse, TaskDocumentsResponse, TaskEventReviewRequest,
  TaskReleaseRequest, TaskReleaseEvidenceUpdateRequest, TaskReleaseReviewRequest,
  TaskReleaseRevokeRequest, TaskTrainingExecutionRequest, TaskTrainingCancelRequest,
  TaskReviewImportRequest, TaskTrainingCapabilitiesResponse,
} from '../types/generated-openapi'
import { api, unwrap } from './client'

export type { TaskTrainingCapabilitiesResponse } from '../types/generated-openapi'

const path = (taskId: string) => `/analysis-tasks/${encodeURIComponent(taskId)}`
const eventPath = (taskId: string, eventId: string) => `${path(taskId)}/events/${encodeURIComponent(eventId)}`
const releasePath = (taskId: string, releaseId: string) => `${path(taskId)}/releases/${encodeURIComponent(releaseId)}`
const workflowPath = (taskId: string, workflowId: string) => `${path(taskId)}/workflows/${encodeURIComponent(workflowId)}`
const keyHeader = (key: string) => ({ headers: { 'Idempotency-Key': key } })

export const getTaskEvents = (taskId: string) =>
  api.get<ApiResponse<TaskDocumentsResponse>>(`${path(taskId)}/events`).then(unwrap)
export const reviewTaskEvent = (taskId: string, eventId: string, body: TaskEventReviewRequest) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${eventPath(taskId, eventId)}/review`, body).then(unwrap)
export const getTaskEventReplay = (taskId: string, eventId: string) =>
  api.get<ApiResponse<TaskDocumentResponse>>(`${eventPath(taskId, eventId)}/replay`).then(unwrap)
export const taskEventEvidenceUrl = (taskId: string, eventId: string) => `/api${eventPath(taskId, eventId)}/evidence`

export const getTaskReleaseDossiers = (taskId: string) =>
  api.get<ApiResponse<TaskDocumentsResponse>>(`${path(taskId)}/releases`).then(unwrap)
export const getTaskLifecycleReadiness = (taskId: string, releaseId?: string) =>
  api.get<ApiResponse<TaskDocumentResponse>>(`${path(taskId)}/releases/readiness`, {
    params: releaseId ? { release_id: releaseId } : undefined,
  }).then(unwrap)
export const requestTaskRelease = (taskId: string, body: TaskReleaseRequest, key: string) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${path(taskId)}/releases`, body, keyHeader(key)).then(unwrap)
export const reviseTaskReleaseEvidence = (taskId: string, releaseId: string, body: TaskReleaseEvidenceUpdateRequest, key: string) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${releasePath(taskId, releaseId)}/evidence`, body, keyHeader(key)).then(unwrap)
export const reviewTaskRelease = (taskId: string, releaseId: string, body: TaskReleaseReviewRequest, key: string) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${releasePath(taskId, releaseId)}/review`, body, keyHeader(key)).then(unwrap)
export const revokeTaskRelease = (taskId: string, releaseId: string, body: TaskReleaseRevokeRequest, key: string) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${releasePath(taskId, releaseId)}/revoke`, body, keyHeader(key)).then(unwrap)

export const getTaskTrainingExecutions = (taskId: string) =>
  api.get<ApiResponse<TaskDocumentsResponse>>(`${path(taskId)}/training-executions`).then(unwrap)
export const getTaskTrainingCapabilities = (taskId: string) =>
  api.get<ApiResponse<TaskTrainingCapabilitiesResponse>>(`${path(taskId)}/training-capabilities`).then(unwrap)
export const submitTaskTrainingExecution = (taskId: string, workflowId: string, body: TaskTrainingExecutionRequest, key: string) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/training-executions`, body, keyHeader(key)).then(unwrap)
export const cancelTaskTrainingExecution = (taskId: string, jobId: string, body: TaskTrainingCancelRequest) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${path(taskId)}/training-executions/${encodeURIComponent(jobId)}/cancel`, body).then(unwrap)
export const exportTaskAnnotationsV2 = (taskId: string, workflowId: string, target: 'cvat' | 'fiftyone') =>
  api.get<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/annotation-export`, { params: { target } }).then(unwrap)
export const importTaskAnnotationsV2 = (taskId: string, workflowId: string, body: TaskReviewImportRequest) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/annotation-import`, body).then(unwrap)
