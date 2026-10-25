import type { ApiResponse } from '../types/api'
import type {
  AnalysisTaskResponse,
  AnalysisTasksResponse,
  AnalysisTaskTemplatesResponse,
  TaskBundleCreateRequest, TaskCreateRequest, TaskUpdateRequest,
  TaskModelRegisterRequest, TaskDocumentResponse, TaskDocumentsResponse,
  TaskResultsResponse, TaskReleaseReadinessResponse,
  TaskCollectionRequest, TaskRevisionRequest, TaskReviewRequest, TaskReviewImportRequest,
  TaskTrainingProposalRequest, TaskCandidateImportRequest,
} from '../types/generated-openapi'
import { api, unwrap } from './client'

export type AnalysisTask = AnalysisTaskResponse

export const getAnalysisTasks = (cameraId?: string) =>
  api.get<ApiResponse<AnalysisTasksResponse>>('/analysis-tasks', {
    params: cameraId ? { camera_id: cameraId } : undefined,
  }).then(unwrap)

export const getAnalysisTask = (taskId: string) =>
  api.get<ApiResponse<AnalysisTaskResponse>>(
    `/analysis-tasks/${encodeURIComponent(taskId)}`,
  ).then(unwrap)

export const getAnalysisTaskTemplates = () =>
  api.get<ApiResponse<AnalysisTaskTemplatesResponse>>('/analysis-task-templates').then(unwrap)

const path = (taskId: string) => `/analysis-tasks/${encodeURIComponent(taskId)}`
const keyHeader = (key: string) => ({ headers: { 'Idempotency-Key': key } })

export const createAnalysisTask = (body: TaskCreateRequest, key: string) =>
  api.post<ApiResponse<AnalysisTask>>( '/analysis-tasks', body, keyHeader(key)).then(unwrap)
export const updateAnalysisTask = (taskId: string, body: TaskUpdateRequest, key: string) =>
  api.put<ApiResponse<AnalysisTask>>(path(taskId), body, keyHeader(key)).then(unwrap)
export const archiveAnalysisTask = (taskId: string, expectedRevision: number, key: string) =>
  api.post<ApiResponse<AnalysisTask>>(`${path(taskId)}/archive`, { expected_revision: expectedRevision }, keyHeader(key)).then(unwrap)
export const getTaskModels = () =>
  api.get<ApiResponse<TaskDocumentsResponse>>('/analysis-task-models').then(unwrap)
export const registerTaskModel = (body: TaskModelRegisterRequest, key: string) =>
  api.post<ApiResponse<TaskDocumentResponse>>('/analysis-task-models', body, keyHeader(key)).then(unwrap)
export const getCompatibleTaskModels = (taskId: string) =>
  api.get<ApiResponse<TaskDocumentsResponse>>(`${path(taskId)}/compatible-models`).then(unwrap)
export const getTaskModelCompatibility = (taskId: string) =>
  api.get<ApiResponse<TaskDocumentsResponse>>(`${path(taskId)}/model-compatibility`).then(unwrap)
export const getTaskBundles = (taskId: string) =>
  api.get<ApiResponse<TaskDocumentsResponse>>(`${path(taskId)}/bundles`).then(unwrap)
export const freezeTaskBundle = (taskId: string, body: TaskBundleCreateRequest, key: string) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${path(taskId)}/bundles`, body, keyHeader(key)).then(unwrap)
export const startAnalysisTask = (taskId: string, bundleId: string) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${path(taskId)}/runtime/start`, { bundle_id: bundleId }).then(unwrap)
export const stopAnalysisTask = (taskId: string) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${path(taskId)}/runtime/stop`).then(unwrap)
export const getTaskResults = (taskId: string) =>
  api.get<ApiResponse<TaskResultsResponse>>(`${path(taskId)}/results`).then(unwrap)
export const getTaskReleaseReadiness = (taskId: string) =>
  api.get<ApiResponse<TaskReleaseReadinessResponse>>(`${path(taskId)}/release-readiness`).then(unwrap)

const workflowPath = (taskId: string, workflowId: string) => `${path(taskId)}/workflows/${encodeURIComponent(workflowId)}`
export const getTaskWorkflows = (taskId: string) =>
  api.get<ApiResponse<TaskDocumentsResponse>>(`${path(taskId)}/workflows`).then(unwrap)
export const createTaskWorkflow = (taskId: string, key: string) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${path(taskId)}/workflows`, {}, keyHeader(key)).then(unwrap)
export const collectTaskData = (taskId: string, workflowId: string, body: TaskCollectionRequest) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/collect`, body).then(unwrap)
export const stopTaskCollection = (taskId: string, workflowId: string, body: TaskRevisionRequest) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/stop`, body).then(unwrap)
export const reviewTaskSample = (taskId: string, workflowId: string, body: TaskReviewRequest) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/review`, body).then(unwrap)
export const freezeTaskDataset = (taskId: string, workflowId: string, body: TaskRevisionRequest) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/freeze`, body).then(unwrap)
export const exportTaskReview = (taskId: string, workflowId: string, target: 'cvat' | 'fiftyone') =>
  api.get<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/review-export`, { params: { target } }).then(unwrap)
export const importTaskReview = (taskId: string, workflowId: string, body: TaskReviewImportRequest) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/review-import`, body).then(unwrap)
export const suggestTaskAnnotations = (taskId: string, workflowId: string, body: TaskRevisionRequest) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/preannotations`, body).then(unwrap)
export const getTaskTrainingRecipes = (taskId: string) =>
  api.get<ApiResponse<TaskDocumentsResponse>>(`${path(taskId)}/training-recipes`).then(unwrap)
export const proposeTaskTraining = (taskId: string, workflowId: string, body: TaskTrainingProposalRequest) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/training`, body).then(unwrap)
export const approveTaskTraining = (taskId: string, workflowId: string, body: TaskRevisionRequest) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/training-approve`, body).then(unwrap)
export const exportTaskTraining = (taskId: string, workflowId: string) =>
  api.get<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/training-export`).then(unwrap)
export const importTaskCandidate = (taskId: string, workflowId: string, body: TaskCandidateImportRequest) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${workflowPath(taskId, workflowId)}/candidate`, body).then(unwrap)
export const taskSampleImageUrl = (taskId: string, workflowId: string, sampleId: string) =>
  `/api${workflowPath(taskId, workflowId)}/samples/${encodeURIComponent(sampleId)}/image`
