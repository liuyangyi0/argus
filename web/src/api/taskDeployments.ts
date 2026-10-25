import { api, unwrap } from './client'
import type { ApiResponse } from '../types/api'
import type {
  TaskConfigurationRestoreRequest, TaskDeploymentRequest, TaskDocumentResponse, TaskDocumentsResponse, CameraSourceResponse,
} from '../types/generated-openapi'

const base = (taskId: string) => `/analysis-tasks/${encodeURIComponent(taskId)}`
export const getTaskDeployment = (taskId: string) =>
  api.get<ApiResponse<TaskDocumentResponse>>(`${base(taskId)}/deployment`).then(unwrap)
export const getTaskDeploymentOperations = (taskId: string) =>
  api.get<ApiResponse<TaskDocumentsResponse>>(`${base(taskId)}/deployment-operations`).then(unwrap)
export const requestTaskDeployment = (taskId: string, body: TaskDeploymentRequest, key: string) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${base(taskId)}/deployments`, body,
    { headers: { 'Idempotency-Key': key } }).then(unwrap)
export const reconcileTaskDeployment = (taskId: string) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${base(taskId)}/deployment/reconcile`).then(unwrap)
export const restoreTaskConfiguration = (taskId: string, body: TaskConfigurationRestoreRequest, key: string) =>
  api.post<ApiResponse<TaskDocumentResponse>>(`${base(taskId)}/configuration-restores`, body,
    { headers: { 'Idempotency-Key': key } }).then(unwrap)

const source = (cameraId: string) => `/cameras/${encodeURIComponent(cameraId)}/source`
export const getCameraSource = (cameraId: string) =>
  api.get<ApiResponse<CameraSourceResponse>>(source(cameraId)).then(unwrap)
export const commandCameraSource = (cameraId: string, action: 'start' | 'stop') =>
  api.post<ApiResponse<CameraSourceResponse>>(`${source(cameraId)}/${action}`).then(unwrap)
