import type { ApiResponse } from '../types/api'
import { api, unwrap } from './client'

export interface SceneProfileSummary {
  profile_id: string
  approved: boolean
  active: boolean
  model_version_id: string | null
}

export interface SceneWorkflow {
  workflow_id: string
  profile_id: string
  state: string
  created_by: string
  created_at: string
  reference_sha256: string
  baseline_schema: string | null
  training_ready: boolean
  train_count: number
  calibration_count: number
  baseline_task_id: string | null
  training_job_id: string | null
  training_status: string | null
  model_version_id: string | null
  approved: boolean
  active: boolean
  error: string | null
  blockers: string[]
}

export interface SceneReference {
  path: string
  sha256: string
}

export interface SceneWorkflowSnapshot {
  camera_id: string
  workflows: SceneWorkflow[]
  issues?: { workflow_id: string; message: string }[]
  profiles: SceneProfileSummary[]
  references: SceneReference[]
  source_root_label: string
  active_profile_revision: string | null
  requirements: {
    baseline_schema: 'argus.scene-profile-baseline.v3'
    min_sources: number
    min_train_images: number
    min_calibration_images: number
    production_gate_unchanged: true
  }
}

export interface CreateSceneWorkflowRequest {
  workflow_id: string
  reference_path: string
  reference_sha256: string
  normal_reference_confirmed: true
}

export interface SceneBaselineSource {
  video: string
  annotations: string
}

export interface SceneBaselineJobRequest {
  sources: SceneBaselineSource[]
  sample_every: number
  max_images: number
}

function cameraPath(cameraId: string): string {
  return `/scene-workflow/${encodeURIComponent(cameraId)}`
}

function workflowPath(cameraId: string, workflowId: string): string {
  return `${cameraPath(cameraId)}/workflows/${encodeURIComponent(workflowId)}`
}

export const getSceneWorkflow = (cameraId: string): Promise<SceneWorkflowSnapshot> =>
  api.get<ApiResponse<SceneWorkflowSnapshot>>(cameraPath(cameraId)).then(unwrap)

export const createSceneWorkflow = (
  cameraId: string,
  data: CreateSceneWorkflowRequest,
): Promise<SceneWorkflow> =>
  api.post<ApiResponse<SceneWorkflow>>(`${cameraPath(cameraId)}/workflows`, data).then(unwrap)

export const startSceneBaselineJob = (
  cameraId: string,
  workflowId: string,
  data: SceneBaselineJobRequest,
): Promise<{ task_id: string }> =>
  api.post<ApiResponse<{ task_id: string }>>(`${workflowPath(cameraId, workflowId)}/baseline-jobs`, data).then(unwrap)

export const createSceneTrainingJob = (
  cameraId: string,
  workflowId: string,
  data: { model_type: 'patchcore'; image_size: number },
): Promise<{ job_id: string; status: 'pending_confirmation' }> =>
  api.post<ApiResponse<{ job_id: string; status: 'pending_confirmation' }>>(
    `${workflowPath(cameraId, workflowId)}/training-jobs`, data,
  ).then(unwrap)

export const approveSceneProfile = (
  cameraId: string,
  workflowId: string,
): Promise<{ profile_id: string }> =>
  api.post<ApiResponse<{ profile_id: string }>>(
    `${workflowPath(cameraId, workflowId)}/approve-profile`, { operator_approved: true },
  ).then(unwrap)

export const activateSceneProfile = (
  cameraId: string,
  workflowId: string,
  expectedActiveProfileId: string | null,
  expectedActiveProfileRevision: string | null,
): Promise<{ profile_id: string }> =>
  api.post<ApiResponse<{ profile_id: string }>>(
    `${workflowPath(cameraId, workflowId)}/activate-profile`, {
      acknowledge_runtime_unchanged: true,
      expected_active_profile_id: expectedActiveProfileId,
      expected_active_profile_revision: expectedActiveProfileRevision,
    },
  ).then(unwrap)
