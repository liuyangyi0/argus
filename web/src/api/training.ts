import { api, u } from './client'

// ── Training (legacy baseline endpoints) ──
export const startTraining = (data: FormData) => api.post('/baseline/train', data).then(u)
export const getTrainingHistory = (params?: Record<string, any>) =>
  api.get('/baseline/training-history/json', { params }).then(u)
export const compareModels = (data: { old_record_id: number; new_record_id: number }) =>
  api.post('/baseline/compare', data, { timeout: 120000 }).then(u)

// ── Training Jobs ──
export const getTrainingJobs = (params?: Record<string, any>) =>
  api.get('/training-jobs/json', { params }).then(u)
export const getTrainingJob = (jobId: string) => api.get(`/training-jobs/${jobId}`).then(u)
export const createTrainingJob = (data: {
  job_type: string; camera_id?: string; zone_id?: string;
  model_type?: string; trigger_type?: string; triggered_by?: string;
  hyperparameters?: Record<string, any>;
}) => api.post('/training-jobs/', data).then(u)
export const confirmTrainingJob = (jobId: string, data?: { confirmed_by?: string }) =>
  api.post(`/training-jobs/${jobId}/confirm`, data || {}).then(u)
export const rejectTrainingJob = (jobId: string, data?: { rejected_by?: string; reason?: string }) =>
  api.post(`/training-jobs/${jobId}/reject`, data || {}).then(u)
export const getBackbones = () => api.get('/training-jobs/backbones/json').then(u)
