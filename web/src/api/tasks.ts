import type { ApiResponse, TasksPayload } from '../types/api'
import { api, u, unwrap } from './client'

// ── Tasks ──
export const getTasks = () =>
  api.get<ApiResponse<TasksPayload>>('/tasks/json').then(unwrap)
export const dismissTask = (taskId: string) => api.delete(`/tasks/${taskId}`).then(u)
