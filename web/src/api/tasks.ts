import { api, u } from './client'

// ── Tasks ──
export const getTasks = () => api.get('/tasks/json').then(u)
export const dismissTask = (taskId: string) => api.delete(`/tasks/${taskId}`).then(u)
