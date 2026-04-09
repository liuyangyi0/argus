import { api, u } from './client'

// ── Users ──
export const getUsers = () => api.get('/users/json').then(u)
export const createUser = (data: { username: string; password: string; role: string; display_name: string }) =>
  api.post('/users/json', data).then(u)
export const deleteUser = (username: string) => api.delete(`/users/${username}/json`).then(u)
export const toggleUserActive = (username: string) =>
  api.post(`/users/${username}/toggle-active/json`).then(u)

// ── Audit ──
export const getAuditLogs = (params?: { page?: number; page_size?: number; user?: string; action?: string }) =>
  api.get('/audit/json', { params }).then(u)
