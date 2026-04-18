import { api, u } from './client'

export interface UserItem {
  username: string
  role: 'admin' | 'operator' | 'viewer'
  display_name: string
  active: boolean
  last_login?: string | null
  created_at?: string | null
}

export interface UserQueryParams {
  username?: string
  display_name?: string
  role?: string
}

export interface CreateUserPayload {
  username: string
  password: string
  role: UserItem['role']
  display_name: string
  active?: boolean
}

export interface UpdateUserPayload {
  password?: string
  role: UserItem['role']
  display_name: string
  active: boolean
}

export const getUsers = (params?: UserQueryParams) =>
  api.get('/users/json', { params }).then(u) as Promise<{ users: UserItem[] }>

export const createUser = (data: CreateUserPayload) =>
  api.post('/users/json', data).then(u) as Promise<{ user: UserItem }>

export const updateUser = (username: string, data: UpdateUserPayload) =>
  api.put(`/users/${username}/json`, data).then(u) as Promise<{ user: UserItem }>

export const deleteUser = (username: string) =>
  api.delete(`/users/${username}/json`).then(u) as Promise<{ username: string }>

export const toggleUserActive = (username: string) =>
  api.post(`/users/${username}/toggle-active/json`).then(u) as Promise<{ username: string; active: boolean }>

export const getAuditLogs = (params?: { page?: number; page_size?: number; user?: string; action?: string }) =>
  api.get('/audit/json', { params }).then(u)
