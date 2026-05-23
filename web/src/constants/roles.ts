import type { UserRole } from '../types/api'

export interface RoleOption {
  label: string
  value: UserRole
  color: string
}

export const USER_ROLE_OPTIONS = [
  { label: '管理员', value: 'admin', color: 'red' },
  { label: '工程师', value: 'engineer', color: 'purple' },
  { label: '操作员', value: 'operator', color: 'blue' },
  { label: '观察者', value: 'viewer', color: 'default' },
] satisfies RoleOption[]

export const ROLE_META = Object.fromEntries(
  USER_ROLE_OPTIONS.map((item) => [item.value, {
    color: item.color,
    label: item.label,
  }]),
) as Record<UserRole, { color: string; label: string }>
