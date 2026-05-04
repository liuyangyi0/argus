import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export type ErrorSeverity = 'info' | 'warning' | 'error' | 'critical'

export interface SystemError {
  severity: ErrorSeverity
  source: string
  code: string
  message: string
  context: Record<string, unknown>
  timestamp: string  // ISO-8601
  // 客户端注入字段
  id: string         // 客户端生成的 uuid(用于 list key + 已读状态追踪)
  read: boolean
  receivedAt: number // Date.now()
}

const MAX_KEEP = 100  // 内存最多保留 100 条,溢出丢最旧
const NOTIFY_SEVERITIES: ErrorSeverity[] = ['error', 'critical']

export const useErrorStore = defineStore('errors', () => {
  const errors = ref<SystemError[]>([])
  const drawerOpen = ref(false)

  // 未读数:仅统计 error/critical(info/warning 不打扰用户)
  const unreadCount = computed(() =>
    errors.value.filter(e => !e.read && NOTIFY_SEVERITIES.includes(e.severity)).length
  )

  // 全部未读数(包括 info/warning),给 drawer 用
  const totalUnread = computed(() =>
    errors.value.filter(e => !e.read).length
  )

  function pushError(payload: any) {
    if (!payload || typeof payload !== 'object') return
    const now = Date.now()
    const evt: SystemError = {
      severity: (payload.severity || 'error') as ErrorSeverity,
      source: String(payload.source || 'unknown'),
      code: String(payload.code || 'unknown'),
      message: String(payload.message || ''),
      context: (payload.context && typeof payload.context === 'object') ? payload.context : {},
      timestamp: String(payload.timestamp || new Date().toISOString()),
      id: `${now}-${Math.random().toString(36).slice(2, 8)}`,
      read: false,
      receivedAt: now,
    }
    errors.value.unshift(evt)
    if (errors.value.length > MAX_KEEP) {
      errors.value.length = MAX_KEEP
    }
  }

  function markAllRead() {
    errors.value.forEach(e => { e.read = true })
  }

  function markRead(id: string) {
    const evt = errors.value.find(e => e.id === id)
    if (evt) evt.read = true
  }

  function clear() {
    errors.value = []
  }

  function openDrawer() {
    drawerOpen.value = true
  }

  function closeDrawer() {
    drawerOpen.value = false
  }

  return {
    errors,
    drawerOpen,
    unreadCount,
    totalUnread,
    pushError,
    markAllRead,
    markRead,
    clear,
    openDrawer,
    closeDrawer,
  }
})
