import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '../api'
import { logger } from '../utils/logger'

/**
 * Role identifiers used by both backend RBAC and client-side route guards.
 * Exported so component-level permission checks (e.g. button gating) can
 * reuse the same constants without re-declaring string literals.
 */
export const ROLES = {
  ADMIN: 'admin',
  OPERATOR: 'operator',
  VIEWER: 'viewer',
} as const

export type Role = (typeof ROLES)[keyof typeof ROLES]

export interface CurrentUser {
  username: string
  role: Role
  display_name?: string
}

// Cache TTL for /api/me. Within this window, calls are served from store state
// without hitting the backend. The router fires beforeEach on every navigation,
// so without this we'd hammer /api/me on every route change.
const CACHE_TTL_MS = 60_000

// Module-scoped in-flight promise. When N concurrent navigations all need the
// current user (e.g. parent + child route both have requiresAuth), they all
// await the same network call instead of issuing N parallel requests.
let inflight: Promise<CurrentUser | null> | null = null

export const useAuthStore = defineStore('auth', () => {
  const currentUser = ref<CurrentUser | null>(null)
  const isLoading = ref(false)
  const lastFetchedAt = ref<number>(0)

  async function fetchCurrentUser(force = false): Promise<CurrentUser | null> {
    // Serve from cache if fresh enough and caller didn't force a refresh.
    const now = Date.now()
    if (
      !force &&
      currentUser.value &&
      now - lastFetchedAt.value < CACHE_TTL_MS
    ) {
      return currentUser.value
    }

    // Coalesce concurrent calls: all callers await the same promise.
    if (inflight) return inflight

    isLoading.value = true
    inflight = (async () => {
      try {
        const res = await api.get<{
          username: string
          role: string
          display_name?: string
        }>('/me')
        const data = res.data
        const user: CurrentUser = {
          username: data.username,
          role: data.role as Role,
          display_name: data.display_name,
        }
        currentUser.value = user
        lastFetchedAt.value = Date.now()
        return user
      } catch (err: any) {
        // 401: client.ts interceptor already triggered the /login redirect.
        // Just clear local state and return null so the guard can cancel nav.
        const status = err?.status ?? err?.response?.status
        if (status === 401) {
          currentUser.value = null
          return null
        }
        // Other errors (network, 5xx, etc.): log and degrade to null. We do
        // not throw — a thrown error inside beforeEach hangs the router.
        logger.error('[auth] fetchCurrentUser failed', err)
        return null
      } finally {
        isLoading.value = false
        inflight = null
      }
    })()

    return inflight
  }

  function clear() {
    currentUser.value = null
    lastFetchedAt.value = 0
  }

  function hasRole(roles: string[]): boolean {
    if (!currentUser.value) return false
    return roles.includes(currentUser.value.role)
  }

  return {
    currentUser,
    isLoading,
    lastFetchedAt,
    fetchCurrentUser,
    clear,
    hasRole,
  }
})
