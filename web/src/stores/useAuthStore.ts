import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api, unwrap } from '../api/client'
import { ApiError, type ApiResponse, type UserRole } from '../types/api'
import { logger } from '../utils/logger'

/**
 * Role identifiers used by both backend RBAC and client-side route guards.
 * Exported so component-level permission checks (e.g. button gating) can
 * reuse the same constants without re-declaring string literals.
 */
export const ROLES = {
  ADMIN: 'admin',
  ENGINEER: 'engineer',
  OPERATOR: 'operator',
  VIEWER: 'viewer',
} as const

export type Role = UserRole

export interface CurrentUser {
  username: string
  role: Role
  display_name?: string
}

function isRole(value: unknown): value is Role {
  return typeof value === 'string' && Object.values(ROLES).includes(value as Role)
}

function validateCurrentUser(value: unknown): CurrentUser {
  if (!value || typeof value !== 'object') throw new Error('Invalid /api/me response')
  const candidate = value as Record<string, unknown>
  if (typeof candidate.username !== 'string' || !candidate.username || !isRole(candidate.role)) {
    throw new Error('Invalid /api/me identity')
  }
  if (candidate.display_name !== undefined && typeof candidate.display_name !== 'string') {
    throw new Error('Invalid /api/me display name')
  }
  return {
    username: candidate.username,
    role: candidate.role,
    ...(candidate.display_name ? { display_name: candidate.display_name } : {}),
  }
}

// Cache TTL for /api/me. Within this window, calls are served from store state
// without hitting the backend. The router fires beforeEach on every navigation,
// so without this we'd hammer /api/me on every route change.
const CACHE_TTL_MS = 60_000

export const useAuthStore = defineStore('auth', () => {
  const currentUser = ref<CurrentUser | null>(null)
  const isLoading = ref(false)
  const lastFetchedAt = ref<number>(0)
  // Store-scoped coalescing avoids leaking a request/result across Pinia app
  // instances. Generation invalidation prevents a request started before
  // logout from restoring a stale role after clear().
  let inflight: Promise<CurrentUser | null> | null = null
  let requestGeneration = 0

  async function fetchCurrentUser(force = false): Promise<CurrentUser | null> {
    // A forced authorization refresh is authoritative. Ordinary navigations
    // must join it instead of serving an older privileged cache entry.
    if (inflight) return inflight

    // Serve from cache if fresh enough and caller didn't force a refresh.
    const now = Date.now()
    if (
      !force &&
      currentUser.value &&
      now - lastFetchedAt.value < CACHE_TTL_MS
    ) {
      return currentUser.value
    }

    isLoading.value = true
    const generation = requestGeneration
    const request = (async () => {
      try {
        const response = await api.get<ApiResponse<CurrentUser>>('/me')
        if (generation !== requestGeneration) return null
        const user = validateCurrentUser(unwrap(response))
        currentUser.value = user
        lastFetchedAt.value = Date.now()
        return user
      } catch (err: unknown) {
        if (generation !== requestGeneration) return null
        // Identity is authorization state. Any failed authoritative refresh
        // must fail closed instead of retaining a previously privileged user.
        currentUser.value = null
        lastFetchedAt.value = 0
        // 401: client.ts interceptor already triggered the /login redirect.
        // Just clear local state and return null so the guard can cancel nav.
        const status = err instanceof ApiError ? err.httpStatus : null
        if (status === 401) return null
        // Other errors (network, 5xx, etc.): log and degrade to null. We do
        // not throw — a thrown error inside beforeEach hangs the router.
        logger.error('[auth] fetchCurrentUser failed', err)
        return null
      }
    })()
    inflight = request
    const finish = () => {
      if (inflight === request) {
        isLoading.value = false
        inflight = null
      }
    }
    void request.then(finish, finish)

    return inflight
  }

  function clear() {
    requestGeneration += 1
    inflight = null
    currentUser.value = null
    lastFetchedAt.value = 0
    isLoading.value = false
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
