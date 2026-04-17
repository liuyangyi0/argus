import { ApiError } from '../types/api'

interface AxiosLikeError {
  response?: {
    status?: number
    data?: {
      msg?: unknown
      error?: unknown
      detail?: unknown
      message?: unknown
      code?: unknown
    } | unknown
  }
  message?: string
}

function hasResponse(e: unknown): e is AxiosLikeError {
  return !!e && typeof e === 'object' && 'response' in e
}

/**
 * Extract a user-friendly error message from an axios error, ApiError, or
 * generic exception.
 *
 * Priority (matches the backend unified envelope {code, msg, data}):
 *   data.msg > data.error > data.detail > data.message > e.message > fallback.
 */
export function extractErrorMessage(e: unknown, fallback = '操作失败'): string {
  if (e instanceof ApiError) return e.message || fallback
  if (hasResponse(e)) {
    const data = e.response?.data
    if (data && typeof data === 'object') {
      const d = data as Record<string, unknown>
      const pick = d.msg ?? d.error ?? d.detail ?? d.message
      if (typeof pick === 'string' && pick.trim().length > 0) return pick
    }
    if (typeof e.message === 'string' && e.message.length > 0) return e.message
    return fallback
  }
  if (e instanceof Error && e.message) return e.message
  return fallback
}

/**
 * Wrap an arbitrary thrown value into an `ApiError` so call sites can rely on
 * a consistent shape without using `any`. HTTP status is preserved on the
 * `code` field when available; otherwise `-1` (transport/unknown).
 */
export function toApiError(e: unknown, fallback = '请求失败'): ApiError {
  if (e instanceof ApiError) return e
  if (hasResponse(e)) {
    const status = e.response?.status ?? -1
    const msg = extractErrorMessage(e, fallback)
    const data = e.response?.data
    return new ApiError(status, msg, data)
  }
  if (e instanceof Error) return new ApiError(-1, e.message || fallback)
  return new ApiError(-1, fallback)
}
