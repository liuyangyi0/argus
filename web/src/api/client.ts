import axios, { type AxiosResponse } from 'axios'
import { message } from 'ant-design-vue'
import type { ApiResponse } from '../types/api'
import { ApiError } from '../types/api'
import { logger } from '../utils/logger'

export const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

// Debounce: avoid duplicate toasts/redirects when many in-flight requests fail at once.
let isRedirecting = false
let forbiddenToastAt = 0

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? value as Record<string, unknown> : null
}

// The interceptor used to `message.error(...)` here — which collided with the
// 36+ call sites that also toast in their own catch blocks, producing double
// popups. It is now silent: it logs and normalizes failures into a rejected
// `ApiError` (status on `code`, server's `msg`/`error`/`detail`/`message` as
// the message). Call sites should surface errors via `extractErrorMessage(e)`
// from `utils/error.ts`, which understands this shape.
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const body = err?.response?.data
    const status = err?.response?.status ?? -1
    const bodyRecord = asRecord(body)
    const businessCode = typeof bodyRecord?.code === 'number' ? bodyRecord.code : null
    const details = bodyRecord?.data ?? bodyRecord?.details ?? bodyRecord?.detail
    let msg: string = '请求失败'
    if (bodyRecord) {
      const pick = bodyRecord.msg ?? bodyRecord.error ?? bodyRecord.detail ?? bodyRecord.message
      if (typeof pick === 'string' && pick.trim().length > 0) msg = pick
    }
    if (msg === '请求失败' && typeof err?.message === 'string' && err.message) {
      msg = err.message
    }
    logger.debug('[API]', status, msg)
    if (status === 401) {
      // Backend serves SSR /login — no SPA route exists, must full-page redirect.
      if (!isRedirecting) {
        isRedirecting = true
        message.warning('登录已过期,请重新登录')
        const next = encodeURIComponent(window.location.pathname + window.location.search)
        window.location.href = `/login?next=${next}`
      }
      return Promise.reject(new ApiError({
        httpStatus: status,
        businessCode,
        message: msg,
        details,
        raw: body,
      }))
    }
    if (status === 403) {
      // Coalesce 403 toasts within 1s window so a burst doesn't spam.
      const now = Date.now()
      if (now - forbiddenToastAt > 1000) {
        forbiddenToastAt = now
        message.error('权限不足')
      }
      return Promise.reject(new ApiError({
        httpStatus: status,
        businessCode,
        message: msg,
        details,
        raw: body,
      }))
    }
    return Promise.reject(new ApiError({
      httpStatus: status > 0 ? status : null,
      businessCode,
      message: msg,
      details,
      raw: body,
    }))
  }
)

/**
 * Unwrap the unified response envelope {code, msg, data}.
 * code=0: success — returns data directly.
 * code>0: throws ApiError.
 */
export function unwrap<T>(res: AxiosResponse<ApiResponse<T>>): T {
  const body = res.data
  if (body.code === 0) return body.data
  throw new ApiError({
    httpStatus: res.status || null,
    businessCode: body.code,
    message: body.msg,
    details: body.data,
    raw: body.data,
  })
}

/** Shorthand for generic unwrap; unconstrained callers resolve to `unknown`. */
export const u = <T = unknown>(res: AxiosResponse<ApiResponse<T>>): T => unwrap<T>(res)
