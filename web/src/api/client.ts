import axios, { type AxiosResponse } from 'axios'
import type { ApiResponse } from '../types/api'
import { ApiError } from '../types/api'
import { logger } from '../utils/logger'

export const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

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
    let msg: string = '请求失败'
    if (body && typeof body === 'object') {
      const b = body as Record<string, unknown>
      const pick = b.msg ?? b.error ?? b.detail ?? b.message
      if (typeof pick === 'string' && pick.trim().length > 0) msg = pick
    }
    if (msg === '请求失败' && typeof err?.message === 'string' && err.message) {
      msg = err.message
    }
    logger.debug('[API]', status, msg)
    return Promise.reject(new ApiError(status, msg, body))
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
  throw new ApiError(body.code, body.msg, body.data)
}

/** Shorthand for generic unwrap */
export const u = (res: AxiosResponse<ApiResponse<any>>) => unwrap<any>(res)
