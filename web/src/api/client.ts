import axios, { type AxiosResponse } from 'axios'
import type { ApiResponse } from '../types/api'
import { ApiError } from '../types/api'

export const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const body = err.response?.data
    const msg = body?.msg ?? body?.error ?? err.message ?? '请求失败'
    console.error('[API]', msg)
    return Promise.reject(err)
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
