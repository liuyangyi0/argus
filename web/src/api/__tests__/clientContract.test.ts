import type { AxiosResponse } from 'axios'
import { describe, expect, it } from 'vitest'

import { unwrap } from '../client'
import { ApiError, ErrorCode, type ApiResponse } from '../../types/api'

describe('ApiError contract', () => {
  it('separates HTTP status, business code, and details while retaining code compatibility', () => {
    const error = new ApiError({
      httpStatus: 409,
      businessCode: ErrorCode.CONFLICT,
      message: '发布冲突',
      details: { operation_id: 'op-1' },
    })

    expect(error.httpStatus).toBe(409)
    expect(error.businessCode).toBe(ErrorCode.CONFLICT)
    expect(error.details).toEqual({ operation_id: 'op-1' })
    expect(error.code).toBe(409)
    expect(error.message).toBe('发布冲突')
  })

  it('unwrap preserves the HTTP and business layers for envelope failures', () => {
    const response = {
      status: 404,
      data: {
        code: ErrorCode.NOT_FOUND,
        msg: '摄像头不存在',
        data: { camera_id: 'missing' },
      },
    } as AxiosResponse<ApiResponse<never>>

    try {
      unwrap(response)
      throw new Error('expected unwrap to throw')
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError)
      expect(error).toMatchObject({
        httpStatus: 404,
        businessCode: ErrorCode.NOT_FOUND,
        code: 404,
        details: { camera_id: 'missing' },
      })
    }
  })
})
