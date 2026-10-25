import { beforeEach, describe, expect, it, vi } from 'vitest'

const client = vi.hoisted(() => ({ post: vi.fn(), get: vi.fn() }))
vi.mock('../client', () => ({
  api: client,
  unwrap: (response: { data: { data: unknown } }) => response.data.data,
  u: (response: { data: { data: unknown } }) => response.data.data,
}))

function result(status: 'pending' | 'applied' | 'failed', key?: string) {
  return { data: { data: {
    operation: {
      operation_id: 'op-1', model_version_id: 'model', status,
      idempotency_key: key,
    },
    model: {}, runtime_synced: status === 'applied',
  } } }
}

describe('release request idempotency', () => {
  beforeEach(() => {
    client.post.mockReset()
    client.get.mockReset()
    sessionStorage.clear()
    vi.resetModules()
  })

  it('reuses the key after a transport timeout, including a component/module reload', async () => {
    client.post.mockRejectedValueOnce(new Error('timeout'))
    let models = await import('../models')
    const payload = { target_stage: 'shadow', expected_revision: 3 }
    await expect(models.promoteModel('model-timeout', payload)).rejects.toThrow('timeout')
    const firstKey = client.post.mock.calls[0][1].idempotency_key
    expect(firstKey).toMatch(/^[0-9a-f-]{36}$/)

    vi.resetModules()
    models = await import('../models')
    client.post.mockResolvedValueOnce(result('pending'))
    await models.promoteModel('model-timeout', payload)
    expect(client.post.mock.calls[1][1].idempotency_key).toBe(firstKey)
  })

  it('reuses a rollback key and both revisions after a timeout and module reload', async () => {
    client.post.mockRejectedValueOnce(new Error('timeout'))
    let models = await import('../models')
    const payload = {
      source_model_version_id: 'production-1',
      source_release_revision: 7,
      expected_revision: 4,
      reason: 'Restore verified previous version',
    }
    await expect(models.rollbackModel('retired-1', payload)).rejects.toThrow('timeout')
    const firstKey = client.post.mock.calls[0][1].idempotency_key
    expect(firstKey).toMatch(/^[0-9a-f-]{36}$/)
    expect(client.post.mock.calls[0]).toEqual([
      '/models/retired-1/rollback', { ...payload, idempotency_key: firstKey },
    ])

    vi.resetModules()
    models = await import('../models')
    client.post.mockResolvedValueOnce(result('pending'))
    await models.rollbackModel('retired-1', payload)
    expect(client.post.mock.calls[1]).toEqual(client.post.mock.calls[0])
    expect(payload).not.toHaveProperty('idempotency_key')
  })

  it.each([
    ['source model identity', { source_model_version_id: 'production-2' }],
    ['source release revision', { source_release_revision: 8 }],
  ])('uses a separate rollback key when the %s changes', async (_name, change) => {
    const models = await import('../models')
    client.post.mockResolvedValue(result('pending'))
    const payload = {
      source_model_version_id: 'production-1',
      source_release_revision: 7,
      expected_revision: 4,
      reason: 'Restore verified previous version',
    }
    await models.rollbackModel('retired-1', payload)
    const initialKey = client.post.mock.calls[0][1].idempotency_key
    const changedPayload = { ...payload, ...change }
    await models.rollbackModel('retired-1', changedPayload)
    const changedKey = client.post.mock.calls[1][1].idempotency_key
    expect(changedKey).toMatch(/^[0-9a-f-]{36}$/)
    expect(changedKey).not.toBe(initialKey)
    expect(client.post.mock.calls[1][1]).toEqual({
      ...changedPayload, idempotency_key: changedKey,
    })

    await models.rollbackModel('retired-1', payload)
    await models.rollbackModel('retired-1', changedPayload)
    expect(client.post.mock.calls[2][1].idempotency_key).toBe(initialKey)
    expect(client.post.mock.calls[3][1].idempotency_key).toBe(changedKey)
  })

  it('retains a pending key and clears it when polling confirms a terminal result', async () => {
    const models = await import('../models')
    client.post.mockResolvedValue(result('pending'))
    const payload = { target_stage: 'shadow', expected_revision: 3 }
    await models.promoteModel('model-pending', payload)
    await models.promoteModel('model-pending', payload)
    const key = client.post.mock.calls[0][1].idempotency_key
    expect(client.post.mock.calls[1][1].idempotency_key).toBe(key)

    client.get.mockResolvedValue({ data: { data: { operation: {
      status: 'failed', idempotency_key: key,
    } } } })
    await models.getReleaseOperation('op-1')
    await models.promoteModel('model-pending', payload)
    expect(client.post.mock.calls[2][1].idempotency_key).not.toBe(key)
  })

  it('uses distinct keys when revision or release intent changes', async () => {
    const models = await import('../models')
    client.post.mockResolvedValue(result('pending'))
    await models.promoteModel('model-intent', { target_stage: 'shadow', expected_revision: 3 })
    await models.promoteModel('model-intent', { target_stage: 'shadow', expected_revision: 4 })
    await models.retireModel('model-intent', { expected_revision: 4 })
    const keys = client.post.mock.calls.map(call => call[1].idempotency_key)
    expect(new Set(keys).size).toBe(3)
  })

  it('clears an accepted key when a terminal WebSocket operation is acknowledged', async () => {
    const models = await import('../models')
    client.post.mockResolvedValue(result('pending'))
    const payload = { target_stage: 'shadow', expected_revision: 3 }
    await models.promoteModel('model-ws', payload)
    const key = client.post.mock.calls[0][1].idempotency_key
    models.acknowledgeReleaseOperation({
      operation_id: 'op-ws', model_version_id: 'model-ws', camera_id: 'cam',
      from_stage: 'candidate', target_stage: 'shadow', expected_revision: 3,
      applied_revision: 4, status: 'applied', requested_by: 'tester', idempotency_key: key,
    })
    await models.promoteModel('model-ws', payload)
    expect(client.post.mock.calls[1][1].idempotency_key).not.toBe(key)
  })

  it('keeps the key after a server error whose transaction outcome is unknown', async () => {
    const models = await import('../models')
    const { ApiError } = await import('../../types/api')
    client.post.mockRejectedValueOnce(new ApiError({ httpStatus: 500, message: 'connection lost after commit' }))
    const payload = { target_stage: 'shadow', expected_revision: 3 }
    await expect(models.promoteModel('model-server-error', payload)).rejects.toThrow()
    client.post.mockResolvedValueOnce(result('pending'))
    await models.promoteModel('model-server-error', payload)
    expect(client.post.mock.calls[1][1].idempotency_key).toBe(client.post.mock.calls[0][1].idempotency_key)
  })
})
