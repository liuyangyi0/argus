import type { AxiosResponse } from 'axios'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { api } from '../../api/client'
import { useAuthStore } from '../useAuthStore'
import type { ApiResponse } from '../../types/api'

vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/client')>()
  return {
    ...actual,
    api: { get: vi.fn() },
  }
})

function response<T>(data: T): AxiosResponse<ApiResponse<T>> {
  return {
    data: { code: 0, msg: 'ok', data },
    status: 200,
    statusText: 'OK',
    headers: {},
    config: {} as AxiosResponse<ApiResponse<T>>['config'],
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

describe('useAuthStore API envelope', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(api.get).mockReset()
  })

  it('unwraps the current user before applying role checks', async () => {
    vi.mocked(api.get).mockResolvedValue(response({
      username: 'engineer-a',
      role: 'engineer',
      display_name: 'Engineer A',
    }))
    const store = useAuthStore()

    await expect(store.fetchCurrentUser()).resolves.toEqual({
      username: 'engineer-a',
      role: 'engineer',
      display_name: 'Engineer A',
    })
    expect(store.hasRole(['admin', 'engineer'])).toBe(true)
    expect(store.currentUser?.role).toBe('engineer')
  })

  it('rejects a malformed or unknown role instead of creating a truthy fake user', async () => {
    vi.mocked(api.get).mockResolvedValue(response({
      username: 'mystery',
      role: 'superuser',
    }))
    const store = useAuthStore()

    await expect(store.fetchCurrentUser()).resolves.toBeNull()
    expect(store.currentUser).toBeNull()
    expect(store.hasRole(['admin'])).toBe(false)
  })

  it('fails closed and removes a previously privileged identity on refresh error', async () => {
    vi.mocked(api.get)
      .mockResolvedValueOnce(response({ username: 'admin-a', role: 'admin' }))
      .mockResolvedValueOnce(response({ username: 'admin-a', role: 'superuser' }))
    const store = useAuthStore()

    await expect(store.fetchCurrentUser()).resolves.toMatchObject({ role: 'admin' })
    await expect(store.fetchCurrentUser(true)).resolves.toBeNull()

    expect(store.currentUser).toBeNull()
    expect(store.hasRole(['admin'])).toBe(false)
    expect(store.lastFetchedAt).toBe(0)
  })

  it('does not let an identity request started before clear restore the session', async () => {
    const pending = deferred<AxiosResponse<ApiResponse<{ username: string; role: 'admin' }>>>()
    vi.mocked(api.get).mockReturnValue(pending.promise)
    const store = useAuthStore()

    const request = store.fetchCurrentUser()
    store.clear()
    pending.resolve(response({ username: 'late-admin', role: 'admin' }))

    await expect(request).resolves.toBeNull()
    expect(store.currentUser).toBeNull()
    expect(store.hasRole(['admin'])).toBe(false)
    expect(store.isLoading).toBe(false)
  })

  it('makes ordinary navigation join an in-flight authoritative role refresh', async () => {
    const pending = deferred<AxiosResponse<ApiResponse<{ username: string; role: 'viewer' }>>>()
    vi.mocked(api.get)
      .mockResolvedValueOnce(response({ username: 'role-user', role: 'admin' }))
      .mockReturnValueOnce(pending.promise)
    const store = useAuthStore()
    await store.fetchCurrentUser()

    const forced = store.fetchCurrentUser(true)
    const navigation = store.fetchCurrentUser()
    pending.resolve(response({ username: 'role-user', role: 'viewer' }))

    await expect(forced).resolves.toMatchObject({ role: 'viewer' })
    await expect(navigation).resolves.toMatchObject({ role: 'viewer' })
    expect(store.currentUser?.role).toBe('viewer')
    expect(api.get).toHaveBeenCalledTimes(2)
  })

  it('coalesces concurrent identity requests and caches the validated result', async () => {
    vi.mocked(api.get).mockResolvedValue(response({ username: 'viewer-a', role: 'viewer' }))
    const store = useAuthStore()

    const [first, second] = await Promise.all([
      store.fetchCurrentUser(),
      store.fetchCurrentUser(),
    ])
    const cached = await store.fetchCurrentUser()

    expect(first).toEqual(second)
    expect(cached).toEqual(first)
    expect(api.get).toHaveBeenCalledTimes(1)
  })
})
