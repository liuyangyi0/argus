import { flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getHealth, getWallStatus } from '../../api'
import { useWebSocket } from '../../composables/useWebSocket'
import { useWallStore } from '../useWallStore'

vi.mock('../../api', () => ({ getHealth: vi.fn(), getWallStatus: vi.fn() }))
vi.mock('../../composables/useWebSocket', () => ({ useWebSocket: vi.fn() }))

function camera(cameraId: string) {
  return {
    camera_id: cameraId, name: cameraId, status: 'online',
    current_score: 0.2, score_sparkline: [], alert_count_today: 3,
    active_alert: null, degradation: null,
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}

function send(topic: 'wall' | 'health' | 'alerts', data: unknown) {
  const callback = vi.mocked(useWebSocket).mock.calls.at(-1)?.[0].onMessage
  callback?.(topic, data)
}

describe('wall store realtime reconciliation', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(useWebSocket).mockReset()
    vi.mocked(getWallStatus).mockReset()
    vi.mocked(getHealth).mockReset()
    vi.mocked(getHealth).mockResolvedValue({ status: 'healthy' } as any)
  })

  it('adds and removes cameras from a full wall snapshot while retaining rich fields', () => {
    const store = useWallStore()
    store.cameras = [camera('kept'), camera('removed')]

    send('wall', { cameras: [
      { camera_id: 'kept', name: 'Kept', status: 'offline', current_score: 0, score_sparkline: [] },
      { camera_id: 'added', name: 'Added', status: 'online', current_score: 0, score_sparkline: [] },
    ] })

    expect(store.cameras.map(item => item.camera_id)).toEqual(['kept', 'added'])
    expect(store.cameras[0]).toMatchObject({ status: 'offline', alert_count_today: 3 })
    expect(store.cameras[1]).toMatchObject({ alert_count_today: 0, active_alert: null })
    send('wall', { cameras: [] })
    expect(store.cameras).toEqual([])
  })

  it('does not restore a removed camera from a REST request started before the snapshot', async () => {
    const stale = deferred<any>()
    vi.mocked(getWallStatus).mockReturnValue(stale.promise)
    const store = useWallStore()
    const initial = store.fetchInitialStatus()

    send('wall', { cameras: [camera('current')] })
    stale.resolve({ cameras: [camera('removed')] })
    await initial

    expect(store.cameras.map(item => item.camera_id)).toEqual(['current'])
    expect(store.loading).toBe(false)
  })

  it('refreshes wall alert totals immediately after an alert mutation', async () => {
    vi.mocked(getWallStatus).mockResolvedValue({ cameras: [{ ...camera('cam'), alert_count_today: 4 }] } as any)
    const store = useWallStore()

    send('alerts', { alert_id: 'alert-1', type: 'created' })
    await flushPromises()

    expect(getWallStatus).toHaveBeenCalledOnce()
    expect(store.cameras[0]?.alert_count_today).toBe(4)
  })

  it('keeps a newer health event when an older REST health request completes', async () => {
    const stale = deferred<any>()
    vi.mocked(getHealth).mockReturnValue(stale.promise)
    vi.mocked(getWallStatus).mockResolvedValue({ cameras: [] } as any)
    const store = useWallStore()
    const initial = store.fetchInitialStatus()

    send('health', { status: 'unhealthy', cameras: [] })
    stale.resolve({ status: 'healthy' })
    await initial

    expect(store.health.status).toBe('unhealthy')
  })
})
