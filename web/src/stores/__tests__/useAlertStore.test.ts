import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { flushPromises } from '@vue/test-utils'

import { useAlertStore } from '../useAlertStore'
import { getAlerts, getCameras } from '../../api'

vi.mock('../../api', () => ({
  getAlerts: vi.fn(),
  getCameras: vi.fn(),
  acknowledgeAlert: vi.fn(),
  markFalsePositive: vi.fn(),
  deleteAlert: vi.fn(),
  bulkDeleteAlerts: vi.fn(),
  bulkAcknowledge: vi.fn(),
  bulkFalsePositive: vi.fn(),
}))

function alert(overrides: Record<string, any>) {
  return {
    alert_id: 'alert-1',
    camera_id: 'cam_01',
    severity: 'medium',
    workflow_status: 'new',
    anomaly_score: 0.8,
    ...overrides,
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}

describe('useAlertStore realtime updates', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(getAlerts).mockReset()
    vi.mocked(getCameras).mockReset()
  })

  it('keeps a successful alert list when camera options fail', async () => {
    vi.mocked(getAlerts).mockResolvedValue({
      alerts: [alert({ alert_id: 'loaded-alert' })],
      total: 1,
    } as any)
    vi.mocked(getCameras).mockRejectedValue(new Error('camera registry offline'))

    const store = useAlertStore()
    const succeeded = await store.fetchData()

    expect(succeeded).toBe(true)
    expect(store.alerts.map(item => item.alert_id)).toEqual(['loaded-alert'])
    expect(store.error).toBe('')
    expect(store.loaded).toBe(true)
    expect(store.cameras).toEqual([])
    expect(store.camerasError).toContain('camera registry offline')
    expect(store.camerasLoaded).toBe(true)
  })

  it('keeps successful camera options when the alert request fails', async () => {
    vi.mocked(getAlerts).mockRejectedValue(new Error('alert database offline'))
    vi.mocked(getCameras).mockResolvedValue({
      cameras: [{ camera_id: 'cam_01' }],
    } as any)

    const store = useAlertStore()
    const succeeded = await store.fetchData()

    expect(succeeded).toBe(false)
    expect(store.alerts).toEqual([])
    expect(store.error).toContain('alert database offline')
    expect(store.cameras).toEqual([{ camera_id: 'cam_01' }])
    expect(store.camerasError).toBe('')
  })

  it('preserves the last camera options after a transient refresh failure', async () => {
    vi.mocked(getCameras)
      .mockResolvedValueOnce({ cameras: [{ camera_id: 'cam_01' }] } as any)
      .mockRejectedValueOnce(new Error('temporary camera timeout'))

    const store = useAlertStore()
    await store.fetchCameraOptions()
    await store.fetchCameraOptions()

    expect(store.cameras).toEqual([{ camera_id: 'cam_01' }])
    expect(store.camerasError).toContain('temporary camera timeout')
    expect(store.camerasLoading).toBe(false)
  })

  it('inserts matching websocket alerts at the top and caps the visible page', () => {
    const store = useAlertStore()
    store.alerts = Array.from({ length: 20 }, (_, idx) =>
      alert({ alert_id: `old-${idx + 1}`, anomaly_score: 0.4 }),
    )
    store.totalAlerts = 20

    store.updateFromWebSocket(alert({ alert_id: 'new-1', anomaly_score: 0.95 }))

    expect(store.alerts).toHaveLength(20)
    expect(store.alerts[0].alert_id).toBe('new-1')
    expect(store.alerts.at(-1)?.alert_id).toBe('old-19')
    expect(store.totalAlerts).toBe(21)
  })

  it('keeps evidence fields from realtime websocket alerts for the visible row', () => {
    const store = useAlertStore()

    store.updateFromWebSocket(alert({
      alert_id: 'live-with-evidence',
      has_recording: true,
      recording_status: 'recording',
      snapshot_path: 'data/alerts/live/snapshot.jpg',
      heatmap_path: 'data/alerts/live/heatmap.jpg',
    }))

    expect(store.alerts[0]).toMatchObject({
      alert_id: 'live-with-evidence',
      has_recording: true,
      recording_status: 'recording',
      snapshot_path: 'data/alerts/live/snapshot.jpg',
      heatmap_path: 'data/alerts/live/heatmap.jpg',
    })
    expect(store.totalAlerts).toBe(1)
  })

  it('updates an existing row and selected alert from websocket data', () => {
    const store = useAlertStore()
    store.alerts = [alert({ alert_id: 'alert-1', recording_status: 'pending' })]
    store.selectedAlert = alert({ alert_id: 'alert-1', recording_status: 'pending' })
    store.totalAlerts = 1

    store.updateFromWebSocket({
      alert_id: 'alert-1',
      has_recording: true,
      recording_status: 'complete',
    })

    expect(store.alerts[0].recording_status).toBe('complete')
    expect(store.alerts[0].has_recording).toBe(true)
    expect(store.selectedAlert?.recording_status).toBe('complete')
    expect(store.totalAlerts).toBe(1)
  })

  it('updates selected alert from websocket data even when it is outside the visible list', () => {
    const store = useAlertStore()
    store.filters.severity = 'high'
    store.alerts = [alert({
      alert_id: 'visible-alert',
      severity: 'high',
      recording_status: 'recording',
    })]
    store.selectedAlert = alert({
      alert_id: 'deep-linked-alert',
      severity: 'medium',
      has_recording: true,
      recording_status: 'recording',
    })
    store.totalAlerts = 1

    store.updateFromWebSocket({
      alert_id: 'deep-linked-alert',
      severity: 'medium',
      has_recording: true,
      recording_status: 'complete',
      replay_frame_count: 214,
    })

    expect(store.alerts.map(a => a.alert_id)).toEqual(['visible-alert'])
    expect(store.totalAlerts).toBe(1)
    expect(store.selectedAlert).toMatchObject({
      alert_id: 'deep-linked-alert',
      has_recording: true,
      recording_status: 'complete',
      replay_frame_count: 214,
    })
  })

  it('does not mix unrelated websocket alerts into a filtered list', () => {
    const store = useAlertStore()
    store.filters.camera_id = 'cam_01'
    store.filters.severity = 'high'
    store.alerts = [alert({ alert_id: 'visible', severity: 'high' })]
    store.totalAlerts = 1

    store.updateFromWebSocket(alert({
      alert_id: 'wrong-camera',
      camera_id: 'cam_02',
      severity: 'high',
    }))
    store.updateFromWebSocket(alert({
      alert_id: 'wrong-severity',
      camera_id: 'cam_01',
      severity: 'low',
    }))

    expect(store.alerts.map(a => a.alert_id)).toEqual(['visible'])
    expect(store.totalAlerts).toBe(1)
  })

  it('filters websocket snapshot payloads before replacing the visible list', () => {
    const store = useAlertStore()
    store.filters.camera_id = 'cam_01'
    store.filters.severity = 'medium'

    store.updateFromWebSocket([
      alert({ alert_id: 'match', camera_id: 'cam_01', severity: 'medium' }),
      alert({ alert_id: 'other-camera', camera_id: 'cam_02', severity: 'medium' }),
      alert({ alert_id: 'other-severity', camera_id: 'cam_01', severity: 'high' }),
    ])

    expect(store.alerts.map(a => a.alert_id)).toEqual(['match'])
    expect(store.totalAlerts).toBe(1)
  })

  it('removes a deleted alert and closes its selected detail instead of upserting it', async () => {
    vi.mocked(getAlerts).mockResolvedValue({ alerts: [], total: 0 } as any)
    const store = useAlertStore()
    store.alerts = [alert({ alert_id: 'deleted-alert' })]
    store.selectedAlert = store.alerts[0]
    store.totalAlerts = 1

    store.updateFromWebSocket(alert({ alert_id: 'deleted-alert', type: 'deleted' }))

    expect(store.alerts).toEqual([])
    expect(store.selectedAlert).toBeNull()
    expect(store.totalAlerts).toBe(0)
    await flushPromises()
    expect(getAlerts).toHaveBeenCalledOnce()
  })

  it('never inserts a deleted or updated off-page alert as a new alert', async () => {
    const refresh = deferred<any>()
    vi.mocked(getAlerts).mockReturnValue(refresh.promise)
    const store = useAlertStore()
    store.alerts = [alert({ alert_id: 'visible-alert' })]
    store.totalAlerts = 40

    store.updateFromWebSocket(alert({ alert_id: 'off-page', type: 'acknowledged' }))
    expect(store.alerts.map(item => item.alert_id)).toEqual(['visible-alert'])
    expect(store.totalAlerts).toBe(40)
    store.updateFromWebSocket(alert({ alert_id: 'deleted-off-page', type: 'deleted' }))
    expect(store.alerts.map(item => item.alert_id)).toEqual(['visible-alert'])
    expect(store.totalAlerts).toBe(39)

    refresh.resolve({ alerts: [alert({ alert_id: 'visible-alert' })], total: 39 })
    await flushPromises()
    expect(store.totalAlerts).toBe(39)
  })

  it('keeps creations out of later pages and reconciles the same offset', async () => {
    const refresh = deferred<any>()
    vi.mocked(getAlerts)
      .mockResolvedValueOnce({ alerts: [alert({ alert_id: 'page-two' })], total: 40 } as any)
      .mockReturnValueOnce(refresh.promise)
    const store = useAlertStore()
    await store.fetchAlerts({ limit: 20, offset: 20 })

    store.updateFromWebSocket(alert({ alert_id: 'new-first-page', type: 'created' }))

    expect(store.alerts.map(item => item.alert_id)).toEqual(['page-two'])
    expect(getAlerts).toHaveBeenLastCalledWith({ limit: 20, offset: 20 })
    refresh.resolve({ alerts: [alert({ alert_id: 'shifted-page-two' })], total: 41 })
    await flushPromises()
    expect(store.alerts.map(item => item.alert_id)).toEqual(['shifted-page-two'])
    expect(store.totalAlerts).toBe(41)
  })

  it('rejects a REST response started before a newer alert event', async () => {
    const stale = deferred<any>()
    const refresh = deferred<any>()
    vi.mocked(getAlerts).mockReturnValueOnce(stale.promise).mockReturnValueOnce(refresh.promise)
    const store = useAlertStore()
    const originalRequest = store.fetchAlerts()

    store.updateFromWebSocket(alert({ alert_id: 'newer-alert', type: 'created' }))
    stale.resolve({ alerts: [alert({ alert_id: 'stale-alert' })], total: 1 })
    await originalRequest
    expect(store.alerts.map(item => item.alert_id)).toEqual(['newer-alert'])

    refresh.resolve({ alerts: [alert({ alert_id: 'newer-alert' })], total: 1 })
    await flushPromises()
    expect(store.alerts.map(item => item.alert_id)).toEqual(['newer-alert'])
    expect(store.loading).toBe(false)
  })
})
