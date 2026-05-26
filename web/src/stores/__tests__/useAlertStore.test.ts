import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { useAlertStore } from '../useAlertStore'

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

describe('useAlertStore realtime updates', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
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
})
