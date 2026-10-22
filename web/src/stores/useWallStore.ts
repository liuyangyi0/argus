import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { getWallStatus, getHealth } from '../api'
import { useWebSocket } from '../composables/useWebSocket'
import type { CameraTileData } from '../components/VideoTile.vue'

export const useWallStore = defineStore('wall', () => {
  const cameras = ref<CameraTileData[]>([])
  const health = ref<any>(null)
  const loading = ref(false)
  let wallRequestRevision = 0
  let healthRequestRevision = 0
  let wallRefresh: Promise<void> | null = null
  let wallRefreshRequested = false

  async function fetchWallStatus() {
    const revision = ++wallRequestRevision
    loading.value = true
    try {
      const data = await getWallStatus()
      if (revision !== wallRequestRevision) return
      cameras.value = data.cameras || []
    } finally {
      if (revision === wallRequestRevision) loading.value = false
    }
  }

  async function fetchHealth() {
    const revision = ++healthRequestRevision
    try {
      const next = await getHealth()
      if (revision === healthRequestRevision) health.value = next
    } catch { /* silent */ }
  }

  async function fetchInitialStatus() {
    await Promise.all([fetchWallStatus(), fetchHealth()])
  }

  function refreshWallAfterAlert() {
    wallRequestRevision++
    wallRefreshRequested = true
    if (wallRefresh) return
    wallRefresh = (async () => {
      do {
        wallRefreshRequested = false
        try { await fetchWallStatus() } catch { /* The next reconciliation retries. */ }
      } while (wallRefreshRequested)
    })().finally(() => {
      wallRefresh = null
      if (wallRefreshRequested) refreshWallAfterAlert()
    })
  }

  useWebSocket<{ cameras?: CameraTileData[] }>({
    topics: ['wall', 'alerts', 'health'],
    onMessage(topic, data) {
      if (topic === 'wall' && data?.cameras) {
        wallRequestRevision++
        loading.value = false
        const previous = new Map(cameras.value.map(camera => [camera.camera_id, camera]))
        // CameraManager sends a full membership snapshot, with lightweight
        // tile fields. Preserve richer REST fields only for cameras still in
        // that snapshot; newly added and removed cameras must appear at once.
        cameras.value = data.cameras.map(update => {
          const merged = { ...previous.get(update.camera_id), ...update }
          return {
            ...merged,
            alert_count_today: merged.alert_count_today ?? 0,
            active_alert: merged.active_alert ?? null,
            degradation: merged.degradation ?? null,
          }
        })
      }
      if (topic === 'alerts') {
        // Alert mutations change active_alert and daily totals, not just rows
        // on the alerts page. Do not wait 60 seconds for the wall to reconcile.
        refreshWallAfterAlert()
      }
      if (topic === 'health') {
        healthRequestRevision++
        health.value = data
      }
    },
    fallbackPoll: () => Promise.all([fetchWallStatus(), fetchHealth()]),
    fallbackInterval: 5000,
  })

  const highAlertCameras = computed(() => 
    cameras.value.filter(c => c.active_alert?.severity === 'high')
  )

  return { cameras, health, loading, fetchInitialStatus, highAlertCameras }
})
