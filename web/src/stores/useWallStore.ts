import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { getWallStatus, getHealth } from '../api'
import { useWebSocket } from '../composables/useWebSocket'
import type { CameraTileData } from '../components/VideoTile.vue'

export const useWallStore = defineStore('wall', () => {
  const cameras = ref<CameraTileData[]>([])
  const health = ref<any>(null)
  const loading = ref(false)

  async function fetchWallStatus() {
    loading.value = true
    try {
      const data = await getWallStatus()
      cameras.value = data.cameras || []
    } finally {
      loading.value = false
    }
  }

  async function fetchHealth() {
    try {
      health.value = await getHealth()
    } catch { /* silent */ }
  }

  async function fetchInitialStatus() {
    await Promise.all([fetchWallStatus(), fetchHealth()])
  }

  useWebSocket({
    topics: ['wall', 'alerts', 'health'],
    onMessage(topic, data) {
      if (topic === 'wall' && data?.cameras) {
        data.cameras.forEach((update: any) => {
          const idx = cameras.value.findIndex(c => c.camera_id === update.camera_id)
          if (idx !== -1) {
            cameras.value[idx] = { ...cameras.value[idx], ...update }
          }
        })
      }
      if (topic === 'health') {
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
