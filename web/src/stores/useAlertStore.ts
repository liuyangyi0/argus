import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { getAlerts, getCameras, acknowledgeAlert, markFalsePositive, deleteAlert, bulkDeleteAlerts, bulkAcknowledge, bulkFalsePositive } from '../api'

export const useAlertStore = defineStore('alertStore', () => {
  // State
  const alerts = ref<any[]>([])
  const cameras = ref<any[]>([])
  const totalAlerts = ref(0)
  const loading = ref(false)

  // Filters
  const filters = ref({ camera_id: '', severity: '' })

  // Selection
  const selectedAlert = ref<any>(null)
  
  // Computed stats
  const activeCount = computed(() => alerts.value.filter(a => a.workflow_status === 'new').length)
  const resolvedCount = computed(() => alerts.value.filter(a => ['resolved', 'closed'].includes(a.workflow_status)).length)

  // Actions
  async function fetchData(paramsOverride?: Record<string, any>) {
    loading.value = true
    try {
      const params: Record<string, any> = { limit: 100, ...paramsOverride }
      if (filters.value.camera_id) params.camera_id = filters.value.camera_id
      if (filters.value.severity) params.severity = filters.value.severity
      
      const [aRes, cRes] = await Promise.all([getAlerts(params), getCameras()])
      alerts.value = aRes.alerts || []
      totalAlerts.value = aRes.total ?? (aRes.alerts?.length || 0)
      cameras.value = cRes.cameras || []
      
      // Update selectedAlert reference if data refreshed
      if (selectedAlert.value) {
        const updated = alerts.value.find(a => a.alert_id === selectedAlert.value.alert_id)
        if (updated) {
          selectedAlert.value = { ...selectedAlert.value, ...updated }
        }
      }
    } finally {
      loading.value = false
    }
  }

  // Handle incoming WS data
  function updateFromWebSocket(data: any) {
    if (data && typeof data === 'object' && !Array.isArray(data) && data.alert_id) {
      const idx = alerts.value.findIndex(a => a.alert_id === data.alert_id)
      if (idx >= 0) {
        alerts.value[idx] = { ...alerts.value[idx], ...data }
        if (selectedAlert.value?.alert_id === data.alert_id) {
          selectedAlert.value = { ...selectedAlert.value, ...data }
        }
      } else {
        alerts.value.unshift(data)
        totalAlerts.value++
      }
    } else if (Array.isArray(data)) {
      alerts.value = data
      totalAlerts.value = data.length
    }
  }

  // API wrappers (returns Promise so UI can show message/modal)
  async function ackAlert(id: string) {
    await acknowledgeAlert(id)
    if (selectedAlert.value?.alert_id === id) {
      selectedAlert.value.workflow_status = 'acknowledged'
    }
    await fetchData()
  }

  async function fpAlert(id: string) {
    await markFalsePositive(id)
    if (selectedAlert.value?.alert_id === id) {
      selectedAlert.value.workflow_status = 'false_positive'
    }
    await fetchData()
  }

  async function delAlert(id: string) {
    await deleteAlert(id)
    if (selectedAlert.value?.alert_id === id) {
      selectedAlert.value = null
    }
    await fetchData()
  }

  async function bulkDel(keys: string[]) {
    const res = await bulkDeleteAlerts(keys)
    if (selectedAlert.value && keys.includes(selectedAlert.value.alert_id)) {
      selectedAlert.value = null
    }
    await fetchData()
    return res
  }

  async function bulkAck(keys: string[]) {
    const res = await bulkAcknowledge(keys)
    await fetchData()
    return res
  }

  async function bulkFp(keys: string[]) {
    const res = await bulkFalsePositive(keys)
    await fetchData()
    return res
  }

  return {
    alerts, cameras, totalAlerts, loading, filters, selectedAlert,
    activeCount, resolvedCount,
    fetchData, updateFromWebSocket,
    ackAlert, fpAlert, delAlert, bulkDel, bulkAck, bulkFp
  }
})
