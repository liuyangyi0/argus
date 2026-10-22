import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { getAlerts, getCameras, acknowledgeAlert, markFalsePositive, deleteAlert, bulkDeleteAlerts, bulkAcknowledge, bulkFalsePositive } from '../api'
import type { AlertQueryParams, WsEvent } from '../types/api'
import { toApiError } from '../utils/error'

export const useAlertStore = defineStore('alertStore', () => {
  // State
  const alerts = ref<any[]>([])
  const cameras = ref<any[]>([])
  const totalAlerts = ref(0)
  const loading = ref(false)
  const loaded = ref(false)
  const error = ref('')
  const errorStatus = ref<number | null>(null)
  const camerasLoading = ref(false)
  const camerasLoaded = ref(false)
  const camerasError = ref('')
  const camerasErrorStatus = ref<number | null>(null)

  // Filters
  const filters = ref({ camera_id: '', severity: '' })
  const pageParams = ref({ limit: 20, offset: 0 })
  let lastFilterKey = ''
  let alertRequestRevision = 0
  let cameraRequestRevision = 0
  let reconciliation: Promise<void> | null = null
  let reconciliationRequested = false

  // Selection
  const selectedAlert = ref<any>(null)
  
  // Computed stats
  const activeCount = computed(() => alerts.value.filter(a => a.workflow_status === 'new').length)
  const resolvedCount = computed(() => alerts.value.filter(a => ['resolved', 'closed'].includes(a.workflow_status)).length)

  // Actions
  async function fetchAlerts(paramsOverride: AlertQueryParams = {}) {
    const filterKey = `${filters.value.camera_id}\u0000${filters.value.severity}`
    const filterChanged = filterKey !== lastFilterKey
    const requestRevision = ++alertRequestRevision
    loading.value = true
    error.value = ''
    errorStatus.value = null
    try {
      if (filterChanged && !Object.prototype.hasOwnProperty.call(paramsOverride, 'offset')) {
        pageParams.value.offset = 0
      }

      const params: AlertQueryParams = { ...pageParams.value, ...paramsOverride }
      pageParams.value = {
        limit: params.limit ?? pageParams.value.limit,
        offset: params.offset ?? pageParams.value.offset,
      }
      if (filters.value.camera_id) params.camera_id = filters.value.camera_id
      if (filters.value.severity) params.severity = filters.value.severity
      
      const aRes = await getAlerts(params)
      if (requestRevision !== alertRequestRevision) return false

      alerts.value = aRes.alerts || []
      totalAlerts.value = aRes.total ?? (aRes.alerts?.length || 0)
      lastFilterKey = filterKey
      
      // Update selectedAlert reference if data refreshed
      if (selectedAlert.value) {
        const updated = alerts.value.find(a => a.alert_id === selectedAlert.value.alert_id)
        if (updated) {
          selectedAlert.value = { ...selectedAlert.value, ...updated }
        }
      }
      return true
    } catch (e) {
      if (requestRevision !== alertRequestRevision) return false

      // Keep the last successfully fetched rows on refresh failure. The page
      // renders this error separately from the true empty state.
      // If the filters changed, those old rows no longer describe the active
      // query and must not remain visible under the new filter controls.
      if (filterChanged) {
        alerts.value = []
        totalAlerts.value = 0
      }
      const apiError = toApiError(e, '告警列表加载失败')
      error.value = apiError.message
      errorStatus.value = apiError.httpStatus
      if ([403, 404].includes(apiError.httpStatus ?? 0)) {
        alerts.value = []
        totalAlerts.value = 0
      }
      return false
    } finally {
      if (requestRevision === alertRequestRevision) {
        loaded.value = true
        loading.value = false
      }
    }
  }

  async function fetchCameraOptions() {
    const requestRevision = ++cameraRequestRevision
    camerasLoading.value = true
    camerasError.value = ''
    camerasErrorStatus.value = null
    try {
      const response = await getCameras()
      if (requestRevision !== cameraRequestRevision) return false

      cameras.value = response.cameras || []
      return true
    } catch (e) {
      if (requestRevision !== cameraRequestRevision) return false

      const apiError = toApiError(e, '摄像头筛选项加载失败')
      camerasError.value = apiError.message
      camerasErrorStatus.value = apiError.httpStatus
      // Keep the last successful options for transient refresh failures, but
      // do not retain options the current user is no longer allowed to see.
      if ([403, 404].includes(apiError.httpStatus ?? 0)) {
        cameras.value = []
      }
      return false
    } finally {
      if (requestRevision === cameraRequestRevision) {
        camerasLoaded.value = true
        camerasLoading.value = false
      }
    }
  }

  async function fetchData(paramsOverride: AlertQueryParams = {}) {
    const [alertsSucceeded] = await Promise.all([
      fetchAlerts(paramsOverride),
      fetchCameraOptions(),
    ])
    return alertsSucceeded
  }

  // Handle incoming WS data
  function matchesCurrentFilters(alert: any): boolean {
    if (filters.value.camera_id && alert.camera_id !== filters.value.camera_id) {
      return false
    }
    if (filters.value.severity && alert.severity !== filters.value.severity) {
      return false
    }
    return true
  }

  function reconcileAlertEvents() {
    reconciliationRequested = true
    if (reconciliation) return
    // An event is newer than every REST request already in flight. Invalidate
    // those responses before starting an authoritative, coalesced page reload.
    // Repeating once after events arrive during that reload also repairs totals
    // and page boundaries without guessing where an off-page row belongs.
    reconciliation = (async () => {
      do {
        reconciliationRequested = false
        await fetchAlerts()
      } while (reconciliationRequested)
    })().finally(() => {
      reconciliation = null
      if (reconciliationRequested) reconcileAlertEvents()
    })
  }

  function updateFromWebSocket(data: any, event?: WsEvent<unknown>) {
    if (data && typeof data === 'object' && !Array.isArray(data) && data.alert_id) {
      const eventType = event?.type || (typeof data.type === 'string' ? data.type : '')
      const authoritativeEvent = Boolean(eventType)
      if (authoritativeEvent) alertRequestRevision++
      const idx = alerts.value.findIndex(a => a.alert_id === data.alert_id)
      const selectedMatches = selectedAlert.value?.alert_id === data.alert_id
      if (eventType === 'deleted') {
        if (idx >= 0) alerts.value.splice(idx, 1)
        if (selectedMatches) selectedAlert.value = null
        if (idx >= 0 || matchesCurrentFilters(data)) {
          totalAlerts.value = Math.max(0, totalAlerts.value - 1)
        }
        reconcileAlertEvents()
        return
      }
      if (idx >= 0) {
        const merged = { ...alerts.value[idx], ...data }
        if (matchesCurrentFilters(merged)) {
          alerts.value[idx] = merged
        } else {
          alerts.value.splice(idx, 1)
          totalAlerts.value = Math.max(0, totalAlerts.value - 1)
        }
        if (selectedMatches) {
          selectedAlert.value = { ...selectedAlert.value, ...data }
        }
      } else {
        if (selectedMatches) {
          selectedAlert.value = { ...selectedAlert.value, ...data }
        }
        // A workflow/recording update is not a new alert. In particular, never
        // splice an off-page update (or a first-page creation) into page two.
        // Unversioned callers retain the old upsert behavior for compatibility.
        if ((!authoritativeEvent || eventType === 'created') && matchesCurrentFilters(data)) {
          if (pageParams.value.offset === 0) {
            alerts.value.unshift(data)
            if (pageParams.value.limit > 0 && alerts.value.length > pageParams.value.limit) {
              alerts.value = alerts.value.slice(0, pageParams.value.limit)
            }
          }
          totalAlerts.value++
        }
      }
      if (authoritativeEvent) reconcileAlertEvents()
    } else if (Array.isArray(data)) {
      const filtered = data.filter(matchesCurrentFilters)
      alerts.value = filtered.slice(0, pageParams.value.limit)
      totalAlerts.value = filtered.length
    }
  }

  // API wrappers (returns Promise so UI can show message/modal)
  async function ackAlert(id: string) {
    await acknowledgeAlert(id)
    if (selectedAlert.value?.alert_id === id) {
      selectedAlert.value.workflow_status = 'acknowledged'
    }
    await fetchAlerts()
  }

  async function fpAlert(id: string) {
    await markFalsePositive(id)
    if (selectedAlert.value?.alert_id === id) {
      selectedAlert.value.workflow_status = 'false_positive'
    }
    await fetchAlerts()
  }

  async function delAlert(id: string) {
    await deleteAlert(id)
    if (selectedAlert.value?.alert_id === id) {
      selectedAlert.value = null
    }
    await fetchAlerts()
  }

  async function bulkDel(keys: string[]) {
    const res = await bulkDeleteAlerts(keys)
    if (selectedAlert.value && keys.includes(selectedAlert.value.alert_id)) {
      selectedAlert.value = null
    }
    await fetchAlerts()
    return res
  }

  async function bulkAck(keys: string[]) {
    const res = await bulkAcknowledge(keys)
    await fetchAlerts()
    return res
  }

  async function bulkFp(keys: string[]) {
    const res = await bulkFalsePositive(keys)
    await fetchAlerts()
    return res
  }

  return {
    alerts, cameras, totalAlerts, loading, loaded, error, errorStatus,
    camerasLoading, camerasLoaded, camerasError, camerasErrorStatus,
    filters, selectedAlert,
    activeCount, resolvedCount,
    fetchData, fetchAlerts, fetchCameraOptions, updateFromWebSocket,
    ackAlert, fpAlert, delAlert, bulkDel, bulkAck, bulkFp
  }
})
