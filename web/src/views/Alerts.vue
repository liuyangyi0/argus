<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { useRoute, useRouter } from 'vue-router'
import { Button } from 'ant-design-vue'

import { useAlertStore } from '../stores/useAlertStore'
import { useWebSocket } from '../composables/useWebSocket'
import { getAlert } from '../api/alerts'
import AlertsToolbar from '../components/alerts/AlertsToolbar.vue'
import AlertsTable from '../components/alerts/AlertsTable.vue'
import AlertDetailPanel from '../components/alerts/AlertDetailPanel.vue'
import PageShell from '../components/common/PageShell.vue'
import AsyncState from '../components/common/AsyncState.vue'
import StatusBanner from '../components/common/StatusBanner.vue'
import { toApiError } from '../utils/error'

defineOptions({ name: 'AlertsPage' })

const route = useRoute()
const router = useRouter()

const store = useAlertStore()
const {
  alerts,
  selectedAlert,
  loading: storeLoading,
  loaded: storeLoaded,
  error: storeError,
  errorStatus: storeErrorStatus,
} = storeToRefs(store)
const initialListError = computed(() => (
  storeError.value
    && alerts.value.length === 0
    && ![403, 404].includes(storeErrorStatus.value ?? 0)
    ? storeError.value
    : ''
))
const refreshListError = computed(() => (
  storeError.value && alerts.value.length > 0 ? storeError.value : ''
))
const detailIssue = ref<{ httpStatus: number | null; message: string } | null>(null)

const fetchStarted = ref(false)
const fetchData = () => {
  fetchStarted.value = true
  // The table is recreated on entry and starts on page one; do not reuse a
  // previous visit's Pinia offset while displaying the page-one indicator.
  return store.fetchData({ offset: 0 })
}
const fetchAlerts = () => {
  fetchStarted.value = true
  return store.fetchAlerts()
}
const selectedIndex = ref(-1)
let routeSelectionSeq = 0
let detailFetchSeq = 0
let detailReturnFocus: HTMLElement | null = null

function routeAlertId(): string {
  return typeof route.query.id === 'string' ? route.query.id : ''
}

async function selectAlertById(
  alertId: string,
  options: { fetchMissing?: boolean } = {},
) {
  if (!alertId) return
  const fetchMissing = options.fetchMissing ?? true
  const seq = ++routeSelectionSeq
  detailIssue.value = null
  const existing = alerts.value.find(a => a.alert_id === alertId)
  if (existing) {
    showDetail(existing)
    return
  }
  if (!fetchMissing) return
  const detailSeq = ++detailFetchSeq
  try {
    const detail = await getAlert(alertId)
    if (
      seq !== routeSelectionSeq ||
      detailSeq !== detailFetchSeq ||
      routeAlertId() !== alertId
    ) return
    selectedAlert.value = detail
    selectedIndex.value = -1
  } catch (e) {
    if (seq === routeSelectionSeq) {
      selectedAlert.value = null
      selectedIndex.value = -1
      const apiError = toApiError(e, '告警详情加载失败')
      detailIssue.value = {
        httpStatus: apiError.httpStatus,
        message: apiError.message,
      }
    }
  }
}

function syncRouteSelectionAfterListChange() {
  const id = routeAlertId()
  if (!id || selectedAlert.value?.alert_id === id) return
  void selectAlertById(id, { fetchMissing: false })
}

useWebSocket({
  topics: ['alerts'],
  onMessage(topic, data, event) {
    if (topic !== 'alerts') return
    if (data && typeof data === 'object' && 'alert_id' in data
      && data.alert_id === selectedAlert.value?.alert_id) {
      // A detail request started before this event must not overwrite the
      // newer workflow/recording state (or reopen a just-deleted alert).
      detailFetchSeq++
    }
    store.updateFromWebSocket(data, event)
    syncRouteSelectionAfterListChange()
  },
  fallbackPoll: reconcileAlerts,
  fallbackInterval: 15000,
})

async function reconcileAlerts() {
  await fetchAlerts()
  // A lost deletion/update may affect a deep-linked detail outside this page.
  // Refresh it as well; list reconciliation alone cannot prove it still exists.
  if (selectedAlert.value) showDetail(selectedAlert.value)
}

function showDetail(record: any) {
  if (
    !selectedAlert.value
    && document.activeElement instanceof HTMLElement
    && document.activeElement !== document.body
  ) {
    detailReturnFocus = document.activeElement
  }
  detailIssue.value = null
  selectedAlert.value = record
  selectedIndex.value = alerts.value.findIndex(a => a.alert_id === record.alert_id)
  const alertId = record?.alert_id
  if (!alertId) return
  const seq = ++detailFetchSeq
  void (async () => {
    try {
      const detail = await getAlert(alertId)
      if (seq !== detailFetchSeq || selectedAlert.value?.alert_id !== alertId) return
      if (detail) {
        selectedAlert.value = { ...selectedAlert.value, ...detail }
      }
    } catch (e) {
      if (seq !== detailFetchSeq || selectedAlert.value?.alert_id !== alertId) return
      // Keep the optimistic row data visible, but never hide the failed detail
      // refresh behind what looks like a fully loaded panel.
      const apiError = toApiError(e, '告警详情刷新失败')
      if (apiError.httpStatus === 404) {
        // REST reconciliation is also the recovery path for a lost deletion
        // event. Keep the not-found explanation, not the deleted alert itself.
        detailFetchSeq++
        selectedAlert.value = null
        selectedIndex.value = -1
      }
      detailIssue.value = {
        httpStatus: apiError.httpStatus,
        message: apiError.message,
      }
    }
  })()
}

async function closeDetail() {
  detailFetchSeq++
  detailIssue.value = null
  selectedAlert.value = null
  selectedIndex.value = -1
  if (route.query.id) {
    router.replace({ query: {} })
  }
  await nextTick()
  if (detailReturnFocus?.isConnected) detailReturnFocus.focus()
  detailReturnFocus = null
}

// Keyboard navigation — lives at the page level so arrow keys work from
// anywhere in the layout, not just when the table has focus.

function handleKeydown(e: KeyboardEvent) {
  const target = e.target as HTMLElement | null
  if (target?.closest('input, textarea, select, button, [contenteditable="true"]')) return
  if (!alerts.value.length) return
  if (e.key === 'ArrowDown') {
    e.preventDefault()
    selectedIndex.value = Math.min(selectedIndex.value + 1, alerts.value.length - 1)
    showDetail(alerts.value[selectedIndex.value])
  } else if (e.key === 'ArrowUp') {
    e.preventDefault()
    selectedIndex.value = Math.max(selectedIndex.value - 1, 0)
    showDetail(alerts.value[selectedIndex.value])
  } else if (e.key === 'Escape') {
    closeDetail()
  }
}

onMounted(async () => {
  await fetchData()
  // Support URL query: ?id=xxx to auto-open alert detail
  await selectAlertById(routeAlertId())
  document.addEventListener('keydown', handleKeydown)
})

watch(() => route.query.id, (id) => {
  const alertId = typeof id === 'string' ? id : ''
  if (alertId) {
    void selectAlertById(alertId)
  } else if (selectedAlert.value) {
    void closeDetail()
  }
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <PageShell class="alerts-page" compact :busy="storeLoading">
    <div class="alerts-layout glass" :class="{ 'alerts-layout--detail': selectedAlert }">
      <!-- Left: list panel -->
      <div class="alerts-list-panel" :class="{ 'alerts-list-panel--compact': selectedAlert }">
        <AlertsToolbar />
        <StatusBanner
          v-if="refreshListError"
          class="alerts-state-banner"
          status="error"
          title="告警同步失败，当前显示上次成功获取的数据"
          :description="refreshListError"
        >
          <template #action><Button size="small" @click="fetchAlerts">重试</Button></template>
        </StatusBanner>
        <StatusBanner
          v-if="detailIssue"
          class="alerts-state-banner"
          :status="detailIssue.httpStatus === 404 ? 'warning' : 'error'"
          :title="detailIssue.httpStatus === 404 ? '告警不存在或已被删除' : '告警详情加载失败'"
          :description="detailIssue.message"
        >
          <template #action><Button size="small" @click="closeDetail">关闭</Button></template>
        </StatusBanner>
        <AsyncState
          class="alerts-list-state"
          :loading="!storeLoaded && (!fetchStarted || storeLoading)"
          :refreshing="storeLoaded && storeLoading"
          :error="initialListError"
          :forbidden="storeErrorStatus === 403"
          :not-found="storeErrorStatus === 404"
          :empty="storeLoaded && !storeError && alerts.length === 0"
          skeleton-type="table"
          :skeleton-rows="8"
          empty-title="没有符合条件的告警"
          empty-description="可调整摄像头或严重度筛选条件。"
          @retry="fetchAlerts"
        >
          <AlertsTable @select="showDetail" />
        </AsyncState>
      </div>

      <!-- Right: detail panel -->
      <AlertDetailPanel @close="closeDetail" />
    </div>
  </PageShell>
</template>

<style scoped>
.alerts-page {
  padding: 0;
  background: transparent;
}

.alerts-layout {
  display: flex;
  flex: 1;
  min-height: 0;
  overflow: hidden;
  /* margin: 12px; */
  border-radius: var(--r-lg, 12px);
  background: var(--argus-surface, rgba(255,255,255,0.85));
  backdrop-filter: blur(16px);
}

.alerts-list-panel {
  display: flex;
  flex: 0 0 100%;
  flex-direction: column;
  width: 100%;
  min-width: 0;
  overflow: hidden;
  transition: flex-basis 0.25s ease, width 0.25s ease;
}

.alerts-list-panel--compact {
  flex-basis: 520px;
  width: 520px;
  border-right: 1px solid var(--argus-sidebar-border);
}

.alerts-state-banner {
  margin: 8px 12px 0;
}

.alerts-list-state {
  min-height: 0;
}

@media (max-width: 1279px) {
  .alerts-list-panel--compact {
    flex-basis: 42%;
    width: 42%;
    min-width: 360px;
  }
}

@media (max-width: 767px) {
  .alerts-layout {
    min-height: calc(100dvh - 92px);
  }

  .alerts-layout--detail .alerts-list-panel {
    display: none;
  }

  .alerts-layout--detail :deep(.detail-panel) {
    width: 100%;
  }
}
</style>
