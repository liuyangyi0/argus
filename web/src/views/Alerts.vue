<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useRoute, useRouter } from 'vue-router'

import { useAlertStore } from '../stores/useAlertStore'
import { useWebSocket } from '../composables/useWebSocket'
import AlertsToolbar from '../components/alerts/AlertsToolbar.vue'
import AlertsTable from '../components/alerts/AlertsTable.vue'
import AlertDetailPanel from '../components/alerts/AlertDetailPanel.vue'
import ContentSkeleton from '../components/ContentSkeleton.vue'

defineOptions({ name: 'AlertsPage' })

const route = useRoute()
const router = useRouter()

const store = useAlertStore()
const { alerts, selectedAlert, loading: storeLoading } = storeToRefs(store)

// Show skeleton only on the very first load — later refreshes (WebSocket
// fallback poll, ack/del) keep the existing rows visible and let the
// AlertsTable's built-in loading prop handle row-level feedback.
const hasLoadedOnce = ref(false)
const showSkeleton = computed(() => !hasLoadedOnce.value && storeLoading.value)

const fetchData = () => store.fetchData()

useWebSocket({
  topics: ['alerts'],
  onMessage(topic, data) {
    if (topic === 'alerts') store.updateFromWebSocket(data)
  },
  fallbackPoll: fetchData,
  fallbackInterval: 15000,
})

function showDetail(record: any) {
  selectedAlert.value = record
  selectedIndex.value = alerts.value.findIndex(a => a.alert_id === record.alert_id)
}

function closeDetail() {
  selectedAlert.value = null
  selectedIndex.value = -1
  if (route.query.id) {
    router.replace({ query: {} })
  }
}

// Keyboard navigation — lives at the page level so arrow keys work from
// anywhere in the layout, not just when the table has focus.
const selectedIndex = ref(-1)

function handleKeydown(e: KeyboardEvent) {
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
  hasLoadedOnce.value = true
  // Support URL query: ?id=xxx to auto-open alert detail
  if (route.query.id) {
    const target = alerts.value.find(a => a.alert_id === route.query.id)
    if (target) showDetail(target)
  }
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <main class="alerts-layout glass">
    <!-- Initial-load skeleton sits inside the layout so the page reserves the
         right amount of space immediately. -->
    <ContentSkeleton v-if="showSkeleton" type="table" :rows="8" />
    <template v-else>
      <!-- Left: list panel -->
      <div
        :style="{
          width: selectedAlert ? '520px' : '100%',
          flexShrink: 0,
          display: 'flex',
          flexDirection: 'column',
          transition: 'width 0.25s ease',
          borderRight: selectedAlert ? '1px solid var(--argus-sidebar-border)' : 'none',
          overflow: 'hidden',
        }"
      >
        <AlertsToolbar />
        <AlertsTable @select="showDetail" />
      </div>

      <!-- Right: detail panel -->
      <AlertDetailPanel @close="closeDetail" />
    </template>
  </main>
</template>

<style scoped>
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
</style>
