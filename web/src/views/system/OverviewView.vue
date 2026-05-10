<script setup lang="ts">
import { onMounted, ref, computed } from 'vue'
import { Card as ACard, Tag as ATag, Empty as AEmpty } from 'ant-design-vue'
import SystemOverviewPanel from '../../components/system/SystemOverviewPanel.vue'
import ModelStatusPanel from '../../components/system/ModelStatusPanel.vue'
import { getAnomalyDegradation, type AnomalyDegradationStatus } from '../../api/system'
import { useWebSocket } from '../../composables/useWebSocket'

// Anomaly head degradation card — shows whether each pipeline's anomaly
// detector is in the simplex-only fallback mode (see core/pipeline.py).
// Uses initial fetch + WebSocket push; no polling because state changes are
// rare and the WS broadcast covers the live transitions.
const degradation = ref<AnomalyDegradationStatus['anomaly'] | null>(null)
const loading = ref(true)

async function fetchDegradation() {
  try {
    const res = await getAnomalyDegradation()
    degradation.value = res?.anomaly ?? null
  } catch {
    degradation.value = null
  } finally {
    loading.value = false
  }
}

useWebSocket({
  topics: ['system_degradation'],
  onMessage: (_topic, _data) => {
    // Any state change on this topic invalidates our snapshot — re-query
    // for the authoritative aggregated view.
    fetchDegradation()
  },
})

onMounted(fetchDegradation)

const aggregateState = computed(() => {
  if (loading.value) return { label: '加载中…', color: 'default' as const }
  if (!degradation.value) return { label: '未知', color: 'default' as const }
  return degradation.value.degraded
    ? { label: '降级', color: 'red' as const }
    : { label: '正常', color: 'green' as const }
})

function formatSince(since: number | null): string {
  if (!since) return '—'
  const d = new Date(since * 1000)
  return d.toLocaleString()
}
</script>

<template>
  <div>
    <SystemOverviewPanel />
    <div style="margin-top: 16px">
      <ModelStatusPanel />
    </div>

    <a-card
      title="降级监控"
      size="small"
      style="margin-top: 16px"
    >
      <template #extra>
        <a-tag :color="aggregateState.color">异常检测：{{ aggregateState.label }}</a-tag>
      </template>

      <div v-if="degradation && degradation.cameras && degradation.cameras.length > 0">
        <div
          v-for="cam in degradation.cameras"
          :key="cam.camera_id"
          style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(10,10,15,.06)"
        >
          <div>
            <strong style="font-size: 13px">{{ cam.camera_id }}</strong>
            <div v-if="cam.degraded" style="font-size: 11px; color: var(--ink-5); margin-top: 2px">
              原因：{{ cam.reason ?? '未知' }} · 进入时间：{{ formatSince(cam.since) }}
            </div>
          </div>
          <a-tag :color="cam.degraded ? 'red' : 'green'">
            {{ cam.degraded ? 'degraded' : '正常' }}
          </a-tag>
        </div>
      </div>

      <a-empty v-else-if="!loading" description="暂无摄像头" :image-style="{ height: '40px' }" />
      <div v-else style="text-align: center; padding: 20px; color: var(--ink-5); font-size: 12px">
        加载中…
      </div>
    </a-card>
  </div>
</template>
