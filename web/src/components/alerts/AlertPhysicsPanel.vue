<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { getAlertTrajectory, type TrajectoryResponse } from '../../api/physics'
import type { TrajectoryFit } from '../../types/api'

const props = defineProps<{ alertId: string }>()

const data = ref<TrajectoryResponse | null>(null)
const loading = ref(false)

async function loadData() {
  loading.value = true
  try {
    data.value = await getAlertTrajectory(props.alertId)
  } catch {
    // Physics data may not be available for this alert
    data.value = null
  } finally {
    loading.value = false
  }
}

const primary = computed(() => data.value?.primary ?? null)
const secondaries = computed<TrajectoryFit[]>(() => {
  const list = data.value?.trajectories ?? []
  return list.filter((t) => !t.is_primary)
})
const classification = computed(() => data.value?.classification ?? null)

function formatMm(p: { x_mm: number; y_mm: number; z_mm: number } | null): string {
  if (!p) return '—'
  return `(${p.x_mm.toFixed(0)}, ${p.y_mm.toFixed(0)}, ${p.z_mm.toFixed(0)}) mm`
}

function modelLabel(model: string | null): string {
  if (!model) return '—'
  const map: Record<string, string> = {
    free_fall: '自由落体',
    projectile: '抛物线',
    projectile_drag: '带阻力抛物线',
  }
  return map[model] || model
}

function trackHue(id: number): number {
  return (id * 137.508) % 360
}

function trackColor(id: number, alpha = 1): string {
  return `hsla(${trackHue(id).toFixed(1)}, 70%, 55%, ${alpha})`
}

onMounted(loadData)
</script>

<template>
  <a-card
    v-if="data && (primary || classification || secondaries.length > 0)"
    title="物理分析"
    :bordered="false"
    size="small"
    :loading="loading"
  >
    <a-descriptions :column="1" size="small" bordered>
      <a-descriptions-item v-if="primary?.speed_ms != null" label="速度">
        <a-tag color="blue">{{ primary.speed_ms.toFixed(2) }} m/s</a-tag>
        <span
          v-if="primary.speed_px_per_sec != null"
          style="color: #888; font-size: 12px; margin-left: 8px"
        >
          ({{ primary.speed_px_per_sec.toFixed(0) }} px/s)
        </span>
      </a-descriptions-item>

      <a-descriptions-item v-if="primary?.trajectory_model" label="轨迹模型">
        <a-tag :color="primary.trajectory_model === 'free_fall' ? 'orange' : 'purple'">
          {{ modelLabel(primary.trajectory_model) }}
        </a-tag>
      </a-descriptions-item>

      <a-descriptions-item v-if="classification" label="异物分类">
        <a-tag color="red">{{ classification.label }}</a-tag>
        <span style="color: #888; font-size: 12px; margin-left: 4px">
          ({{ (classification.confidence * 100).toFixed(0) }}%)
        </span>
      </a-descriptions-item>

      <a-descriptions-item v-if="primary?.origin" label="坠落起始位置">
        <span style="font-family: monospace">{{ formatMm(primary.origin) }}</span>
      </a-descriptions-item>

      <a-descriptions-item v-if="primary?.landing" label="入水落点">
        <span style="font-family: monospace">{{ formatMm(primary.landing) }}</span>
      </a-descriptions-item>
    </a-descriptions>

    <!-- Multi-track list: shown only when more than one track was fitted -->
    <div v-if="secondaries.length > 0" style="margin-top: 12px">
      <div style="color: #888; font-size: 12px; margin-bottom: 6px">
        其它并发目标 ({{ secondaries.length }})
      </div>
      <div
        v-for="fit in secondaries"
        :key="fit.track_id"
        style="
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 4px 8px;
          border-left: 3px solid;
          margin-bottom: 4px;
          font-size: 12px;
        "
        :style="{ borderLeftColor: trackColor(fit.track_id) }"
      >
        <span
          style="
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
          "
          :style="{ background: trackColor(fit.track_id) }"
        />
        <span style="font-family: monospace">#{{ fit.track_id }}</span>
        <span>{{ modelLabel(fit.model_type) }}</span>
        <span v-if="fit.speed_ms != null">{{ fit.speed_ms.toFixed(2) }} m/s</span>
        <span style="color: #888">R²={{ fit.r_squared.toFixed(2) }}</span>
      </div>
    </div>
  </a-card>
</template>
