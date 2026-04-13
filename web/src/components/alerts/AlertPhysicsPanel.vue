<script setup lang="ts">
import { ref, onMounted } from 'vue'
import axios from 'axios'

const props = defineProps<{ alertId: string }>()

interface PhysicsData {
  speed_ms: number | null
  speed_px_per_sec: number | null
  trajectory_model: string | null
  origin: { x_mm: number; y_mm: number; z_mm: number } | null
  landing: { x_mm: number; y_mm: number; z_mm: number } | null
  classification: { label: string; confidence: number } | null
}

const data = ref<PhysicsData | null>(null)
const loading = ref(false)

async function loadData() {
  loading.value = true
  try {
    const res = await axios.get(`/api/physics/${props.alertId}/trajectory`)
    data.value = res.data?.data || res.data
  } catch {
    // Physics data may not be available for this alert
  } finally {
    loading.value = false
  }
}

function formatPoint(p: { x_mm: number; y_mm: number; z_mm: number }): string {
  return `(${p.x_mm.toFixed(0)}, ${p.y_mm.toFixed(0)}, ${p.z_mm.toFixed(0)}) mm`
}

function modelLabel(model: string): string {
  const map: Record<string, string> = {
    free_fall: '自由落体',
    projectile: '抛物线',
  }
  return map[model] || model
}

onMounted(loadData)
</script>

<template>
  <a-card
    v-if="data && (data.speed_ms != null || data.trajectory_model || data.classification)"
    title="物理分析"
    :bordered="false"
    size="small"
    :loading="loading"
  >
    <a-descriptions :column="1" size="small" bordered>
      <!-- Speed -->
      <a-descriptions-item v-if="data.speed_ms != null" label="速度">
        <a-tag color="blue">{{ data.speed_ms.toFixed(2) }} m/s</a-tag>
        <span v-if="data.speed_px_per_sec" style="color: #888; font-size: 12px; margin-left: 8px">
          ({{ data.speed_px_per_sec.toFixed(0) }} px/s)
        </span>
      </a-descriptions-item>

      <!-- Trajectory model -->
      <a-descriptions-item v-if="data.trajectory_model" label="轨迹模型">
        <a-tag :color="data.trajectory_model === 'free_fall' ? 'orange' : 'purple'">
          {{ modelLabel(data.trajectory_model) }}
        </a-tag>
      </a-descriptions-item>

      <!-- Classification -->
      <a-descriptions-item v-if="data.classification" label="异物分类">
        <a-tag color="red">{{ data.classification.label }}</a-tag>
        <span style="color: #888; font-size: 12px; margin-left: 4px">
          ({{ (data.classification.confidence * 100).toFixed(0) }}%)
        </span>
      </a-descriptions-item>

      <!-- Origin -->
      <a-descriptions-item v-if="data.origin" label="坠落起始位置">
        <span style="font-family: monospace">{{ formatPoint(data.origin) }}</span>
      </a-descriptions-item>

      <!-- Landing -->
      <a-descriptions-item v-if="data.landing" label="入水落点">
        <span style="font-family: monospace">{{ formatPoint(data.landing) }}</span>
      </a-descriptions-item>
    </a-descriptions>
  </a-card>
</template>
