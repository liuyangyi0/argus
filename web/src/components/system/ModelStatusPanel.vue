<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { getModelsStatus } from '../../api/system'
import type { ModelHealthStatus } from '../../types/api'

const models = ref<ModelHealthStatus[]>([])
const lastFetch = ref<number | null>(null)
const fetchError = ref<string>('')
let timer: number | undefined

async function refresh() {
  try {
    models.value = await getModelsStatus()
    lastFetch.value = Date.now() / 1000
    fetchError.value = ''
  } catch (e) {
    fetchError.value = String((e as Error)?.message ?? e)
  }
}

onMounted(() => {
  refresh()
  timer = window.setInterval(refresh, 5000)
})
onBeforeUnmount(() => {
  if (timer !== undefined) window.clearInterval(timer)
})

const failingCount = computed(
  () => models.value.filter((m) => m.consecutive_failures > 5).length,
)
const degradedCount = computed(
  () => models.value.filter(
    (m) => !m.loaded || m.backend === 'ssim-fallback' || m.backend === 'none',
  ).length,
)

type BadgeColor = 'green' | 'orange' | 'red' | 'default'
function backendColor(m: ModelHealthStatus): BadgeColor {
  if (!m.loaded) return 'red'
  if (m.consecutive_failures > 5) return 'red'
  if (m.backend === 'ssim-fallback') return 'orange'
  if (m.backend === 'torch-cpu' || m.backend === 'cpu') return 'orange'
  if (m.backend === 'openvino' || m.backend.startsWith('torch-cuda') || m.backend === 'cuda') {
    return 'green'
  }
  return 'default'
}

function relativeTime(ts: number | null): string {
  if (ts == null) return '—'
  const dt = Math.max(0, Date.now() / 1000 - ts)
  if (dt < 1) return '刚刚'
  if (dt < 60) return `${Math.round(dt)} s ago`
  if (dt < 3600) return `${Math.round(dt / 60)} m ago`
  return `${Math.round(dt / 3600)} h ago`
}

function rowKey(m: ModelHealthStatus): string {
  return `${m.camera_id}:${m.name}`
}
</script>

<template>
  <a-card
    title="模型运行状态"
    :bordered="false"
    size="small"
    :loading="models.length === 0 && !fetchError"
  >
    <template #extra>
      <span style="color: #888; font-size: 12px">
        {{ lastFetch ? `最近刷新 ${relativeTime(lastFetch)}` : '加载中...' }}
      </span>
    </template>

    <a-alert
      v-if="failingCount > 0"
      type="error"
      show-icon
      :message="`⚠ ${failingCount} 个模型正在持续失败（连续 >5 次推理错误）`"
      style="margin-bottom: 12px"
    />
    <a-alert
      v-else-if="degradedCount > 0"
      type="warning"
      show-icon
      :message="`${degradedCount} 个模型未加载或降级到 fallback 模式`"
      style="margin-bottom: 12px"
    />
    <a-alert
      v-if="fetchError"
      type="error"
      :message="`状态拉取失败: ${fetchError}`"
      style="margin-bottom: 12px"
    />

    <a-table
      :data-source="models"
      :row-key="rowKey"
      :pagination="false"
      size="small"
    >
      <a-table-column title="模型" data-index="name" :width="90">
        <template #default="{ record }">
          <a-tag>{{ record.name }}</a-tag>
        </template>
      </a-table-column>

      <a-table-column title="相机" data-index="camera_id" :width="120" />

      <a-table-column title="Backend" :width="140">
        <template #default="{ record }">
          <a-tag :color="backendColor(record)">
            {{ record.loaded ? record.backend : '未加载' }}
          </a-tag>
        </template>
      </a-table-column>

      <a-table-column title="连续失败" :width="100" align="center">
        <template #default="{ record }">
          <span :style="{ color: record.consecutive_failures > 0 ? '#f5222d' : '#888' }">
            {{ record.consecutive_failures }}
          </span>
        </template>
      </a-table-column>

      <a-table-column title="推理计数" :width="120">
        <template #default="{ record }">
          <span style="font-family: monospace; color: #888">
            {{ record.total_inferences }} 成功
            <span v-if="record.total_failures > 0" style="color: #f5222d">
              / {{ record.total_failures }} 失败
            </span>
          </span>
        </template>
      </a-table-column>

      <a-table-column title="最近成功" :width="110">
        <template #default="{ record }">
          <span style="color: #888; font-size: 12px">
            {{ relativeTime(record.last_success_ts) }}
          </span>
        </template>
      </a-table-column>

      <a-table-column title="最近错误">
        <template #default="{ record }">
          <a-tooltip v-if="record.last_error" :title="record.last_error">
            <span style="color: #f5222d; font-family: monospace; font-size: 12px">
              {{ record.last_error.length > 50
                  ? record.last_error.slice(0, 50) + '…'
                  : record.last_error }}
            </span>
            <span style="color: #888; margin-left: 6px">
              ({{ relativeTime(record.last_error_ts) }})
            </span>
          </a-tooltip>
          <span v-else style="color: #52c41a">—</span>
        </template>
      </a-table-column>
    </a-table>
  </a-card>
</template>
