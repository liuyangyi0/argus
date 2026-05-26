<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Statistic, Card, message, Alert } from 'ant-design-vue'
import { getModelRegistry } from '../../api'
import { useWebSocket } from '../../composables/useWebSocket'
import ModelTable from './ModelTable.vue'
import EventLog from './EventLog.vue'
import BatchInference from './BatchInference.vue'

const props = defineProps<{
  cameras: any[]
  focusVersionId?: string
}>()

const allModels = ref<any[]>([])
const modelsLoaded = ref(false)

async function loadAllModels() {
  try {
    const res = await getModelRegistry()
    allModels.value = res.models || []
  } catch (e) {
    message.error('加载模型列表失败')
  } finally {
    modelsLoaded.value = true
  }
}

// Listen for model.activation_failed — pipeline.reload_anomaly_model returned
// False, so despite the registry showing "activated" the camera is still
// running the previous engine (atomic swap kept the old one).  Warn the
// operator explicitly so they don't assume the new version is live.
useWebSocket({
  topics: ['models'],
  onMessage: (_topic, data) => {
    const payload = data as {
      event?: string
      camera_id?: string
      attempted_version?: string | null
      current_version?: string | null
    }
    if (payload?.event !== 'model.activation_failed') return
    const cam = payload.camera_id ?? '?'
    const attempted = payload.attempted_version ?? '新版本'
    const current = payload.current_version ?? '旧版本'
    message.warning({
      content: `摄像头 ${cam} 激活 ${attempted} 失败，仍在使用 ${current}，请检查模型文件`,
      duration: 8,
    })
    loadAllModels()
  },
})

const totalModels = computed(() => allModels.value.length)
const activeModels = computed(() => allModels.value.filter((m: any) => m.is_active).length)
const pipelineModels = computed(() => allModels.value.filter((m: any) =>
  m.stage && m.stage !== 'retired' && m.stage !== 'production'
).length)
const focusedModel = computed(() => {
  const vid = props.focusVersionId
  if (!vid) return null
  return allModels.value.find((m: any) => m.model_version_id === vid) || null
})

function handleModelsChanged() {
  loadAllModels()
}

onMounted(loadAllModels)
</script>

<template>
  <Alert
    v-if="focusVersionId && modelsLoaded"
    :type="focusedModel ? 'info' : 'warning'"
    show-icon
    style="margin-bottom: 12px"
    :message="focusedModel ? '已定位触发模型' : '未找到模型版本'"
    :description="focusedModel
      ? `${focusedModel.model_version_id} · ${focusedModel.camera_id} · ${focusedModel.stage}`
      : `当前注册表中没有 ${focusVersionId}，可能已被清理或来自旧数据。`"
  />

  <!-- Summary stats -->
  <div style="display: flex; gap: 16px; margin-bottom: 16px">
    <Card size="small" style="flex: 1; text-align: center">
      <Statistic title="模型总数" :value="totalModels" />
    </Card>
    <Card size="small" style="flex: 1; text-align: center">
      <Statistic title="活跃模型" :value="activeModels" :value-style="{ color: '#15a34a' }" />
    </Card>
    <Card size="small" style="flex: 1; text-align: center">
      <Statistic title="流水线中" :value="pipelineModels" :value-style="{ color: '#3b82f6' }" />
    </Card>
  </div>

  <!-- Unified Model Table (pipeline steps + table) -->
  <ModelTable
    :models="allModels"
    :cameras="cameras"
    :focused-version-id="focusVersionId"
    @changed="handleModelsChanged"
  />

  <!-- Event Log (collapsed) -->
  <div style="margin-bottom: 16px">
    <EventLog />
  </div>

  <!-- Batch Inference (collapsed) -->
  <BatchInference :cameras="cameras" />
</template>
