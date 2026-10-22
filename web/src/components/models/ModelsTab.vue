<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Statistic, Card, message, Alert, Button } from 'ant-design-vue'
import { getModelRegistry, getReleaseReadiness } from '../../api'
import { useWebSocket } from '../../composables/useWebSocket'
import ModelTable from './ModelTable.vue'
import EventLog from './EventLog.vue'
import BatchInference from './BatchInference.vue'
import AsyncState from '../common/AsyncState.vue'
import StatusBanner from '../common/StatusBanner.vue'
import { toApiError } from '../../utils/error'
import type { CameraSummary, ModelInfo, ReleaseReadiness } from '../../types/api'

const props = defineProps<{
  cameras: CameraSummary[]
  camerasLoading?: boolean
  camerasLoaded?: boolean
  camerasError?: string
  focusVersionId?: string
}>()

const emit = defineEmits<{ retryCameras: [] }>()

const allModels = ref<ModelInfo[]>([])
const modelsLoaded = ref(false)
const modelsLoading = ref(true)
const modelsError = ref('')
const modelsErrorStatus = ref<number | null>(null)
const releaseReadiness = ref<Record<string, ReleaseReadiness>>({})
const releaseReadinessError = ref('')
let registryRequestRevision = 0
const initialModelsError = computed(() => (
  modelsError.value
    && allModels.value.length === 0
    && ![403, 404].includes(modelsErrorStatus.value ?? 0)
    ? modelsError.value
    : ''
))
const refreshModelsError = computed(() => (
  modelsError.value && allModels.value.length > 0 ? modelsError.value : ''
))

async function loadAllModels() {
  const requestRevision = ++registryRequestRevision
  modelsLoading.value = true
  modelsError.value = ''
  modelsErrorStatus.value = null
  releaseReadinessError.value = ''
  try {
    const res = await getModelRegistry()
    const models = res.models || []
    if (requestRevision !== registryRequestRevision) return
    allModels.value = models

    // Production controls fail closed.  Readiness is an independent,
    // server-authoritative read: a failed request never enables an action and
    // is shown separately from a successful registry query.
    const readinessTargets = models.filter(model =>
      model.stage === 'canary' || model.stage === 'production'
    )
    const results = await Promise.allSettled(
      readinessTargets.map(model => getReleaseReadiness(model.model_version_id)),
    )
    if (requestRevision !== registryRequestRevision) return
    const nextReadiness: Record<string, ReleaseReadiness> = {}
    let firstReadinessError = ''
    results.forEach((result, index) => {
      const model = readinessTargets[index]
      if (!model) return
      if (result.status === 'fulfilled') {
        nextReadiness[model.model_version_id] = result.value
      } else if (!firstReadinessError) {
        firstReadinessError = toApiError(
          result.reason,
          `模型 ${model.model_version_id} 的发布 readiness 获取失败`,
        ).message
      }
    })
    releaseReadiness.value = nextReadiness
    releaseReadinessError.value = firstReadinessError
  } catch (e) {
    if (requestRevision !== registryRequestRevision) return
    const apiError = toApiError(e, '加载模型列表失败')
    modelsError.value = apiError.message
    modelsErrorStatus.value = apiError.httpStatus
    if ([403, 404].includes(apiError.httpStatus ?? 0)) allModels.value = []
  } finally {
    if (requestRevision !== registryRequestRevision) return
    modelsLoaded.value = true
    modelsLoading.value = false
  }
}

// Reconcile every registry/release mutation immediately, including deletions
// from another browser. Activation failures also need an explicit warning:
// the atomic swap kept the old engine, not the attempted new version.
useWebSocket({
  // model_release resync_required must have an authoritative REST loader too;
  // listening only to models left a dropped release event unreconciled.
  topics: ['models', 'model_release'],
  onMessage: (topic, data, event) => {
    const payload = data as {
      event?: string
      type?: string
      camera_id?: string
      attempted_version?: string | null
      current_version?: string | null
    }
    const eventType = payload?.type ?? payload?.event ?? event?.type
    if (eventType === 'model.activation_failed') {
      const cam = payload?.camera_id ?? '?'
      const attempted = payload?.attempted_version ?? '新版本'
      const current = payload?.current_version ?? '旧版本'
      message.warning({
        content: `摄像头 ${cam} 激活 ${attempted} 失败，仍在使用 ${current}，请检查模型文件`,
        duration: 8,
      })
    }
    if (topic === 'models' || topic === 'model_release') void loadAllModels()
  },
  fallbackPoll: loadAllModels,
})

const totalModels = computed(() => allModels.value.length)
const activeModels = computed(() => allModels.value.filter(m => m.is_active).length)
const pipelineModels = computed(() => allModels.value.filter(m =>
  m.stage && m.stage !== 'retired' && m.stage !== 'production'
).length)
const focusedModel = computed(() => {
  const vid = props.focusVersionId
  if (!vid) return null
  return allModels.value.find(m => m.model_version_id === vid) || null
})

function handleModelsChanged() {
  loadAllModels()
}

onMounted(loadAllModels)
</script>

<template>
  <StatusBanner
    v-if="refreshModelsError"
    class="models-state-banner"
    status="error"
    title="模型列表刷新失败，当前显示上次成功获取的数据"
    :description="refreshModelsError"
  >
    <template #action><Button size="small" @click="loadAllModels">重试</Button></template>
  </StatusBanner>

  <StatusBanner
    v-if="releaseReadinessError"
    class="models-state-banner"
    status="warning"
    title="发布 readiness 获取失败，生产操作保持 NO-GO"
    :description="releaseReadinessError"
  >
    <template #action><Button size="small" :loading="modelsLoading" @click="loadAllModels">重试</Button></template>
  </StatusBanner>

  <StatusBanner
    v-if="camerasLoading && modelsLoaded"
    class="models-state-banner"
    status="loading"
    title="正在同步摄像头配置"
  />

  <StatusBanner
    v-else-if="camerasError"
    class="models-state-banner"
    status="warning"
    :title="cameras.length ? '摄像头列表刷新失败' : '摄像头列表不可用'"
    :description="camerasError"
  >
    <template #action><Button size="small" :loading="camerasLoading" @click="emit('retryCameras')">重试</Button></template>
  </StatusBanner>

  <StatusBanner
    v-else-if="camerasLoaded && cameras.length === 0"
    class="models-state-banner"
    status="neutral"
    title="尚未配置摄像头"
    description="可以查看已有模型，但发布和批量推理操作暂不可用。"
  />

  <Alert
    v-if="focusVersionId && modelsLoaded && !modelsError"
    :type="focusedModel ? 'info' : 'warning'"
    show-icon
    style="margin-bottom: 12px"
    :message="focusedModel ? '已定位触发模型' : '未找到模型版本'"
    :description="focusedModel
      ? `${focusedModel.model_version_id} · ${focusedModel.camera_id} · ${focusedModel.stage}`
      : `当前注册表中没有 ${focusVersionId}，可能已被清理或来自旧数据。`"
  />

  <AsyncState
    :loading="modelsLoading && !modelsLoaded"
    :refreshing="modelsLoading && modelsLoaded"
    :error="initialModelsError"
    :forbidden="modelsErrorStatus === 403"
    :not-found="modelsErrorStatus === 404"
    :empty="modelsLoaded && !modelsError && allModels.length === 0"
    skeleton-type="table"
    :skeleton-rows="7"
    empty-title="暂无注册的模型版本"
    empty-description="完成训练并登记制品后，模型版本会出现在这里。"
    @retry="loadAllModels"
  >
    <!-- Summary stats -->
    <div class="models-stat-grid">
      <Card size="small">
        <Statistic title="模型总数" :value="totalModels" />
      </Card>
      <Card size="small">
        <Statistic title="活跃模型" :value="activeModels" :value-style="{ color: '#166534' }" />
      </Card>
      <Card size="small">
        <Statistic title="流水线中" :value="pipelineModels" :value-style="{ color: '#1d4ed8' }" />
      </Card>
    </div>

    <!-- Unified Model Table (pipeline steps + table) -->
    <ModelTable
      :models="allModels"
      :cameras="cameras"
      :release-readiness="releaseReadiness"
      :focused-version-id="focusVersionId"
      @changed="handleModelsChanged"
    />

    <!-- Event Log (collapsed) -->
    <div style="margin-bottom: 16px">
      <EventLog />
    </div>

    <!-- Batch Inference (collapsed) -->
    <BatchInference :cameras="cameras" />
  </AsyncState>
</template>

<style scoped>
.models-state-banner {
  margin-bottom: 12px;
}

.models-stat-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 16px;
  text-align: center;
}

@media (max-width: 767px) {
  .models-stat-grid {
    grid-template-columns: 1fr;
    gap: 8px;
  }
}
</style>
