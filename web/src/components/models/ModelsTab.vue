<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Statistic, Card } from 'ant-design-vue'
import { getModelRegistry } from '../../api'
import ModelRegistry from './ModelRegistry.vue'
import ReleasePipeline from './ReleasePipeline.vue'
import EventLog from './EventLog.vue'
import BatchInference from './BatchInference.vue'

const props = defineProps<{
  cameras: any[]
}>()

const allModels = ref<any[]>([])

async function loadAllModels() {
  try {
    const res = await getModelRegistry()
    allModels.value = res.data.models || []
  } catch (e) {
    console.error('Failed to load models', e)
  }
}

const totalModels = computed(() => allModels.value.length)
const activeModels = computed(() => allModels.value.filter((m: any) => m.is_active).length)
const pipelineModels = computed(() => allModels.value.filter((m: any) =>
  m.stage && m.stage !== 'retired' && m.stage !== 'production'
).length)

function handleModelsChanged() {
  loadAllModels()
}

onMounted(loadAllModels)
</script>

<template>
  <!-- Summary stats -->
  <div style="display: flex; gap: 16px; margin-bottom: 16px">
    <Card size="small" style="flex: 1; text-align: center">
      <Statistic title="模型总数" :value="totalModels" />
    </Card>
    <Card size="small" style="flex: 1; text-align: center">
      <Statistic title="活跃模型" :value="activeModels" value-style="color: #52c41a" />
    </Card>
    <Card size="small" style="flex: 1; text-align: center">
      <Statistic title="流水线中" :value="pipelineModels" value-style="color: #3b82f6" />
    </Card>
  </div>

  <!-- Model Registry -->
  <ModelRegistry :models="allModels" @changed="handleModelsChanged" />

  <!-- Release Pipeline -->
  <ReleasePipeline :cameras="cameras" :models="allModels" @changed="handleModelsChanged" />

  <!-- Event Log (collapsed) -->
  <div style="margin-bottom: 16px">
    <EventLog />
  </div>

  <!-- Batch Inference (collapsed) -->
  <BatchInference :cameras="cameras" />
</template>
