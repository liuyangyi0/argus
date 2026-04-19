<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { Tabs, Typography } from 'ant-design-vue'
import { useModelState } from '../composables/useModelState'
import BaselineTab from '../components/models/BaselineTab.vue'
import TrainingTab from '../components/models/TrainingTab.vue'
import ModelsTab from '../components/models/ModelsTab.vue'
import ModelComparison from '../components/models/ModelComparison.vue'
import LabelingTab from '../components/models/LabelingTab.vue'
import ThresholdPreview from '../components/models/ThresholdPreview.vue'

const route = useRoute()
const {
  cameras,
  loadCameras,
  loadTasks,
  captureTasks,
  taskTitle,
  taskProgressStatus,
  canPauseTask,
  canResumeTask,
  canAbortTask,
  canDismissTask,
  handleDismissTask,
} = useModelState()

// Determine initial tab from query param (supports redirect from /training)
const initialTab = route.query.tab === 'training' ? 'training'
  : route.query.tab === 'models' ? 'models'
  : route.query.tab === 'comparison' ? 'comparison'
  : route.query.tab === 'labeling' ? 'labeling'
  : 'baselines'
const activeTab = ref(initialTab)

function onTabChange(key: string | number) {
  activeTab.value = String(key)
}

onMounted(async () => {
  await loadCameras()
  loadTasks()
})
</script>

<template>
  <main class="glass" style="padding: 24px; border-radius: var(--r-lg); min-width: 0; display: flex; flex-direction: column; flex: 1;">
    <Typography.Title :level="3" style="margin-bottom: 24px; color: var(--ink)">模型管理</Typography.Title>

    <Tabs :activeKey="activeTab" @change="onTabChange">
      <Tabs.TabPane key="baselines" tab="基线管理">
        <BaselineTab
          :cameras="cameras"
          :capture-tasks="captureTasks"
          :task-title="taskTitle"
          :task-progress-status="taskProgressStatus"
          :can-pause-task="canPauseTask"
          :can-resume-task="canResumeTask"
          :can-abort-task="canAbortTask"
          :can-dismiss-task="canDismissTask"
          :handle-dismiss-task="handleDismissTask"
          :load-tasks="loadTasks"
        />
      </Tabs.TabPane>

      <Tabs.TabPane key="training" tab="训练与评估">
        <TrainingTab :cameras="cameras" />
      </Tabs.TabPane>

      <Tabs.TabPane key="models" tab="模型与发布">
        <ModelsTab :cameras="cameras" />
      </Tabs.TabPane>

      <Tabs.TabPane key="comparison" tab="A/B 对比">
        <ModelComparison :cameras="cameras" />
      </Tabs.TabPane>

      <Tabs.TabPane key="labeling" tab="标注队列">
        <LabelingTab :cameras="cameras" />
      </Tabs.TabPane>

      <Tabs.TabPane key="threshold" tab="阈值预览">
        <ThresholdPreview :cameras="cameras" />
      </Tabs.TabPane>
    </Tabs>
  </main>
</template>
