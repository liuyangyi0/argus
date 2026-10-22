<script setup lang="ts">
import { onMounted } from 'vue'
import { useModelState } from '../../composables/useModelState'
import BaselineTab from '../../components/models/BaselineTab.vue'
import ModelCameraState from '../../components/models/ModelCameraState.vue'
import SceneWorkflowPanel from '../../components/models/SceneWorkflowPanel.vue'
import StatusBanner from '../../components/common/StatusBanner.vue'

const {
  cameras,
  loadCameras,
  camerasLoading,
  camerasLoaded,
  camerasError,
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

onMounted(async () => {
  await loadCameras()
  loadTasks()
})
</script>

<template>
  <ModelCameraState
    :loading="camerasLoading"
    :loaded="camerasLoaded"
    :error="camerasError"
    :camera-count="cameras.length"
    @retry="loadCameras"
  >
    <SceneWorkflowPanel :cameras="cameras" />
    <StatusBanner
      class="legacy-baseline-notice"
      status="warning"
      title="下方旧版采集仅用于诊断与历史兼容"
      description="旧图片采集、合并与随机拆分不生成正式场景 v3 证据，不能直接用于 production 发布。请优先使用上方场景工作流；现有发布安全门保持不变。"
    />
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
  </ModelCameraState>
</template>

<style scoped>
.legacy-baseline-notice { margin-bottom: 16px; }
</style>
