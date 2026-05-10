<script setup lang="ts">
import { onMounted } from 'vue'
import { useModelState } from '../../composables/useModelState'
import BaselineTab from '../../components/models/BaselineTab.vue'

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

onMounted(async () => {
  await loadCameras()
  loadTasks()
})
</script>

<template>
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
</template>
