<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  Card, Button, Space, Progress, Tag, Modal, message,
} from 'ant-design-vue'
import {
  PauseCircleOutlined, CheckCircleOutlined, CloseCircleOutlined, LoadingOutlined,
  StopOutlined, CaretRightOutlined,
} from '@ant-design/icons-vue'

import { pauseCaptureJob, resumeCaptureJob, abortCaptureJob } from '../../api'
import { JOB_STATUS_MAP } from '../../composables/useModelState'
import { useBaselineStore } from '../../stores/useBaselineStore'
import { extractErrorMessage } from '../../utils/error'
import type { CameraSummary, TaskInfo } from '../../types/api'

import BaselineTable from './BaselineTable.vue'
import BaselineVersionDrawer from './BaselineVersionDrawer.vue'
import BaselineCaptureModals from './BaselineCaptureModals.vue'

defineOptions({ name: 'BaselineTab' })

const props = defineProps<{
  cameras: CameraSummary[]
  captureTasks: TaskInfo[]
  taskTitle: (task: TaskInfo) => string
  taskProgressStatus: (task: TaskInfo) => 'success' | 'exception' | 'normal' | 'active'
  canPauseTask: (task: TaskInfo) => boolean
  canResumeTask: (task: TaskInfo) => boolean
  canAbortTask: (task: TaskInfo) => boolean
  canDismissTask: (task: TaskInfo) => boolean
  handleDismissTask: (taskId: string) => void
  loadTasks: () => void
}>()

const store = useBaselineStore()

const captureVisible = ref(false)
const advCaptureVisible = ref(false)

async function handlePauseJob(taskId: string) {
  try {
    await pauseCaptureJob(taskId)
    message.success('任务已暂停')
    props.loadTasks()
  } catch (e) {
    message.error(extractErrorMessage(e, '暂停失败'))
  }
}

async function handleResumeJob(taskId: string) {
  try {
    await resumeCaptureJob(taskId)
    message.success('任务已恢复')
    props.loadTasks()
  } catch (e) {
    message.error(extractErrorMessage(e, '恢复失败'))
  }
}

function handleAbortJob(taskId: string) {
  Modal.confirm({
    title: '确认中止采集任务？',
    content: '已采集的帧将被保留，但任务不可恢复。',
    okText: '中止',
    okType: 'danger',
    cancelText: '取消',
    async onOk() {
      try {
        await abortCaptureJob(taskId)
        message.success('任务已中止')
        props.loadTasks()
      } catch (e) {
        message.error(extractErrorMessage(e, '中止失败'))
      }
    },
  })
}

// Re-export for the parent Models page which calls these via ref().
defineExpose({
  loadBaselines: () => store.loadBaselines(),
  loadCameraGroups: () => store.loadCameraGroups(),
})

onMounted(() => {
  store.loadBaselines()
  store.loadCameraGroups()
})
</script>

<template>
  <!-- Active capture tasks -->
  <div v-if="captureTasks.length > 0" style="margin-bottom: 16px">
    <Card
      v-for="task in captureTasks"
      :key="task.task_id"
      size="small"
      style="margin-bottom: 8px"
    >
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px">
        <Space>
          <PauseCircleOutlined v-if="task.status === 'paused'" style="color: #d97706" />
          <LoadingOutlined v-else-if="task.status === 'running'" spin style="color: #3b82f6" />
          <CheckCircleOutlined v-else-if="task.status === 'complete'" style="color: #15a34a" />
          <CloseCircleOutlined v-else-if="task.status === 'failed'" style="color: #e5484d" />
          <StopOutlined v-else-if="task.status === 'aborted'" style="color: #8890a0" />
          <span style="font-weight: 500">{{ taskTitle(task) }}</span>
          <span v-if="task.camera_id" style="color: #8890a0">{{ task.camera_id }}</span>
          <Tag :color="(JOB_STATUS_MAP[task.status] || {}).color || 'default'">
            {{ (JOB_STATUS_MAP[task.status] || {}).text || task.status }}
          </Tag>
        </Space>
        <Space>
          <Button v-if="canPauseTask(task)" size="small" @click="handlePauseJob(task.task_id)">
            <template #icon><PauseCircleOutlined /></template>
            暂停
          </Button>
          <Button v-if="canResumeTask(task)" size="small" type="primary" @click="handleResumeJob(task.task_id)">
            <template #icon><CaretRightOutlined /></template>
            恢复
          </Button>
          <Button v-if="canAbortTask(task)" size="small" danger @click="handleAbortJob(task.task_id)">
            <template #icon><StopOutlined /></template>
            中止
          </Button>
          <Button v-if="canDismissTask(task)" size="small" @click="handleDismissTask(task.task_id)">关闭</Button>
        </Space>
      </div>
      <Progress :percent="task.progress" :status="taskProgressStatus(task)" size="small" />
      <div style="font-size: 12px; color: #8890a0; margin-top: 4px">{{ task.message }}</div>
      <div v-if="task.error" style="font-size: 12px; color: #e5484d; margin-top: 4px">{{ task.error }}</div>
    </Card>
  </div>

  <BaselineTable
    @open-capture="captureVisible = true"
    @open-advanced-capture="advCaptureVisible = true"
  />

  <BaselineVersionDrawer />

  <BaselineCaptureModals
    :cameras="cameras"
    v-model:capture-visible="captureVisible"
    v-model:advanced-visible="advCaptureVisible"
    @capture-started="loadTasks"
  />
</template>
