import { ref, computed } from 'vue'
import { message } from 'ant-design-vue'
import { getCameras, getTasks, dismissTask } from '../api'
import { useWebSocket } from './useWebSocket'
import type { CameraSummary, TaskInfo } from '../types/api'

// ── Constants ──

export const MODEL_TYPES = [
  { value: 'patchcore', label: 'PatchCore' },
  { value: 'efficient_ad', label: 'EfficientAD' },
  { value: 'fastflow', label: 'FastFlow' },
  { value: 'padim', label: 'PaDiM' },
  { value: 'dinomaly2', label: 'Dinomaly2' },
]

export const SESSION_LABELS = [
  { value: 'daytime', label: '白天/日间' },
  { value: 'night', label: '夜间' },
  { value: 'maintenance', label: '检修期间' },
  { value: 'custom', label: '自定义' },
]

export const SAMPLING_STRATEGIES = [
  { value: 'uniform', label: '均匀采样', desc: '按固定间隔抽帧，简单可靠' },
  { value: 'active', label: '主动采样（推荐）', desc: 'DINOv2 特征去冗余，自动过滤相似帧' },
  { value: 'scheduled', label: '定时采样', desc: '按昼夜时段自动采集，内含主动去冗余' },
]

export const JOB_STATUS_MAP: Record<string, { text: string; color: string }> = {
  pending: { text: '等待中', color: 'default' },
  running: { text: '运行中', color: 'processing' },
  paused: { text: '已暂停', color: 'warning' },
  complete: { text: '已完成', color: 'success' },
  failed: { text: '失败', color: 'error' },
  aborted: { text: '已中止', color: 'default' },
}

export const GRADE_COLORS: Record<string, string> = {
  A: 'green', B: 'blue', C: 'orange', F: 'red',
}

export const RECOMMENDATION_COLORS: Record<string, string> = {
  deploy: 'green', keep_old: 'orange', review: 'blue',
}

export const RECOMMENDATION_TEXT: Record<string, string> = {
  deploy: '推荐部署新模型', keep_old: '保留旧模型', review: '需要人工审核',
}

export const BASELINE_STATE_MAP: Record<string, { text: string; color: string }> = {
  draft: { text: '草稿', color: 'default' },
  verified: { text: '已审核', color: 'blue' },
  active: { text: '生产中', color: 'green' },
  retired: { text: '已退役', color: '' },
}

export const STAGE_MAP: Record<string, { text: string; color: string }> = {
  candidate: { text: '候选', color: 'blue' },
  shadow: { text: '影子', color: 'purple' },
  canary: { text: '金丝雀', color: 'orange' },
  production: { text: '生产', color: 'green' },
  retired: { text: '退役', color: 'default' },
}

export const VALID_TRANSITIONS: Record<string, string[]> = {
  candidate: ['shadow'],
  shadow: ['canary', 'candidate'],
  canary: ['production', 'shadow'],
  production: [],
}

export const STAGE_LABELS: Record<string, string> = {
  shadow: '推进到影子模式',
  canary: '推进到金丝雀',
  production: '推进到生产',
  candidate: '回退到候选',
}

export const TRAINING_STATUS_MAP: Record<string, { text: string; color: string }> = {
  pending_confirmation: { text: '待确认', color: 'orange' },
  queued: { text: '排队中', color: 'blue' },
  running: { text: '训练中', color: 'processing' },
  validating: { text: '验证中', color: 'cyan' },
  complete: { text: '完成', color: 'green' },
  failed: { text: '失败', color: 'red' },
  rejected: { text: '已拒绝', color: 'default' },
}

export const JOB_TYPE_LABELS: Record<string, string> = {
  ssl_backbone: 'SSL 骨干',
  anomaly_head: '异常检测头',
}

export const TRIGGER_LABELS: Record<string, string> = {
  manual: '手动',
  drift_suggested: '漂移建议',
  scheduled: '定时',
}

// ── Shared state composable ──

export function useModelState() {
  const cameras = ref<CameraSummary[]>([])
  const tasks = ref<TaskInfo[]>([])

  function upsertTaskUpdate(task: TaskInfo) {
    const index = tasks.value.findIndex(existing => existing.task_id === task.task_id)
    if (index >= 0) {
      tasks.value[index] = { ...tasks.value[index], ...task }
      tasks.value = [...tasks.value]
      return
    }
    tasks.value = [task, ...tasks.value]
  }

  useWebSocket({
    topics: ['tasks'],
    onMessage: (_topic, data) => {
      if (Array.isArray(data?.tasks)) {
        tasks.value = data.tasks
        return
      }
      if (data?.task_id) {
        upsertTaskUpdate(data)
      }
    },
    fallbackPoll: loadTasks,
    fallbackInterval: 5000,
  })

  async function loadCameras() {
    try {
      const res = await getCameras()
      cameras.value = res.cameras || []
    } catch (e) {
      message.error('加载摄像头列表失败')
    }
  }

  async function loadTasks() {
    try {
      const res = await getTasks()
      tasks.value = res.tasks || []
    } catch (e) {
      message.error('加载任务列表失败')
    }
  }

  async function handleDismissTask(taskId: string) {
    try {
      await dismissTask(taskId)
      loadTasks()
    } catch (e) {
      message.error('关闭任务失败')
    }
  }

  const captureTasks = computed(() =>
    tasks.value.filter(t => t.task_type === 'baseline_capture' || t.task_type === 'baseline_capture_job')
  )
  const trainingTasks = computed(() =>
    tasks.value.filter(t => t.task_type === 'model_training')
  )

  function taskTitle(task: TaskInfo) {
    if (task.task_type === 'model_training') return '模型训练'
    if (task.task_type === 'baseline_capture') return '基线采集'
    return task.task_type
  }

  function taskProgressStatus(task: TaskInfo) {
    if (task.status === 'failed') return 'exception'
    if (task.status === 'complete') return 'success'
    if (task.status === 'paused') return 'normal'
    return 'active'
  }

  function canPauseTask(task: TaskInfo) {
    return task.task_type === 'baseline_capture' && task.status === 'running'
  }

  function canResumeTask(task: TaskInfo) {
    return task.task_type === 'baseline_capture' && task.status === 'paused'
  }

  function canAbortTask(task: TaskInfo) {
    return task.task_type === 'baseline_capture' && (task.status === 'running' || task.status === 'paused')
  }

  function canDismissTask(task: TaskInfo) {
    return task.status === 'complete' || task.status === 'failed' || task.status === 'aborted'
  }

  return {
    cameras,
    tasks,
    loadCameras,
    loadTasks,
    handleDismissTask,
    captureTasks,
    trainingTasks,
    taskTitle,
    taskProgressStatus,
    canPauseTask,
    canResumeTask,
    canAbortTask,
    canDismissTask,
  }
}
