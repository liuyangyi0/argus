<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import {
  Tabs, Table, Card, Button, Select, Form, InputNumber, Space,
  Typography, Progress, Tag, Modal, message, Descriptions, Tooltip,
  Radio, Collapse, Slider, Drawer, Input, Divider,
} from 'ant-design-vue'
import {
  PlayCircleOutlined, RocketOutlined, ReloadOutlined,
  CheckCircleOutlined, CloseCircleOutlined, LoadingOutlined,
  HistoryOutlined, ExperimentOutlined, SwapOutlined,
  RollbackOutlined, CheckOutlined, PauseCircleOutlined,
  StopOutlined, CaretRightOutlined, SafetyCertificateOutlined,
  TeamOutlined, MergeCellsOutlined, DeleteOutlined,
} from '@ant-design/icons-vue'
import {
  getCameras, getBaselines, startCapture, startTraining,
  getModels, deployModel, getTrainingHistory, getTasks, dismissTask,
  optimizeBaseline, previewOptimize, getModelRegistry, activateModel,
  rollbackModel, compareModels, batchInference,
  startCaptureJob, pauseCaptureJob, resumeCaptureJob, abortCaptureJob,
  getBaselineVersions, verifyBaseline, activateBaseline, retireBaseline, deleteBaselineVersion,
  getCameraGroups, mergeGroupBaseline, mergeFalsePositives,
  promoteModel, retireModel, getStageHistory, getVersionEvents, getShadowReport,
} from '../api'
import { useWebSocket } from '../composables/useWebSocket'

// ── Shared state ──
const activeTab = ref('baselines')
const cameras = ref<any[]>([])

// ── Tab 1: Baselines ──
const baselines = ref<any[]>([])
const baselinesLoading = ref(false)
const captureModalVisible = ref(false)
const captureForm = ref({ camera_id: '', count: 100, interval: 2.0, session_label: 'daytime' })
const captureSubmitting = ref(false)

// ── Advanced capture job ──
const advCaptureVisible = ref(false)
const advCaptureForm = ref({
  camera_id: '',
  target_frames: 1000,
  duration_hours: 24,
  sampling_strategy: 'active',
  diversity_threshold: 0.3,
  frames_per_period: 50,
})
const advCaptureSubmitting = ref(false)

// ── Baseline optimization ──
const optimizingBaseline = ref<string | null>(null)

// ── Tab 2: Training ──
const trainForm = ref({ camera_id: '', model_type: 'patchcore', export_format: 'openvino' })
const trainSubmitting = ref(false)
const tasks = ref<any[]>([])

// ── Tab 3: Models ──
const models = ref<any[]>([])
const modelsLoading = ref(false)
const trainingHistory = ref<any[]>([])
const historyLoading = ref(false)
const deployingModel = ref<string | null>(null)
const historyDetailVisible = ref(false)
const historyDetail = ref<any>(null)

// ── Tab 3: Model Registry ──
const registry = ref<any[]>([])
const registryLoading = ref(false)
const activatingModel = ref<string | null>(null)

// ── Tab 3: Compare ──
const compareVisible = ref(false)
const compareForm = ref({ old_record_id: undefined as number | undefined, new_record_id: undefined as number | undefined })
const compareResult = ref<any>(null)
const comparing = ref(false)

// ── Baseline Lifecycle ──
const versionDrawerVisible = ref(false)
const versionDrawerCamera = ref('')
const baselineVersions = ref<any[]>([])
const versionsLoading = ref(false)
const verifyingVersion = ref<string | null>(null)
const activatingVersion = ref<string | null>(null)

// ── Camera Groups ──
const cameraGroups = ref<any[]>([])
const groupsLoading = ref(false)
const mergingGroup = ref<string | null>(null)

// ── FP Merge ──
const mergingFP = ref<string | null>(null)
const deletingBaseline = ref<string | null>(null)

function upsertTaskUpdate(task: any) {
  const index = tasks.value.findIndex(existing => existing.task_id === task.task_id)
  if (index >= 0) {
    tasks.value[index] = { ...tasks.value[index], ...task }
    tasks.value = [...tasks.value]
    return
  }
  tasks.value = [task, ...tasks.value]
}

// ── WebSocket for tasks ──
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

const MODEL_TYPES = [
  { value: 'patchcore', label: 'PatchCore' },
  { value: 'efficient_ad', label: 'EfficientAD' },
  { value: 'fastflow', label: 'FastFlow' },
  { value: 'padim', label: 'PaDiM' },
  { value: 'dinomaly2', label: 'Dinomaly2' },
]

const EXPORT_FORMATS = [
  { value: 'openvino', label: 'OpenVINO (recommended)' },
  { value: 'onnx', label: 'ONNX' },
  { value: 'none', label: 'None (PyTorch only)' },
]

const SESSION_LABELS = [
  { value: 'daytime', label: '白天/日间' },
  { value: 'night', label: '夜间' },
  { value: 'maintenance', label: '检修期间' },
  { value: 'custom', label: '自定义' },
]

const SAMPLING_STRATEGIES = [
  { value: 'uniform', label: '均匀采样', desc: '按固定间隔抽帧，简单可靠' },
  { value: 'active', label: '主动采样（推荐）', desc: 'DINOv2 特征去冗余，自动过滤相似帧' },
  { value: 'scheduled', label: '定时采样', desc: '按昼夜时段自动采集，内含主动去冗余' },
]

const JOB_STATUS_MAP: Record<string, { text: string; color: string }> = {
  pending: { text: '等待中', color: 'default' },
  running: { text: '运行中', color: 'processing' },
  paused: { text: '已暂停', color: 'warning' },
  complete: { text: '已完成', color: 'success' },
  failed: { text: '失败', color: 'error' },
  aborted: { text: '已中止', color: 'default' },
}

function taskProgressStatus(task: any) {
  if (task.status === 'failed') return 'exception'
  if (task.status === 'complete') return 'success'
  if (task.status === 'paused') return 'normal'
  return 'active'
}

function taskTitle(task: any) {
  if (task.task_type === 'model_training') return '模型训练'
  if (task.task_type === 'baseline_capture') return '基线采集'
  return task.task_type
}

function canPauseTask(task: any) {
  return task.task_type === 'baseline_capture' && task.status === 'running'
}

function canResumeTask(task: any) {
  return task.task_type === 'baseline_capture' && task.status === 'paused'
}

function canAbortTask(task: any) {
  return task.task_type === 'baseline_capture' && (task.status === 'running' || task.status === 'paused')
}

function canDismissTask(task: any) {
  return task.status === 'complete' || task.status === 'failed' || task.status === 'aborted'
}

const GRADE_COLORS: Record<string, string> = {
  A: 'green', B: 'blue', C: 'orange', F: 'red',
}

const RECOMMENDATION_COLORS: Record<string, string> = {
  deploy: 'green', keep_old: 'orange', review: 'blue',
}
const RECOMMENDATION_TEXT: Record<string, string> = {
  deploy: '推荐部署新模型', keep_old: '保留旧模型', review: '需要人工审核',
}

const BASELINE_STATE_MAP: Record<string, { text: string; color: string }> = {
  draft: { text: '草稿', color: 'default' },
  verified: { text: '已审核', color: 'blue' },
  active: { text: '生产中', color: 'green' },
  retired: { text: '已退役', color: '' },
}

// ── Data loading ──

async function loadCameras() {
  try {
    const res = await getCameras()
    cameras.value = res.data.cameras || []
    if (cameras.value.length > 0 && !captureForm.value.camera_id) {
      captureForm.value.camera_id = cameras.value[0].camera_id
      trainForm.value.camera_id = cameras.value[0].camera_id
    }
  } catch (e) {
    console.error('Failed to load cameras', e)
  }
}

async function loadBaselines() {
  baselinesLoading.value = true
  try {
    const res = await getBaselines()
    baselines.value = res.data.baselines || []
  } catch (e) {
    console.error('Failed to load baselines', e)
  } finally {
    baselinesLoading.value = false
  }
}

async function loadTasks() {
  try {
    const res = await getTasks()
    tasks.value = res.data.tasks || []
  } catch (e) {
    console.error('Failed to load tasks', e)
  }
}

async function loadModels() {
  modelsLoading.value = true
  try {
    const res = await getModels()
    models.value = res.data.models || []
  } catch (e) {
    console.error('Failed to load models', e)
  } finally {
    modelsLoading.value = false
  }
}

async function loadHistory() {
  historyLoading.value = true
  try {
    const res = await getTrainingHistory()
    trainingHistory.value = res.data.records || []
  } catch (e) {
    console.error('Failed to load training history', e)
  } finally {
    historyLoading.value = false
  }
}

async function loadRegistry() {
  registryLoading.value = true
  try {
    const res = await getModelRegistry()
    registry.value = res.data.models || []
  } catch (e) {
    console.error('Failed to load model registry', e)
  } finally {
    registryLoading.value = false
  }
}

async function loadBaselineVersions(cameraId: string) {
  versionsLoading.value = true
  try {
    const res = await getBaselineVersions({ camera_id: cameraId })
    baselineVersions.value = res.data.versions || []
  } catch (e) {
    console.error('Failed to load baseline versions', e)
    baselineVersions.value = []
  } finally {
    versionsLoading.value = false
  }
}

async function loadCameraGroups() {
  groupsLoading.value = true
  try {
    const res = await getCameraGroups()
    cameraGroups.value = res.data.groups || []
  } catch (e) {
    console.error('Failed to load camera groups', e)
  } finally {
    groupsLoading.value = false
  }
}

// ── Actions ──

function openVersionDrawer(cameraId: string) {
  versionDrawerCamera.value = cameraId
  versionDrawerVisible.value = true
  loadBaselineVersions(cameraId)
}

function handleVerify(record: any) {
  let verifiedBy = ''
  Modal.confirm({
    title: '审核基线版本',
    content: `确认审核通过 ${record.version}？请输入审核人姓名。`,
    okText: '确认审核',
    cancelText: '取消',
    async onOk() {
      verifiedBy = verifiedBy || 'operator'
      verifyingVersion.value = record.version
      try {
        await verifyBaseline({
          camera_id: record.camera_id,
          version: record.version,
          verified_by: verifiedBy,
        })
        message.success(`${record.version} 已审核通过`)
        loadBaselineVersions(versionDrawerCamera.value)
        loadBaselines()
      } catch (e: any) {
        message.error(e.response?.data?.error || '审核失败')
      } finally {
        verifyingVersion.value = null
      }
    },
  })
}

function handleActivateBaseline(record: any) {
  Modal.confirm({
    title: '激活基线版本',
    content: `确定将 ${record.version} 设为生产基线？当前 Active 版本将自动退役。`,
    okText: '确认激活',
    cancelText: '取消',
    async onOk() {
      activatingVersion.value = record.version
      try {
        await activateBaseline({
          camera_id: record.camera_id,
          version: record.version,
        })
        message.success(`${record.version} 已激活`)
        loadBaselineVersions(versionDrawerCamera.value)
        loadBaselines()
      } catch (e: any) {
        message.error(e.response?.data?.error || '激活失败')
      } finally {
        activatingVersion.value = null
      }
    },
  })
}

function handleRetireBaseline(record: any) {
  Modal.confirm({
    title: '退役基线版本',
    content: `确定退役 ${record.version}？退役后将保留数据但不再用于训练。`,
    okText: '确认退役',
    okType: 'danger',
    cancelText: '取消',
    async onOk() {
      try {
        await retireBaseline({
          camera_id: record.camera_id,
          version: record.version,
          reason: '手动退役',
        })
        message.success(`${record.version} 已退役`)
        loadBaselineVersions(versionDrawerCamera.value)
        loadBaselines()
      } catch (e: any) {
        message.error(e.response?.data?.error || '退役失败')
      }
    },
  })
}

function handleDeleteBaseline(record: any) {
  Modal.confirm({
    title: '删除基线版本',
    content: `确定删除 ${record.camera_id} / ${record.version}？该操作会删除磁盘中的整套基线数据。`,
    okText: '确认删除',
    okType: 'danger',
    cancelText: '取消',
    async onOk() {
      deletingBaseline.value = `${record.camera_id}-${record.version}`
      try {
        await deleteBaselineVersion({
          camera_id: record.camera_id,
          version: record.version,
        })
        message.success(`${record.version} 已删除`)
        if (versionDrawerVisible.value && versionDrawerCamera.value === record.camera_id) {
          loadBaselineVersions(versionDrawerCamera.value)
        }
        loadBaselines()
      } catch (e: any) {
        message.error(e.response?.data?.error || '删除失败')
      } finally {
        deletingBaseline.value = null
      }
    },
  })
}

function handleMergeGroup(groupId: string) {
  Modal.confirm({
    title: '合并摄像头组基线',
    content: `将组 ${groupId} 内所有成员摄像头的基线合并为一个组版本（Draft 状态）。`,
    okText: '执行合并',
    cancelText: '取消',
    async onOk() {
      mergingGroup.value = groupId
      try {
        const res = await mergeGroupBaseline({ group_id: groupId })
        message.success(`组基线合并完成: ${res.data.version}, ${res.data.image_count} 张图片`)
        loadCameraGroups()
      } catch (e: any) {
        message.error(e.response?.data?.error || '合并失败')
      } finally {
        mergingGroup.value = null
      }
    },
  })
}

function handleMergeFP(cameraId: string) {
  Modal.confirm({
    title: '合并误报到基线',
    content: `将 ${cameraId} 的误报候选池帧合并到新基线版本（Draft 状态，需审核后才能训练）。`,
    okText: '执行合并',
    cancelText: '取消',
    async onOk() {
      mergingFP.value = cameraId
      try {
        const res = await mergeFalsePositives({ camera_id: cameraId })
        message.success(`误报合并完成: ${res.data.version}, 新增 ${res.data.fp_included} 张误报帧`)
        loadBaselines()
      } catch (e: any) {
        message.error(e.response?.data?.error || '合并失败')
      } finally {
        mergingFP.value = null
      }
    },
  })
}

async function handleCapture() {
  if (!captureForm.value.camera_id) {
    message.warning('请先选择摄像头')
    return
  }
  captureSubmitting.value = true
  try {
    const form = new FormData()
    form.append('camera_id', captureForm.value.camera_id)
    form.append('count', String(captureForm.value.count))
    form.append('interval', String(captureForm.value.interval))
    form.append('session_label', captureForm.value.session_label)
    await startCapture(form)
    message.success('采集任务已启动')
    captureModalVisible.value = false
    loadTasks()
  } catch (e: any) {
    message.error(e.response?.data?.error || '启动采集失败')
  } finally {
    captureSubmitting.value = false
  }
}

async function handleTrain() {
  if (!trainForm.value.camera_id) {
    message.warning('请先选择摄像头')
    return
  }
  trainSubmitting.value = true
  try {
    const form = new FormData()
    form.append('camera_id', trainForm.value.camera_id)
    form.append('model_type', trainForm.value.model_type)
    form.append('export_format', trainForm.value.export_format)
    await startTraining(form)
    message.success('训练任务已启动')
    loadTasks()
  } catch (e: any) {
    message.error(e.response?.data?.error || '启动训练失败')
  } finally {
    trainSubmitting.value = false
  }
}

async function handleDeploy(record: any) {
  deployingModel.value = record.model_path
  try {
    await deployModel({
      camera_id: record.camera_id,
      model_path: record.model_path,
    })
    message.success(`模型已部署到 ${record.camera_id}`)
  } catch (e: any) {
    message.error(e.response?.data?.error || '部署失败')
  } finally {
    deployingModel.value = null
  }
}

async function handleDismissTask(taskId: string) {
  try {
    await dismissTask(taskId)
    loadTasks()
  } catch (e) {
    console.error('Failed to dismiss task', e)
  }
}

// ── Advanced capture job actions ──

async function handleAdvCapture() {
  if (!advCaptureForm.value.camera_id) {
    message.warning('请先选择摄像头')
    return
  }
  advCaptureSubmitting.value = true
  try {
    const form = new FormData()
    form.append('camera_id', advCaptureForm.value.camera_id)
    form.append('target_frames', String(advCaptureForm.value.target_frames))
    form.append('duration_hours', String(advCaptureForm.value.duration_hours))
    form.append('sampling_strategy', advCaptureForm.value.sampling_strategy)
    form.append('diversity_threshold', String(advCaptureForm.value.diversity_threshold))
    form.append('frames_per_period', String(advCaptureForm.value.frames_per_period))
    await startCaptureJob(form)
    message.success('高级采集任务已启动')
    advCaptureVisible.value = false
    loadTasks()
  } catch (e: any) {
    message.error(e.response?.data?.error || '启动高级采集失败')
  } finally {
    advCaptureSubmitting.value = false
  }
}

async function handlePauseJob(taskId: string) {
  try {
    await pauseCaptureJob(taskId)
    message.success('任务已暂停')
    loadTasks()
  } catch (e: any) {
    message.error(e.response?.data?.error || '暂停失败')
  }
}

async function handleResumeJob(taskId: string) {
  try {
    await resumeCaptureJob(taskId)
    message.success('任务已恢复')
    loadTasks()
  } catch (e: any) {
    message.error(e.response?.data?.error || '恢复失败')
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
        loadTasks()
      } catch (e: any) {
        message.error(e.response?.data?.error || '中止失败')
      }
    },
  })
}

async function handleOptimize(record: any) {
  optimizingBaseline.value = `${record.camera_id}-${record.version}`
  try {
    const preview = await previewOptimize({
      camera_id: record.camera_id,
      zone_id: 'default',
      target_ratio: 0.2,
    })
    const { total, keep, move } = preview.data
    optimizingBaseline.value = null

    Modal.confirm({
      title: '确认优化',
      content: `共 ${total} 张图片，将保留 ${keep} 张，移除 ${move} 张。确认执行优化？`,
      okText: '确认优化',
      cancelText: '取消',
      async onOk() {
        optimizingBaseline.value = `${record.camera_id}-${record.version}`
        try {
          const res = await optimizeBaseline({
            camera_id: record.camera_id,
            zone_id: 'default',
            target_ratio: 0.2,
          })
          const data = res.data
          message.success(`优化完成: 保留 ${data.selected} 张, 移除 ${data.moved} 张`)
          loadBaselines()
        } catch (e: any) {
          message.error(e.response?.data?.error || '优化失败')
        } finally {
          optimizingBaseline.value = null
        }
      },
    })
  } catch (e: any) {
    message.error(e.response?.data?.error || '优化预览失败')
    optimizingBaseline.value = null
  }
}

async function handleActivate(record: any) {
  Modal.confirm({
    title: '确认激活',
    content: `确定要激活模型版本 ${record.model_version_id} 吗？这将停用该摄像头的其他模型。`,
    okText: '确认',
    cancelText: '取消',
    async onOk() {
      activatingModel.value = record.model_version_id
      try {
        await activateModel(record.model_version_id)
        message.success(`模型 ${record.model_version_id} 已激活`)
        loadRegistry()
      } catch (e: any) {
        message.error(e.response?.data?.error || '激活失败')
      } finally {
        activatingModel.value = null
      }
    },
  })
}

async function handleRollback(record: any) {
  Modal.confirm({
    title: '确认回滚',
    content: `确定要回滚摄像头 ${record.camera_id} 到上一个模型版本吗？`,
    okText: '确认回滚',
    cancelText: '取消',
    okType: 'danger',
    async onOk() {
      try {
        const res = await rollbackModel(record.model_version_id)
        message.success(`已回滚到 ${res.data.activated}`)
        loadRegistry()
      } catch (e: any) {
        message.error(e.response?.data?.error || '回滚失败')
      }
    },
  })
}

async function handleCompare() {
  if (compareForm.value.old_record_id == null || compareForm.value.new_record_id == null) {
    message.warning('请选择两个训练记录进行对比')
    return
  }
  if (compareForm.value.old_record_id === compareForm.value.new_record_id) {
    message.warning('请选择不同的训练记录')
    return
  }
  comparing.value = true
  compareResult.value = null
  try {
    const res = await compareModels({
      old_record_id: compareForm.value.old_record_id,
      new_record_id: compareForm.value.new_record_id,
    })
    compareResult.value = res.data
  } catch (e: any) {
    message.error(e.response?.data?.error || '对比失败')
  } finally {
    comparing.value = false
  }
}

function showHistoryDetail(record: any) {
  historyDetail.value = record
  historyDetailVisible.value = true
}

// ── Tab change handler ──

function onTabChange(key: string | number) {
  activeTab.value = String(key)
  if (key === 'baselines') { loadBaselines(); loadTasks() }
  if (key === 'training') loadTasks()
  if (key === 'models') { loadModels(); loadHistory(); loadRegistry() }
  if (key === 'release') { loadReleaseModels(); loadVersionEvents() }
}

// ── Computed: active tasks by type ──

const captureTasks = computed(() =>
  tasks.value.filter(t => t.task_type === 'baseline_capture' || t.task_type === 'baseline_capture_job')
)
const trainingTasks = computed(() =>
  tasks.value.filter(t => t.task_type === 'model_training')
)
const completedRecords = computed(() =>
  trainingHistory.value.filter(r => r.status === 'complete')
)

// ── Table columns ──

const baselineColumns = [
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '版本', dataIndex: 'version', key: 'version' },
  { title: '图片数量', dataIndex: 'image_count', key: 'image_count' },
  { title: '采集场景', dataIndex: 'session_label', key: 'session_label' },
  { title: '生命周期', key: 'state', width: 100 },
  { title: '操作', key: 'action', width: 360 },
]

const versionColumns = [
  { title: '版本', dataIndex: 'version', key: 'version', width: 80 },
  { title: '状态', key: 'state', width: 90 },
  { title: '图片', dataIndex: 'image_count', key: 'image_count', width: 70 },
  { title: '审核人', dataIndex: 'verified_by', key: 'verified_by', width: 100 },
  { title: '审核时间', dataIndex: 'verified_at', key: 'verified_at', width: 140 },
  { title: '操作', key: 'action', width: 180 },
]

const groupColumns = [
  { title: '组 ID', dataIndex: 'group_id', key: 'group_id' },
  { title: '名称', dataIndex: 'name', key: 'name' },
  { title: '成员摄像头', key: 'camera_ids' },
  { title: '图片数', dataIndex: 'image_count', key: 'image_count', width: 80 },
  { title: '当前版本', dataIndex: 'current_version', key: 'current_version', width: 100 },
  { title: '操作', key: 'action', width: 120 },
]

const modelColumns = [
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '格式', dataIndex: 'format', key: 'format' },
  { title: '大小', key: 'size_mb', customRender: ({ record }: any) => `${record.size_mb} MB` },
  { title: '训练时间', dataIndex: 'trained_at', key: 'trained_at' },
  { title: '路径', dataIndex: 'model_path', key: 'model_path', ellipsis: true },
  { title: '操作', key: 'action', width: 100 },
]

const registryColumns = [
  { title: '版本 ID', dataIndex: 'model_version_id', key: 'version_id', ellipsis: true },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '模型类型', dataIndex: 'model_type', key: 'model_type' },
  { title: '模型哈希', key: 'model_hash', width: 100 },
  { title: '创建时间', dataIndex: 'created_at', key: 'created_at', width: 160 },
  { title: '状态', key: 'is_active', width: 80 },
  { title: '操作', key: 'action', width: 160 },
]

const historyColumns = [
  { title: 'ID', dataIndex: 'id', key: 'id', width: 60 },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '模型', dataIndex: 'model_type', key: 'model_type' },
  { title: '基线数', dataIndex: 'baseline_count', key: 'baseline_count', width: 80 },
  { title: '训练/验证', key: 'split', width: 100 },
  { title: '质量', key: 'grade', width: 70 },
  { title: '推荐阈值', key: 'threshold', width: 100 },
  { title: '状态', key: 'status', width: 80 },
  { title: '耗时', key: 'duration', width: 80 },
  { title: '时间', dataIndex: 'trained_at', key: 'trained_at', width: 160 },
  { title: '操作', key: 'action', width: 80 },
]

// ── Lifecycle ──

// ── Tab 4: Release Pipeline ──
const releaseModels = ref<any[]>([])
const releaseLoading = ref(false)
const versionEvents = ref<any[]>([])
const eventsLoading = ref(false)
const promotingModel = ref<string | null>(null)
const promoteModalVisible = ref(false)
const promoteForm = ref({ version_id: '', target_stage: '', triggered_by: '', reason: '', canary_camera_id: '' })
const shadowReport = ref<any>(null)
const shadowReportVisible = ref(false)
const shadowReportLoading = ref(false)
const stageHistoryVisible = ref(false)
const stageHistory = ref<any[]>([])
const stageHistoryLoading = ref(false)

const STAGE_MAP: Record<string, { text: string; color: string }> = {
  candidate: { text: '候选', color: 'blue' },
  shadow: { text: '影子', color: 'purple' },
  canary: { text: '金丝雀', color: 'orange' },
  production: { text: '生产', color: 'green' },
  retired: { text: '退役', color: 'default' },
}

const VALID_TRANSITIONS: Record<string, string[]> = {
  candidate: ['shadow'],
  shadow: ['canary', 'candidate'],
  canary: ['production', 'shadow'],
  production: [],
}

const STAGE_LABELS: Record<string, string> = {
  shadow: '推进到影子模式',
  canary: '推进到金丝雀',
  production: '推进到生产',
  candidate: '回退到候选',
}

async function loadReleaseModels() {
  releaseLoading.value = true
  try {
    const res = await getModelRegistry()
    const models = res.data.models || []
    // Sort: production first, then canary, shadow, candidate, retired last
    const order: Record<string, number> = { production: 0, canary: 1, shadow: 2, candidate: 3, retired: 4 }
    releaseModels.value = models.sort((a: any, b: any) =>
      (order[a.stage] ?? 5) - (order[b.stage] ?? 5)
    )
  } catch (e) {
    console.error('Failed to load release models', e)
  } finally {
    releaseLoading.value = false
  }
}

async function loadVersionEvents() {
  eventsLoading.value = true
  try {
    const res = await getVersionEvents({ limit: 50 })
    versionEvents.value = res.data.events || []
  } catch (e) {
    console.error('Failed to load version events', e)
  } finally {
    eventsLoading.value = false
  }
}

function handlePromote(record: any) {
  const transitions = VALID_TRANSITIONS[record.stage] || []
  if (transitions.length === 0) {
    message.warning('该模型当前阶段不支持推进')
    return
  }
  promoteForm.value = {
    version_id: record.model_version_id,
    target_stage: transitions[0],
    triggered_by: '',
    reason: '',
    canary_camera_id: '',
  }
  promoteModalVisible.value = true
}

async function submitPromote() {
  if (!promoteForm.value.triggered_by) {
    message.warning('请输入操作人')
    return
  }
  if (promoteForm.value.target_stage === 'canary' && !promoteForm.value.canary_camera_id) {
    message.warning('金丝雀阶段需要选择目标摄像头')
    return
  }
  promotingModel.value = promoteForm.value.version_id
  try {
    await promoteModel(promoteForm.value.version_id, {
      target_stage: promoteForm.value.target_stage,
      triggered_by: promoteForm.value.triggered_by,
      reason: promoteForm.value.reason || undefined,
      canary_camera_id: promoteForm.value.canary_camera_id || undefined,
    })
    message.success(`模型已推进到 ${STAGE_MAP[promoteForm.value.target_stage]?.text || promoteForm.value.target_stage}`)
    promoteModalVisible.value = false
    loadReleaseModels()
    loadVersionEvents()
  } catch (e: any) {
    message.error(e.response?.data?.error || '推进失败')
  } finally {
    promotingModel.value = null
  }
}

function handleRetire(record: any) {
  Modal.confirm({
    title: '确认退役',
    content: `确定要将模型 ${record.model_version_id} 标记为退役吗？模型文件将保留。`,
    okText: '确认退役',
    cancelText: '取消',
    okType: 'danger',
    async onOk() {
      try {
        await retireModel(record.model_version_id, { triggered_by: 'operator' })
        message.success('模型已退役')
        loadReleaseModels()
        loadVersionEvents()
      } catch (e: any) {
        message.error(e.response?.data?.error || '退役失败')
      }
    },
  })
}

async function handleViewShadowReport(record: any) {
  shadowReportLoading.value = true
  shadowReportVisible.value = true
  shadowReport.value = null
  try {
    const res = await getShadowReport(record.model_version_id, { days: 7 })
    shadowReport.value = res.data
  } catch (e: any) {
    message.error(e.response?.data?.error || '获取影子报告失败')
  } finally {
    shadowReportLoading.value = false
  }
}

async function handleViewStageHistory(record: any) {
  stageHistoryLoading.value = true
  stageHistoryVisible.value = true
  stageHistory.value = []
  try {
    const res = await getStageHistory(record.model_version_id)
    stageHistory.value = res.data.events || []
  } catch (e: any) {
    message.error(e.response?.data?.error || '获取历史失败')
  } finally {
    stageHistoryLoading.value = false
  }
}

const releaseColumns = [
  { title: '版本 ID', dataIndex: 'model_version_id', key: 'version_id', ellipsis: true },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '模型类型', dataIndex: 'model_type', key: 'model_type' },
  { title: '阶段', key: 'stage', width: 100 },
  { title: '组件', key: 'component_type', width: 80 },
  { title: '创建时间', dataIndex: 'created_at', key: 'created_at', width: 160 },
  { title: '操作', key: 'action', width: 300 },
]

const eventColumns = [
  { title: '时间', dataIndex: 'timestamp', key: 'timestamp', width: 160 },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '转换', key: 'transition', width: 200 },
  { title: '操作人', dataIndex: 'triggered_by', key: 'triggered_by' },
  { title: '原因', dataIndex: 'reason', key: 'reason', ellipsis: true },
]

// ── Batch Inference ──
const batchCameraId = ref('')
const batchImagePaths = ref('')
const batchRunning = ref(false)
const batchResults = ref<any[]>([])

async function handleBatchInference() {
  const paths = batchImagePaths.value.split('\n').map(p => p.trim()).filter(Boolean)
  if (!batchCameraId.value) {
    message.error('请选择摄像头')
    return
  }
  if (paths.length === 0) {
    message.error('请输入图片路径')
    return
  }
  batchRunning.value = true
  batchResults.value = []
  try {
    const res = await batchInference(batchCameraId.value, paths)
    batchResults.value = res.data.results || []
    message.success(`完成推理: ${res.data.scored}/${res.data.total} 张图片`)
  } catch (e: any) {
    message.error(e.response?.data?.error || '批量推理失败')
  } finally {
    batchRunning.value = false
  }
}

onMounted(async () => {
  await loadCameras()
  loadBaselines()
  loadTasks()
  loadCameraGroups()
})
</script>

<template>
  <div>
    <Typography.Title :level="3" style="margin-bottom: 24px">模型管理</Typography.Title>

    <Tabs :activeKey="activeTab" @change="onTabChange">
      <!-- ═══════════ Tab 1: Baselines ═══════════ -->
      <Tabs.TabPane key="baselines" tab="基线管理">
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
                <PauseCircleOutlined v-if="task.status === 'paused'" style="color: #faad14" />
                <LoadingOutlined v-else-if="task.status === 'running'" spin style="color: #3b82f6" />
                <CheckCircleOutlined v-else-if="task.status === 'complete'" style="color: #52c41a" />
                <CloseCircleOutlined v-else-if="task.status === 'failed'" style="color: #ff4d4f" />
                <StopOutlined v-else-if="task.status === 'aborted'" style="color: #8890a0" />
                <span style="font-weight: 500">
                  {{ taskTitle(task) }}
                </span>
                <span v-if="task.camera_id" style="color: #8890a0">{{ task.camera_id }}</span>
                <Tag :color="(JOB_STATUS_MAP[task.status] || {}).color || 'default'">
                  {{ (JOB_STATUS_MAP[task.status] || {}).text || task.status }}
                </Tag>
              </Space>
              <Space>
                <Button
                  v-if="canPauseTask(task)"
                  size="small"
                  @click="handlePauseJob(task.task_id)"
                >
                  <template #icon><PauseCircleOutlined /></template>
                  暂停
                </Button>
                <Button
                  v-if="canResumeTask(task)"
                  size="small"
                  type="primary"
                  @click="handleResumeJob(task.task_id)"
                >
                  <template #icon><CaretRightOutlined /></template>
                  恢复
                </Button>
                <Button
                  v-if="canAbortTask(task)"
                  size="small"
                  danger
                  @click="handleAbortJob(task.task_id)"
                >
                  <template #icon><StopOutlined /></template>
                  中止
                </Button>
                <Button
                  v-if="canDismissTask(task)"
                  size="small"
                  @click="handleDismissTask(task.task_id)"
                >关闭</Button>
              </Space>
            </div>
            <Progress
              :percent="task.progress"
              :status="taskProgressStatus(task)"
              size="small"
            />
            <div style="font-size: 12px; color: #8890a0; margin-top: 4px">{{ task.message }}</div>
            <div v-if="task.error" style="font-size: 12px; color: #ff4d4f; margin-top: 4px">{{ task.error }}</div>
          </Card>
        </div>

        <Card>
          <template #title>
            <div style="display: flex; justify-content: space-between; align-items: center">
              <span>基线数据</span>
              <Space>
                <Button @click="loadBaselines">
                  <template #icon><ReloadOutlined /></template>
                  刷新
                </Button>
                <Button type="primary" @click="captureModalVisible = true">
                  <template #icon><PlayCircleOutlined /></template>
                  快速采集
                </Button>
                <Button @click="advCaptureVisible = true">
                  <template #icon><ExperimentOutlined /></template>
                  高级采集
                </Button>
              </Space>
            </div>
          </template>
          <p style="color: #8890a0; margin-bottom: 16px">
            从在线摄像头采集"正常"场景的参考图片，用于训练异常检测模型。
          </p>
          <Table
            :columns="baselineColumns"
            :data-source="baselines"
            :loading="baselinesLoading"
            :pagination="false"
            row-key="(record: any) => `${record.camera_id}-${record.version}`"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'image_count'">
                <span>{{ record.image_count }} 张</span>
              </template>
              <template v-if="column.key === 'session_label'">
                <Tag v-if="record.session_label">{{ record.session_label }}</Tag>
                <span v-else style="color: #666">-</span>
              </template>
              <template v-if="column.key === 'state'">
                <Tag
                  v-if="record.state"
                  :color="(BASELINE_STATE_MAP[record.state] || {}).color || 'default'"
                >
                  {{ (BASELINE_STATE_MAP[record.state] || {}).text || record.state }}
                </Tag>
                <Tag v-else color="green">就绪</Tag>
              </template>
              <template v-if="column.key === 'action'">
                <Space>
                  <Tooltip title="多样性优选">
                    <Button
                      size="small"
                      :loading="optimizingBaseline === `${record.camera_id}-${record.version}`"
                      :disabled="record.image_count < 30"
                      @click="handleOptimize(record)"
                    >
                      优化
                    </Button>
                  </Tooltip>
                  <Button size="small" @click="openVersionDrawer(record.camera_id)">
                    <SafetyCertificateOutlined />
                    版本管理
                  </Button>
                  <Tooltip title="将误报候选帧合并到新基线版本">
                    <Button
                      size="small"
                      :loading="mergingFP === record.camera_id"
                      @click="handleMergeFP(record.camera_id)"
                    >
                      <MergeCellsOutlined />
                      合并误报
                    </Button>
                  </Tooltip>
                  <Button
                    size="small"
                    danger
                    :loading="deletingBaseline === `${record.camera_id}-${record.version}`"
                    :disabled="record.state === 'active'"
                    @click="handleDeleteBaseline(record)"
                  >
                    <DeleteOutlined />
                    删除
                  </Button>
                </Space>
              </template>
            </template>
          </Table>
          <div v-if="baselines.length === 0 && !baselinesLoading" style="text-align: center; padding: 32px; color: #666">
            暂无基线数据，请先采集基线图片
          </div>
        </Card>

        <!-- Camera Groups Panel -->
        <Card v-if="cameraGroups.length > 0" size="small" style="margin-top: 16px">
          <template #title>
            <Space>
              <TeamOutlined />
              <span>摄像头组共享基线</span>
            </Space>
          </template>
          <Table
            :columns="groupColumns"
            :data-source="cameraGroups"
            :loading="groupsLoading"
            :pagination="false"
            row-key="group_id"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'camera_ids'">
                <Tag v-for="cam in record.camera_ids" :key="cam" style="margin: 2px">{{ cam }}</Tag>
              </template>
              <template v-if="column.key === 'action'">
                <Button
                  size="small"
                  type="primary"
                  :loading="mergingGroup === record.group_id"
                  @click="handleMergeGroup(record.group_id)"
                >
                  <MergeCellsOutlined />
                  合并基线
                </Button>
              </template>
            </template>
          </Table>
        </Card>

        <!-- Version Management Drawer -->
        <Drawer
          v-model:open="versionDrawerVisible"
          :title="`基线版本管理 — ${versionDrawerCamera}`"
          width="680"
          placement="right"
        >
          <Table
            :columns="versionColumns"
            :data-source="baselineVersions"
            :loading="versionsLoading"
            :pagination="false"
            row-key="version"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'state'">
                <Tag :color="(BASELINE_STATE_MAP[record.state] || {}).color || 'default'">
                  {{ (BASELINE_STATE_MAP[record.state] || {}).text || record.state }}
                </Tag>
              </template>
              <template v-if="column.key === 'verified_at'">
                <span v-if="record.verified_at">{{ record.verified_at.replace('T', ' ').slice(0, 19) }}</span>
                <span v-else style="color: #999">-</span>
              </template>
              <template v-if="column.key === 'action'">
                <Button
                  v-if="record.state === 'draft'"
                  size="small"
                  type="primary"
                  :loading="verifyingVersion === record.version"
                  @click="handleVerify(record)"
                >
                  审核
                </Button>
                <Button
                  v-if="record.state === 'verified'"
                  size="small"
                  type="primary"
                  :loading="activatingVersion === record.version"
                  @click="handleActivateBaseline(record)"
                >
                  激活
                </Button>
                <Button
                  v-if="record.state === 'active'"
                  size="small"
                  danger
                  @click="handleRetireBaseline(record)"
                >
                  退役
                </Button>
                <Button
                  v-if="record.state !== 'active'"
                  size="small"
                  danger
                  :loading="deletingBaseline === `${record.camera_id}-${record.version}`"
                  @click="handleDeleteBaseline(record)"
                >
                  删除
                </Button>
                <span v-if="record.state === 'retired'" style="color: #999">已退役</span>
              </template>
            </template>
          </Table>
          <Divider />
          <div style="color: #8890a0; font-size: 12px">
            <p><strong>状态流转:</strong> 草稿 → 已审核 → 生产中 → 已退役（严格单向，不可逆转）</p>
            <p><strong>训练要求:</strong> 仅「已审核」或「生产中」的基线可用于模型训练</p>
          </div>
        </Drawer>

        <!-- Capture modal -->
        <Modal
          v-model:open="captureModalVisible"
          title="采集新基线"
          @ok="handleCapture"
          :confirmLoading="captureSubmitting"
          okText="开始采集"
          cancelText="取消"
        >
          <Form layout="vertical" style="margin-top: 16px">
            <Form.Item label="选择摄像头">
              <Select v-model:value="captureForm.camera_id" style="width: 100%">
                <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
                  {{ cam.camera_id }} — {{ cam.name }}
                </Select.Option>
              </Select>
            </Form.Item>
            <Form.Item label="采集场景">
              <Select v-model:value="captureForm.session_label" style="width: 100%">
                <Select.Option v-for="sl in SESSION_LABELS" :key="sl.value" :value="sl.value">
                  {{ sl.label }}
                </Select.Option>
              </Select>
            </Form.Item>
            <Space>
              <Form.Item label="采集帧数">
                <InputNumber v-model:value="captureForm.count" :min="10" :max="1000" />
              </Form.Item>
              <Form.Item label="间隔（秒）">
                <InputNumber v-model:value="captureForm.interval" :min="0.5" :max="60" :step="0.5" />
              </Form.Item>
            </Space>
          </Form>
        </Modal>

        <!-- Advanced capture modal -->
        <Modal
          v-model:open="advCaptureVisible"
          title="高级基线采集"
          @ok="handleAdvCapture"
          :confirmLoading="advCaptureSubmitting"
          okText="开始采集"
          cancelText="取消"
          width="560px"
        >
          <Form layout="vertical" style="margin-top: 16px">
            <Form.Item label="选择摄像头">
              <Select v-model:value="advCaptureForm.camera_id" style="width: 100%">
                <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
                  {{ cam.camera_id }} — {{ cam.name }}
                </Select.Option>
              </Select>
            </Form.Item>

            <Form.Item label="采样策略">
              <Radio.Group v-model:value="advCaptureForm.sampling_strategy" style="width: 100%">
                <div v-for="s in SAMPLING_STRATEGIES" :key="s.value" style="margin-bottom: 8px">
                  <Radio :value="s.value">
                    <span style="font-weight: 500">{{ s.label }}</span>
                    <div style="font-size: 12px; color: #8890a0; margin-left: 24px">{{ s.desc }}</div>
                  </Radio>
                </div>
              </Radio.Group>
            </Form.Item>

            <Space :size="16">
              <Form.Item label="目标帧数">
                <InputNumber
                  v-model:value="advCaptureForm.target_frames"
                  :min="100" :max="10000" :step="100"
                  style="width: 140px"
                />
              </Form.Item>
              <Form.Item label="持续时长（小时）">
                <InputNumber
                  v-model:value="advCaptureForm.duration_hours"
                  :min="1" :max="168" :step="1"
                  style="width: 140px"
                />
              </Form.Item>
            </Space>

            <Collapse ghost>
              <Collapse.Panel key="advanced" header="高级参数">
                <Form.Item
                  label="多样性阈值"
                  v-if="advCaptureForm.sampling_strategy !== 'uniform'"
                >
                  <Slider
                    v-model:value="advCaptureForm.diversity_threshold"
                    :min="0.1" :max="0.9" :step="0.05"
                    :marks="{ 0.1: '低', 0.3: '默认', 0.9: '高' }"
                  />
                  <div style="font-size: 12px; color: #8890a0">
                    值越高，保留的帧越多样；值越低，接受更多相似帧
                  </div>
                </Form.Item>
                <Form.Item
                  label="每时段帧数"
                  v-if="advCaptureForm.sampling_strategy === 'scheduled'"
                >
                  <InputNumber
                    v-model:value="advCaptureForm.frames_per_period"
                    :min="5" :max="200" :step="5"
                    style="width: 140px"
                  />
                  <div style="font-size: 12px; color: #8890a0; margin-top: 4px">
                    每个时段（清晨/正午/傍晚/深夜）采集的目标帧数
                  </div>
                </Form.Item>
              </Collapse.Panel>
            </Collapse>
          </Form>
        </Modal>
      </Tabs.TabPane>

      <!-- ═══════════ Tab 2: Training ═══════════ -->
      <Tabs.TabPane key="training" tab="模型训练">
        <Card title="训练异常检测模型" style="margin-bottom: 16px">
          <p style="color: #8890a0; margin-bottom: 16px">
            使用基线图片训练异常检测模型。训练耗时通常 5-15 分钟。
          </p>
          <Form layout="vertical" style="max-width: 600px">
            <Form.Item label="选择摄像头">
              <Select v-model:value="trainForm.camera_id" style="width: 100%">
                <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
                  {{ cam.camera_id }} — {{ cam.name }}
                </Select.Option>
              </Select>
            </Form.Item>
            <Form.Item label="模型类型">
              <Select v-model:value="trainForm.model_type" style="width: 100%">
                <Select.Option v-for="mt in MODEL_TYPES" :key="mt.value" :value="mt.value">
                  {{ mt.label }}
                </Select.Option>
              </Select>
            </Form.Item>
            <Form.Item label="导出格式">
              <Select v-model:value="trainForm.export_format" style="width: 100%">
                <Select.Option v-for="ef in EXPORT_FORMATS" :key="ef.value" :value="ef.value">
                  {{ ef.label }}
                </Select.Option>
              </Select>
            </Form.Item>
            <Form.Item>
              <Button type="primary" :loading="trainSubmitting" @click="handleTrain">
                <template #icon><ExperimentOutlined /></template>
                开始训练
              </Button>
            </Form.Item>
          </Form>
        </Card>

        <!-- Active training tasks -->
        <Card title="训练任务" v-if="trainingTasks.length > 0 || tasks.some(t => t.task_type === 'model_training')">
          <div v-for="task in trainingTasks" :key="task.task_id" style="margin-bottom: 16px">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px">
              <Space>
                <LoadingOutlined v-if="task.status === 'running'" spin style="color: #3b82f6" />
                <CheckCircleOutlined v-else-if="task.status === 'complete'" style="color: #52c41a" />
                <CloseCircleOutlined v-else-if="task.status === 'failed'" style="color: #ff4d4f" />
                <span style="font-weight: 500">模型训练</span>
                <span v-if="task.camera_id" style="color: #8890a0">{{ task.camera_id }}</span>
                <Tag :color="task.status === 'complete' ? 'green' : task.status === 'failed' ? 'red' : 'blue'">
                  {{ task.status === 'running' ? '运行中' : task.status === 'complete' ? '已完成' : task.status === 'failed' ? '失败' : '等待中' }}
                </Tag>
              </Space>
              <Button
                v-if="task.status === 'complete' || task.status === 'failed'"
                size="small"
                @click="handleDismissTask(task.task_id)"
              >关闭</Button>
            </div>
            <Progress
              :percent="task.progress"
              :status="task.status === 'failed' ? 'exception' : task.status === 'complete' ? 'success' : 'active'"
              size="small"
            />
            <div style="font-size: 12px; color: #8890a0; margin-top: 4px">{{ task.message }}</div>
            <div v-if="task.error" style="font-size: 12px; color: #ff4d4f; margin-top: 4px">{{ task.error }}</div>

            <!-- Training result details -->
            <div
              v-if="task.status === 'complete' && task.result"
              style="margin-top: 12px; padding: 12px; background: rgba(59, 130, 246, 0.05); border-radius: 6px; border: 1px solid rgba(59, 130, 246, 0.15)"
            >
              <Space>
                <Tag v-if="task.result.grade" :color="GRADE_COLORS[task.result.grade] || 'default'" style="font-size: 16px; padding: 4px 12px">
                  {{ task.result.grade }}
                </Tag>
                <span v-if="task.result.threshold" style="color: #8890a0; font-size: 13px">
                  推荐阈值: {{ typeof task.result.threshold === 'number' ? task.result.threshold.toFixed(3) : task.result.threshold }}
                </span>
              </Space>
            </div>
          </div>
          <div v-if="trainingTasks.length === 0" style="text-align: center; padding: 16px; color: #666">
            暂无训练任务
          </div>
        </Card>
      </Tabs.TabPane>

      <!-- ═══════════ Tab 3: Models ═══════════ -->
      <Tabs.TabPane key="models" tab="已训练模型">
        <!-- Trained models table -->
        <Card style="margin-bottom: 16px">
          <template #title>
            <div style="display: flex; justify-content: space-between; align-items: center">
              <span>已训练模型</span>
              <Button @click="loadModels">
                <template #icon><ReloadOutlined /></template>
                刷新
              </Button>
            </div>
          </template>
          <Table
            :columns="modelColumns"
            :data-source="models"
            :loading="modelsLoading"
            :pagination="false"
            row-key="model_path"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'format'">
                <Tag :color="record.format === 'openvino' ? 'blue' : record.format === 'onnx' ? 'green' : 'default'">
                  {{ record.format.toUpperCase() }}
                </Tag>
              </template>
              <template v-if="column.key === 'action'">
                <Tooltip title="部署到摄像头">
                  <Button
                    type="primary"
                    size="small"
                    :loading="deployingModel === record.model_path"
                    @click="handleDeploy(record)"
                  >
                    <template #icon><RocketOutlined /></template>
                    部署
                  </Button>
                </Tooltip>
              </template>
            </template>
          </Table>
          <div v-if="models.length === 0 && !modelsLoading" style="text-align: center; padding: 32px; color: #666">
            暂无已训练的模型
          </div>
        </Card>

        <!-- Model Registry -->
        <Card style="margin-bottom: 16px">
          <template #title>
            <div style="display: flex; justify-content: space-between; align-items: center">
              <span>模型版本注册</span>
              <Button @click="loadRegistry">
                <template #icon><ReloadOutlined /></template>
                刷新
              </Button>
            </div>
          </template>
          <p style="color: #8890a0; margin-bottom: 16px">
            管理已注册的模型版本，支持激活和回滚操作。
          </p>
          <Table
            :columns="registryColumns"
            :data-source="registry"
            :loading="registryLoading"
            :pagination="{ pageSize: 10, showSizeChanger: false }"
            row-key="model_version_id"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'version_id'">
                <span style="font-family: monospace; font-size: 12px">{{ record.model_version_id }}</span>
              </template>
              <template v-if="column.key === 'model_hash'">
                <span style="font-family: monospace; font-size: 12px">{{ record.model_hash?.substring(0, 8) }}</span>
              </template>
              <template v-if="column.key === 'created_at'">
                {{ record.created_at ? record.created_at.replace('T', ' ').substring(0, 16) : '-' }}
              </template>
              <template v-if="column.key === 'is_active'">
                <Tag :color="record.is_active ? 'green' : 'default'">
                  {{ record.is_active ? '已激活' : '未激活' }}
                </Tag>
              </template>
              <template v-if="column.key === 'action'">
                <Space>
                  <Tooltip title="激活此版本">
                    <Button
                      size="small"
                      type="primary"
                      :disabled="record.is_active"
                      :loading="activatingModel === record.model_version_id"
                      @click="handleActivate(record)"
                    >
                      <template #icon><CheckOutlined /></template>
                      激活
                    </Button>
                  </Tooltip>
                  <Tooltip title="回滚到上一版本">
                    <Button
                      size="small"
                      danger
                      :disabled="!record.is_active"
                      @click="handleRollback(record)"
                    >
                      <template #icon><RollbackOutlined /></template>
                      回滚
                    </Button>
                  </Tooltip>
                </Space>
              </template>
            </template>
          </Table>
          <div v-if="registry.length === 0 && !registryLoading" style="text-align: center; padding: 32px; color: #666">
            暂无注册的模型版本
          </div>
        </Card>

        <!-- Training history table -->
        <Card>
          <template #title>
            <div style="display: flex; justify-content: space-between; align-items: center">
              <Space>
                <HistoryOutlined />
                <span>训练历史</span>
              </Space>
              <Space>
                <Button @click="compareVisible = true" :disabled="completedRecords.length < 2">
                  <template #icon><SwapOutlined /></template>
                  A/B 对比
                </Button>
                <Button @click="loadHistory">
                  <template #icon><ReloadOutlined /></template>
                  刷新
                </Button>
              </Space>
            </div>
          </template>
          <Table
            :columns="historyColumns"
            :data-source="trainingHistory"
            :loading="historyLoading"
            :pagination="{ pageSize: 10, showSizeChanger: false }"
            row-key="id"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'split'">
                {{ record.train_count }}/{{ record.val_count }}
              </template>
              <template v-if="column.key === 'grade'">
                <Tag
                  v-if="record.quality_grade"
                  :color="GRADE_COLORS[record.quality_grade] || 'default'"
                  style="font-weight: bold; font-size: 14px"
                >
                  {{ record.quality_grade }}
                </Tag>
                <span v-else style="color: #666">-</span>
              </template>
              <template v-if="column.key === 'threshold'">
                {{ record.threshold_recommended != null ? record.threshold_recommended.toFixed(3) : '-' }}
              </template>
              <template v-if="column.key === 'status'">
                <Tag :color="record.status === 'complete' ? 'green' : 'red'">
                  {{ record.status === 'complete' ? '完成' : '失败' }}
                </Tag>
              </template>
              <template v-if="column.key === 'duration'">
                {{ record.duration_seconds ? `${record.duration_seconds.toFixed(0)}s` : '-' }}
              </template>
              <template v-if="column.key === 'trained_at'">
                {{ record.trained_at ? record.trained_at.replace('T', ' ').substring(0, 16) : '-' }}
              </template>
              <template v-if="column.key === 'action'">
                <Button size="small" type="link" @click="showHistoryDetail(record)">详情</Button>
              </template>
            </template>
          </Table>
          <div v-if="trainingHistory.length === 0 && !historyLoading" style="text-align: center; padding: 32px; color: #666">
            暂无训练记录
          </div>
        </Card>

        <!-- History detail modal -->
        <Modal
          v-model:open="historyDetailVisible"
          title="训练详情"
          :footer="null"
          width="720px"
        >
          <Descriptions v-if="historyDetail" bordered :column="2" size="small" style="margin-top: 16px">
            <Descriptions.Item label="摄像头">{{ historyDetail.camera_id }}</Descriptions.Item>
            <Descriptions.Item label="区域">{{ historyDetail.zone_id }}</Descriptions.Item>
            <Descriptions.Item label="模型类型">{{ historyDetail.model_type }}</Descriptions.Item>
            <Descriptions.Item label="导出格式">{{ historyDetail.export_format || '-' }}</Descriptions.Item>
            <Descriptions.Item label="基线版本">{{ historyDetail.baseline_version }}</Descriptions.Item>
            <Descriptions.Item label="基线数量">{{ historyDetail.baseline_count }}</Descriptions.Item>
            <Descriptions.Item label="训练/验证">{{ historyDetail.train_count }}/{{ historyDetail.val_count }}</Descriptions.Item>
            <Descriptions.Item label="质量评级">
              <Tag v-if="historyDetail.quality_grade" :color="GRADE_COLORS[historyDetail.quality_grade]" style="font-weight:bold">
                {{ historyDetail.quality_grade }}
              </Tag>
              <span v-else>-</span>
            </Descriptions.Item>
            <Descriptions.Item label="推荐阈值">
              {{ historyDetail.threshold_recommended != null ? historyDetail.threshold_recommended.toFixed(4) : '-' }}
            </Descriptions.Item>
            <Descriptions.Item label="状态">
              <Tag :color="historyDetail.status === 'complete' ? 'green' : 'red'">
                {{ historyDetail.status === 'complete' ? '完成' : '失败' }}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="耗时">{{ historyDetail.duration_seconds?.toFixed(1) }}s</Descriptions.Item>
            <Descriptions.Item label="训练时间">{{ historyDetail.trained_at?.replace('T', ' ').substring(0, 19) }}</Descriptions.Item>
            <Descriptions.Item label="验证分数均值">{{ historyDetail.val_score_mean?.toFixed(4) ?? '-' }}</Descriptions.Item>
            <Descriptions.Item label="验证分数标准差">{{ historyDetail.val_score_std?.toFixed(4) ?? '-' }}</Descriptions.Item>
            <Descriptions.Item label="验证分数最大值">{{ historyDetail.val_score_max?.toFixed(4) ?? '-' }}</Descriptions.Item>
            <Descriptions.Item label="验证分数 P95">{{ historyDetail.val_score_p95?.toFixed(4) ?? '-' }}</Descriptions.Item>
            <Descriptions.Item label="预验证通过">
              <CheckCircleOutlined v-if="historyDetail.pre_validation_passed" style="color: #52c41a" />
              <CloseCircleOutlined v-else style="color: #ff4d4f" />
            </Descriptions.Item>
            <Descriptions.Item label="推理延迟">{{ historyDetail.inference_latency_ms != null ? `${historyDetail.inference_latency_ms.toFixed(1)}ms` : '-' }}</Descriptions.Item>
            <Descriptions.Item label="损坏率">
              {{ historyDetail.corruption_rate != null ? `${(historyDetail.corruption_rate * 100).toFixed(1)}%` : '-' }}
            </Descriptions.Item>
            <Descriptions.Item label="近重复率">
              {{ historyDetail.near_duplicate_rate != null ? `${(historyDetail.near_duplicate_rate * 100).toFixed(1)}%` : '-' }}
            </Descriptions.Item>
            <Descriptions.Item label="亮度标准差">
              {{ historyDetail.brightness_std != null ? historyDetail.brightness_std.toFixed(2) : '-' }}
            </Descriptions.Item>
            <Descriptions.Item label="检查点有效">
              <CheckCircleOutlined v-if="historyDetail.checkpoint_valid" style="color: #52c41a" />
              <CloseCircleOutlined v-else-if="historyDetail.checkpoint_valid === false" style="color: #ff4d4f" />
              <span v-else>-</span>
            </Descriptions.Item>
            <Descriptions.Item label="导出有效">
              <CheckCircleOutlined v-if="historyDetail.export_valid" style="color: #52c41a" />
              <CloseCircleOutlined v-else-if="historyDetail.export_valid === false" style="color: #ff4d4f" />
              <span v-else>-</span>
            </Descriptions.Item>
            <Descriptions.Item label="冒烟测试">
              <CheckCircleOutlined v-if="historyDetail.smoke_test_passed" style="color: #52c41a" />
              <CloseCircleOutlined v-else-if="historyDetail.smoke_test_passed === false" style="color: #ff4d4f" />
              <span v-else>-</span>
            </Descriptions.Item>
            <Descriptions.Item v-if="historyDetail.export_path" label="导出路径" :span="2">
              <span style="font-family: monospace; font-size: 12px; word-break: break-all">{{ historyDetail.export_path }}</span>
            </Descriptions.Item>
            <Descriptions.Item v-if="historyDetail.model_path" label="模型路径" :span="2">
              <span style="font-family: monospace; font-size: 12px; word-break: break-all">{{ historyDetail.model_path }}</span>
            </Descriptions.Item>
            <Descriptions.Item v-if="historyDetail.error" label="错误信息" :span="2">
              <span style="color: #ff4d4f">{{ historyDetail.error }}</span>
            </Descriptions.Item>
          </Descriptions>
        </Modal>

        <!-- Compare Modal -->
        <Modal
          v-model:open="compareVisible"
          title="A/B 模型对比"
          :footer="null"
          width="720px"
          @cancel="compareResult = null"
        >
          <Form layout="vertical" style="margin-top: 16px">
            <div style="display: flex; gap: 16px">
              <Form.Item label="基准模型 (旧)" style="flex: 1">
                <Select
                  v-model:value="compareForm.old_record_id"
                  placeholder="选择训练记录"
                  style="width: 100%"
                  :options="completedRecords.map(r => ({ value: r.id, label: `#${r.id} ${r.camera_id} - ${r.model_type} (${r.quality_grade || 'N/A'})` }))"
                />
              </Form.Item>
              <Form.Item label="候选模型 (新)" style="flex: 1">
                <Select
                  v-model:value="compareForm.new_record_id"
                  placeholder="选择训练记录"
                  style="width: 100%"
                  :options="completedRecords.map(r => ({ value: r.id, label: `#${r.id} ${r.camera_id} - ${r.model_type} (${r.quality_grade || 'N/A'})` }))"
                />
              </Form.Item>
            </div>
            <Form.Item>
              <Button type="primary" :loading="comparing" @click="handleCompare">
                <template #icon><SwapOutlined /></template>
                开始对比
              </Button>
            </Form.Item>
          </Form>

          <div v-if="compareResult" style="margin-top: 16px">
            <div style="margin-bottom: 16px; text-align: center">
              <Tag
                :color="RECOMMENDATION_COLORS[compareResult.recommendation] || 'default'"
                style="font-size: 16px; padding: 6px 16px"
              >
                {{ RECOMMENDATION_TEXT[compareResult.recommendation] || compareResult.recommendation }}
              </Tag>
            </div>
            <Table
              :data-source="[
                { metric: '平均分数', old_val: compareResult.old?.mean?.toFixed(4), new_val: compareResult.new?.mean?.toFixed(4) },
                { metric: '标准差', old_val: compareResult.old?.std?.toFixed(4), new_val: compareResult.new?.std?.toFixed(4) },
                { metric: '最大值', old_val: compareResult.old?.max?.toFixed(4), new_val: compareResult.new?.max?.toFixed(4) },
                { metric: 'P95', old_val: compareResult.old?.p95?.toFixed(4), new_val: compareResult.new?.p95?.toFixed(4) },
                { metric: '推理延迟', old_val: compareResult.old?.latency ? `${compareResult.old.latency.toFixed(1)}ms` : '-', new_val: compareResult.new?.latency ? `${compareResult.new.latency.toFixed(1)}ms` : '-' },
              ]"
              :columns="[
                { title: '指标', dataIndex: 'metric', key: 'metric' },
                { title: '基准模型 (旧)', dataIndex: 'old_val', key: 'old_val' },
                { title: '候选模型 (新)', dataIndex: 'new_val', key: 'new_val' },
              ]"
              :pagination="false"
              size="small"
              row-key="metric"
            />
          </div>
        </Modal>

        <!-- Batch Inference -->
        <Card style="margin-top: 16px">
          <template #title>批量推理</template>
          <p style="color: #999; margin-bottom: 16px">输入图片路径（每行一个），使用摄像头的活跃模型进行异常检测评分。</p>
          <Form layout="vertical">
            <Form.Item label="摄像头">
              <Select
                v-model:value="batchCameraId"
                placeholder="选择摄像头"
                style="width: 300px"
                :options="cameras.map(c => ({ value: c.camera_id, label: `${c.camera_id} - ${c.name || ''}` }))"
              />
            </Form.Item>
            <Form.Item label="图片路径（每行一个，最多100张）">
              <textarea
                v-model="batchImagePaths"
                rows="5"
                style="width: 100%; font-family: monospace; padding: 8px; border: 1px solid #d9d9d9; border-radius: 6px"
                placeholder="/path/to/image1.jpg&#10;/path/to/image2.jpg"
              />
            </Form.Item>
            <Form.Item>
              <Button type="primary" :loading="batchRunning" @click="handleBatchInference">
                <template #icon><ExperimentOutlined /></template>
                开始推理
              </Button>
            </Form.Item>
          </Form>
          <Table
            v-if="batchResults.length > 0"
            :data-source="batchResults"
            :columns="[
              { title: '文件路径', dataIndex: 'path', key: 'path', ellipsis: true },
              { title: '分数', dataIndex: 'score', key: 'score', width: 100 },
              { title: '异常', dataIndex: 'is_anomalous', key: 'is_anomalous', width: 80 },
              { title: '错误', dataIndex: 'error', key: 'error', width: 200 },
            ]"
            :pagination="{ pageSize: 20 }"
            size="small"
            row-key="path"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'is_anomalous'">
                <Tag v-if="record.is_anomalous === true" color="red">异常</Tag>
                <Tag v-else-if="record.is_anomalous === false" color="green">正常</Tag>
                <span v-else>-</span>
              </template>
              <template v-if="column.key === 'score'">
                <span v-if="record.score !== undefined">{{ record.score.toFixed(4) }}</span>
                <span v-else>-</span>
              </template>
              <template v-if="column.key === 'error'">
                <Tag v-if="record.error" color="red">{{ record.error }}</Tag>
              </template>
            </template>
          </Table>
        </Card>
      </Tabs.TabPane>

      <!-- ═══════════ Tab 4: Release Pipeline ═══════════ -->
      <Tabs.TabPane key="release" tab="发布管理">
        <!-- Model stage management -->
        <Card style="margin-bottom: 16px">
          <template #title>
            <div style="display: flex; justify-content: space-between; align-items: center">
              <Space>
                <SafetyCertificateOutlined />
                <span>模型阶段管理</span>
              </Space>
              <Button @click="loadReleaseModels">
                <template #icon><ReloadOutlined /></template>
                刷新
              </Button>
            </div>
          </template>
          <p style="color: #8890a0; margin-bottom: 16px">
            四阶段发布流程：候选 → 影子 → 金丝雀 → 生产。每次推进需工程师显式操作。
          </p>
          <Table
            :columns="releaseColumns"
            :data-source="releaseModels"
            :loading="releaseLoading"
            :pagination="{ pageSize: 15, showSizeChanger: false }"
            row-key="model_version_id"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'version_id'">
                <span style="font-family: monospace; font-size: 12px">{{ record.model_version_id }}</span>
              </template>
              <template v-if="column.key === 'stage'">
                <Tag :color="(STAGE_MAP[record.stage] || { color: 'default' }).color">
                  {{ (STAGE_MAP[record.stage] || { text: record.stage }).text }}
                </Tag>
              </template>
              <template v-if="column.key === 'component_type'">
                <Tag v-if="record.component_type === 'head'" color="cyan">Head</Tag>
                <Tag v-else-if="record.component_type === 'backbone'" color="geekblue">Backbone</Tag>
                <Tag v-else>Full</Tag>
              </template>
              <template v-if="column.key === 'created_at'">
                {{ record.created_at ? record.created_at.replace('T', ' ').substring(0, 16) : '-' }}
              </template>
              <template v-if="column.key === 'action'">
                <Space>
                  <Button
                    size="small"
                    type="primary"
                    :disabled="record.stage === 'retired' || record.stage === 'production'"
                    :loading="promotingModel === record.model_version_id"
                    @click="handlePromote(record)"
                  >
                    推进
                  </Button>
                  <Button
                    size="small"
                    danger
                    :disabled="record.stage === 'retired'"
                    @click="handleRetire(record)"
                  >
                    退役
                  </Button>
                  <Tooltip title="查看影子推理报告" v-if="record.stage === 'shadow'">
                    <Button size="small" @click="handleViewShadowReport(record)">
                      <template #icon><ExperimentOutlined /></template>
                    </Button>
                  </Tooltip>
                  <Tooltip title="阶段变更历史">
                    <Button size="small" @click="handleViewStageHistory(record)">
                      <template #icon><HistoryOutlined /></template>
                    </Button>
                  </Tooltip>
                </Space>
              </template>
            </template>
          </Table>
          <div v-if="releaseModels.length === 0 && !releaseLoading" style="text-align: center; padding: 32px; color: #666">
            暂无注册的模型版本
          </div>
        </Card>

        <!-- Version events log -->
        <Card>
          <template #title>
            <Space>
              <HistoryOutlined />
              <span>版本事件日志</span>
            </Space>
          </template>
          <Table
            :columns="eventColumns"
            :data-source="versionEvents"
            :loading="eventsLoading"
            :pagination="{ pageSize: 10, showSizeChanger: false }"
            row-key="id"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'timestamp'">
                {{ record.timestamp ? record.timestamp.replace('T', ' ').substring(0, 19) : '-' }}
              </template>
              <template v-if="column.key === 'transition'">
                <Space>
                  <Tag :color="(STAGE_MAP[record.from_stage] || { color: 'default' }).color">
                    {{ (STAGE_MAP[record.from_stage] || { text: record.from_stage || '?' }).text }}
                  </Tag>
                  <span>→</span>
                  <Tag :color="(STAGE_MAP[record.to_stage] || { color: 'default' }).color">
                    {{ (STAGE_MAP[record.to_stage] || { text: record.to_stage }).text }}
                  </Tag>
                </Space>
              </template>
            </template>
          </Table>
        </Card>
      </Tabs.TabPane>
    </Tabs>

    <!-- ═══════════ Promote Modal ═══════════ -->
    <Modal
      v-model:open="promoteModalVisible"
      title="推进模型阶段"
      :confirmLoading="promotingModel !== null"
      @ok="submitPromote"
      okText="确认推进"
      cancelText="取消"
    >
      <Form layout="vertical" style="margin-top: 16px">
        <Form.Item label="目标阶段">
          <Select v-model:value="promoteForm.target_stage" style="width: 100%">
            <Select.Option
              v-for="stage in (VALID_TRANSITIONS[releaseModels.find(m => m.model_version_id === promoteForm.version_id)?.stage] || [])"
              :key="stage"
              :value="stage"
            >
              {{ STAGE_LABELS[stage] || stage }}
            </Select.Option>
          </Select>
        </Form.Item>
        <Form.Item label="操作人" required>
          <Input v-model:value="promoteForm.triggered_by" placeholder="输入操作人姓名" />
        </Form.Item>
        <Form.Item v-if="promoteForm.target_stage === 'canary'" label="金丝雀摄像头" required>
          <Select v-model:value="promoteForm.canary_camera_id" placeholder="选择目标摄像头" style="width: 100%">
            <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
              {{ cam.camera_id }}
            </Select.Option>
          </Select>
        </Form.Item>
        <Form.Item label="原因">
          <Input.TextArea v-model:value="promoteForm.reason" :rows="2" placeholder="推进原因（可选）" />
        </Form.Item>
      </Form>
    </Modal>

    <!-- ═══════════ Shadow Report Drawer ═══════════ -->
    <Drawer
      v-model:open="shadowReportVisible"
      title="影子推理报告"
      :width="480"
    >
      <div v-if="shadowReportLoading" style="text-align: center; padding: 40px">
        <LoadingOutlined spin style="font-size: 24px" />
      </div>
      <Descriptions v-else-if="shadowReport" :column="1" bordered size="small">
        <Descriptions.Item label="采样总数">{{ shadowReport.total_samples }}</Descriptions.Item>
        <Descriptions.Item label="影子告警率">{{ (shadowReport.shadow_alert_rate * 100).toFixed(1) }}%</Descriptions.Item>
        <Descriptions.Item label="生产告警率">{{ (shadowReport.production_alert_rate * 100).toFixed(1) }}%</Descriptions.Item>
        <Descriptions.Item label="误报差异">
          <span :style="{ color: shadowReport.false_positive_delta > 0 ? '#ff4d4f' : '#52c41a' }">
            {{ shadowReport.false_positive_delta > 0 ? '+' : '' }}{{ shadowReport.false_positive_delta }}
          </span>
        </Descriptions.Item>
        <Descriptions.Item label="平均分数偏差">{{ shadowReport.avg_score_divergence?.toFixed(4) || '-' }}</Descriptions.Item>
        <Descriptions.Item label="平均推理延迟">{{ shadowReport.avg_shadow_latency_ms?.toFixed(1) || '-' }} ms</Descriptions.Item>
      </Descriptions>
      <div v-else style="text-align: center; padding: 40px; color: #666">
        暂无影子推理数据
      </div>
    </Drawer>

    <!-- ═══════════ Stage History Drawer ═══════════ -->
    <Drawer
      v-model:open="stageHistoryVisible"
      title="阶段变更历史"
      :width="560"
    >
      <div v-if="stageHistoryLoading" style="text-align: center; padding: 40px">
        <LoadingOutlined spin style="font-size: 24px" />
      </div>
      <div v-else-if="stageHistory.length > 0">
        <div
          v-for="(event, idx) in stageHistory"
          :key="idx"
          style="padding: 12px; margin-bottom: 8px; background: #1e1e36; border-radius: 6px"
        >
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px">
            <Space>
              <Tag :color="(STAGE_MAP[event.from_stage] || { color: 'default' }).color">
                {{ (STAGE_MAP[event.from_stage] || { text: event.from_stage || '?' }).text }}
              </Tag>
              <span style="color: #8890a0">→</span>
              <Tag :color="(STAGE_MAP[event.to_stage] || { color: 'default' }).color">
                {{ (STAGE_MAP[event.to_stage] || { text: event.to_stage }).text }}
              </Tag>
            </Space>
            <span style="color: #8890a0; font-size: 12px">
              {{ event.timestamp ? event.timestamp.replace('T', ' ').substring(0, 19) : '' }}
            </span>
          </div>
          <div style="font-size: 12px; color: #8890a0">
            <span>操作人: {{ event.triggered_by }}</span>
            <span v-if="event.reason" style="margin-left: 12px">原因: {{ event.reason }}</span>
          </div>
        </div>
      </div>
      <div v-else style="text-align: center; padding: 40px; color: #666">
        暂无阶段变更记录
      </div>
    </Drawer>
  </div>
</template>
