<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import {
  Tabs, Table, Card, Button, Select, Form, InputNumber, Space,
  Typography, Progress, Tag, Modal, message, Descriptions, Tooltip,
} from 'ant-design-vue'
import {
  PlayCircleOutlined, RocketOutlined, ReloadOutlined,
  CheckCircleOutlined, CloseCircleOutlined, LoadingOutlined,
  HistoryOutlined, ExperimentOutlined, SwapOutlined,
  RollbackOutlined, CheckOutlined,
} from '@ant-design/icons-vue'
import {
  getCameras, getBaselines, startCapture, startTraining,
  getModels, deployModel, getTrainingHistory, getTasks, dismissTask,
  optimizeBaseline, previewOptimize, getModelRegistry, activateModel,
  rollbackModel, compareModels,
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

// ── WebSocket for tasks ──
useWebSocket({
  topics: ['tasks'],
  onMessage: (_topic, data) => { tasks.value = data.tasks || [] },
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

const GRADE_COLORS: Record<string, string> = {
  A: 'green', B: 'blue', C: 'orange', F: 'red',
}

const RECOMMENDATION_COLORS: Record<string, string> = {
  deploy: 'green', keep_old: 'orange', review: 'blue',
}
const RECOMMENDATION_TEXT: Record<string, string> = {
  deploy: '推荐部署新模型', keep_old: '保留旧模型', review: '需要人工审核',
}

// ── Data loading ──

async function loadCameras() {
  try {
    const res = await getCameras()
    cameras.value = res.data
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

// ── Actions ──

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
}

// ── Computed: active tasks by type ──

const captureTasks = computed(() =>
  tasks.value.filter(t => t.task_type === 'baseline_capture')
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
  { title: '状态', dataIndex: 'status', key: 'status' },
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

onMounted(async () => {
  await loadCameras()
  loadBaselines()
  loadTasks()
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
                <LoadingOutlined v-if="task.status === 'running'" spin style="color: #3b82f6" />
                <CheckCircleOutlined v-else-if="task.status === 'complete'" style="color: #52c41a" />
                <CloseCircleOutlined v-else-if="task.status === 'failed'" style="color: #ff4d4f" />
                <span style="font-weight: 500">基线采集</span>
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
                  采集新基线
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
              <template v-if="column.key === 'status'">
                <Tag color="green">就绪</Tag>
              </template>
              <template v-if="column.key === 'action'">
                <Tooltip title="多样性优选：预览后确认保留最具代表性的子集">
                  <Button
                    size="small"
                    :loading="optimizingBaseline === `${record.camera_id}-${record.version}`"
                    :disabled="record.image_count < 30"
                    @click="handleOptimize(record)"
                  >
                    优化
                  </Button>
                </Tooltip>
              </template>
            </template>
          </Table>
          <div v-if="baselines.length === 0 && !baselinesLoading" style="text-align: center; padding: 32px; color: #666">
            暂无基线数据，请先采集基线图片
          </div>
        </Card>

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
      </Tabs.TabPane>
    </Tabs>
  </div>
</template>
