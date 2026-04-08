<script setup lang="ts">
import { ref, h } from 'vue'
import {
  Table, Card, Button, Select, Form, InputNumber, Space,
  Progress, Tag, Modal, message, Tooltip, Input,
  Radio, Collapse, Slider, Drawer, Divider,
} from 'ant-design-vue'
import {
  PlayCircleOutlined, ReloadOutlined,
  CheckCircleOutlined, CloseCircleOutlined, LoadingOutlined,
  ExperimentOutlined, PauseCircleOutlined,
  StopOutlined, CaretRightOutlined, SafetyCertificateOutlined,
  TeamOutlined, MergeCellsOutlined, DeleteOutlined,
} from '@ant-design/icons-vue'
import {
  getBaselines, startCapture,
  optimizeBaseline, previewOptimize,
  startCaptureJob, pauseCaptureJob, resumeCaptureJob, abortCaptureJob,
  getBaselineVersions, verifyBaseline, activateBaseline, retireBaseline, deleteBaselineVersion,
  getCameraGroups, mergeGroupBaseline, mergeFalsePositives,
} from '../../api'
import {
  SESSION_LABELS, SAMPLING_STRATEGIES, JOB_STATUS_MAP, BASELINE_STATE_MAP,
} from '../../composables/useModelState'

const props = defineProps<{
  cameras: any[]
  captureTasks: any[]
  taskTitle: (task: any) => string
  taskProgressStatus: (task: any) => string
  canPauseTask: (task: any) => boolean
  canResumeTask: (task: any) => boolean
  canAbortTask: (task: any) => boolean
  canDismissTask: (task: any) => boolean
  handleDismissTask: (taskId: string) => void
  loadTasks: () => void
}>()

// ── State ──
const baselines = ref<any[]>([])
const baselinesLoading = ref(false)
const captureModalVisible = ref(false)
const captureForm = ref({ camera_id: '', count: 100, interval: 2.0, session_label: 'daytime' })
const captureSubmitting = ref(false)

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

const optimizingBaseline = ref<string | null>(null)

const versionDrawerVisible = ref(false)
const versionDrawerCamera = ref('')
const baselineVersions = ref<any[]>([])
const versionsLoading = ref(false)
const verifyingVersion = ref<string | null>(null)
const activatingVersion = ref<string | null>(null)

const cameraGroups = ref<any[]>([])
const groupsLoading = ref(false)
const mergingGroup = ref<string | null>(null)
const mergingFP = ref<string | null>(null)
const deletingBaseline = ref<string | null>(null)

// ── Data loading ──
async function loadBaselines() {
  baselinesLoading.value = true
  try {
    const res = await getBaselines()
    baselines.value = res.baselines || []
  } catch (e) {
    console.error('Failed to load baselines', e)
  } finally {
    baselinesLoading.value = false
  }
}

async function loadBaselineVersions(cameraId: string) {
  versionsLoading.value = true
  try {
    const res = await getBaselineVersions({ camera_id: cameraId })
    baselineVersions.value = res.versions || []
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
    cameraGroups.value = res.groups || []
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
  const verifiedByRef = ref('')
  Modal.confirm({
    title: '审核基线版本',
    content: () => h('div', [
      h('p', `确认审核通过 ${record.version}？`),
      h(Input, {
        placeholder: '请输入审核人姓名',
        value: verifiedByRef.value,
        'onUpdate:value': (v: string) => { verifiedByRef.value = v },
      }),
    ]),
    okText: '确认审核',
    cancelText: '取消',
    async onOk() {
      const verifiedBy = verifiedByRef.value.trim() || 'operator'
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
  const reasonRef = ref('')
  Modal.confirm({
    title: '退役基线版本',
    content: () => h('div', [
      h('p', `确定退役 ${record.version}？退役后将保留数据但不再用于训练。`),
      h(Input, {
        placeholder: '退役原因（可选）',
        value: reasonRef.value,
        'onUpdate:value': (v: string) => { reasonRef.value = v },
      }),
    ]),
    okText: '确认退役',
    okType: 'danger',
    cancelText: '取消',
    async onOk() {
      try {
        await retireBaseline({
          camera_id: record.camera_id,
          version: record.version,
          reason: reasonRef.value.trim() || '手动退役',
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
        message.success(`组基线合并完成: ${res.version}, ${res.image_count} 张图片`)
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
        message.success(`误报合并完成: ${res.version}, 新增 ${res.fp_included} 张误报帧`)
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
    props.loadTasks()
  } catch (e: any) {
    message.error(e.response?.data?.error || '启动采集失败')
  } finally {
    captureSubmitting.value = false
  }
}

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
    props.loadTasks()
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
    props.loadTasks()
  } catch (e: any) {
    message.error(e.response?.data?.error || '暂停失败')
  }
}

async function handleResumeJob(taskId: string) {
  try {
    await resumeCaptureJob(taskId)
    message.success('任务已恢复')
    props.loadTasks()
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
        props.loadTasks()
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
    const { total, keep, move } = preview
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
          message.success(`优化完成: 保留 ${res.selected} 张, 移除 ${res.moved} 张`)
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

// ── Columns ──
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

// ── Init ──
defineExpose({ loadBaselines, loadCameraGroups })

loadBaselines()
loadCameraGroups()
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
          <PauseCircleOutlined v-if="task.status === 'paused'" style="color: #faad14" />
          <LoadingOutlined v-else-if="task.status === 'running'" spin style="color: #3b82f6" />
          <CheckCircleOutlined v-else-if="task.status === 'complete'" style="color: #52c41a" />
          <CloseCircleOutlined v-else-if="task.status === 'failed'" style="color: #ff4d4f" />
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
      :row-key="(record: any) => `${record.camera_id}-${record.version}`"
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
              <Button size="small" :loading="mergingFP === record.camera_id" @click="handleMergeFP(record.camera_id)">
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
          <Button size="small" type="primary" :loading="mergingGroup === record.group_id" @click="handleMergeGroup(record.group_id)">
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
          <Button v-if="record.state === 'draft'" size="small" type="primary" :loading="verifyingVersion === record.version" @click="handleVerify(record)">
            审核
          </Button>
          <Button v-if="record.state === 'verified'" size="small" type="primary" :loading="activatingVersion === record.version" @click="handleActivateBaseline(record)">
            激活
          </Button>
          <Button v-if="record.state === 'active'" size="small" danger @click="handleRetireBaseline(record)">
            退役
          </Button>
          <Button v-if="record.state !== 'active'" size="small" danger :loading="deletingBaseline === `${record.camera_id}-${record.version}`" @click="handleDeleteBaseline(record)">
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
          <InputNumber v-model:value="advCaptureForm.target_frames" :min="100" :max="10000" :step="100" style="width: 140px" />
        </Form.Item>
        <Form.Item label="持续时长（小时）">
          <InputNumber v-model:value="advCaptureForm.duration_hours" :min="1" :max="168" :step="1" style="width: 140px" />
        </Form.Item>
      </Space>
      <Collapse ghost>
        <Collapse.Panel key="advanced" header="高级参数">
          <Form.Item label="多样性阈值" v-if="advCaptureForm.sampling_strategy !== 'uniform'">
            <Slider
              v-model:value="advCaptureForm.diversity_threshold"
              :min="0.1" :max="0.9" :step="0.05"
              :marks="{ 0.1: '低', 0.3: '默认', 0.9: '高' }"
            />
            <div style="font-size: 12px; color: #8890a0">
              值越高，保留的帧越多样；值越低，接受更多相似帧
            </div>
          </Form.Item>
          <Form.Item label="每时段帧数" v-if="advCaptureForm.sampling_strategy === 'scheduled'">
            <InputNumber v-model:value="advCaptureForm.frames_per_period" :min="5" :max="200" :step="5" style="width: 140px" />
            <div style="font-size: 12px; color: #8890a0; margin-top: 4px">
              每个时段（清晨/正午/傍晚/深夜）采集的目标帧数
            </div>
          </Form.Item>
        </Collapse.Panel>
      </Collapse>
    </Form>
  </Modal>
</template>
