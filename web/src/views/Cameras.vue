<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { Badge, Button, Form, Input, InputNumber, Modal, Select, Space, Typography, message } from 'ant-design-vue'
import { DeleteOutlined, EditOutlined, PlusOutlined } from '@ant-design/icons-vue'

import { addCamera, deleteCamera, getCameraConfig, getCameras, getRegions, getUsbDevices, startCamera, stopCamera, updateCamera } from '../api'
import { testCameraConnection, testCameraConnectionDraft } from '../api/cameras'
import { useWebSocket } from '../composables/useWebSocket'
import { useAuthStore } from '../stores/useAuthStore'
import { extractErrorMessage, toApiError } from '../utils/error'
import HealthBadge from '../components/common/HealthBadge.vue'
import ModeBadge from '../components/common/ModeBadge.vue'
import ConnectionTestResult from '../components/common/ConnectionTestResult.vue'
import PageShell from '../components/common/PageShell.vue'
import AsyncState from '../components/common/AsyncState.vue'
import StatusBanner from '../components/common/StatusBanner.vue'
import ResponsiveTable from '../components/common/ResponsiveTable.vue'
import DangerousActionConfirm from '../components/common/DangerousActionConfirm.vue'
import type { CameraSummary, ConnectionTestResult as ConnTestResult } from '../types/api'

// initialLoading distinguishes the very-first fetch (where the table is empty
// and we want a skeleton instead of the bare ant Table loading spinner over an
// empty body) from subsequent refreshes triggered by WebSocket fallback.
const initialLoading = ref(true)

defineOptions({ name: 'CamerasPage' })

const router = useRouter()
const auth = useAuthStore()
const canManageCameraConfig = computed(() => auth.hasRole(['admin', 'engineer']))
const canOperateCamera = computed(() => auth.hasRole(['admin', 'operator', 'engineer']))
const canDeleteCamera = computed(() => auth.hasRole(['admin']))
const cameras = ref<CameraSummary[]>([])
const loading = ref(false)
const loadError = ref('')
const loadErrorStatus = ref<number | null>(null)
const operatingCameraId = ref<string | null>(null)
const deletingCameraId = ref<string | null>(null)
const initialError = computed(() => (
  loadError.value
    && cameras.value.length === 0
    && ![403, 404].includes(loadErrorStatus.value ?? 0)
    ? loadError.value
    : ''
))
const refreshError = computed(() => (
  loadError.value
    && cameras.value.length > 0
    && ![403, 404].includes(loadErrorStatus.value ?? 0)
    ? loadError.value
    : ''
))

const modalVisible = ref(false)
const editingId = ref<string | null>(null)
const modalLoading = ref(false)

const defaultForm = () => ({
  camera_id: '',
  name: '',
  region_id: undefined as number | undefined,
  source: '',
  protocol: 'rtsp',
  fps_target: 5,
  resolution: '1920,1080',
  gige_exposure: 0,
  gige_gain: 0,
  gige_pixel_format: 'Mono8',
  gige_capture_script: '',
})

const cameraForm = ref(defaultForm())

const usbDevices = ref<{ index: number; name: string; width: number; height: number }[]>([])
const usbLoading = ref(false)
const usbManual = ref(false)
const usbError = ref('')
let usbRequest = 0
let disposed = false
const sourceDrafts = new Map<string, string>()
const regionOptions = ref<{ id: number; name: string }[]>([])
const regionsLoading = ref(false)

watch([() => cameraForm.value, () => cameraForm.value.protocol], async ([form, proto], [previousForm, previousProtocol]) => {
  const request = ++usbRequest
  usbLoading.value = false; usbError.value = ''
  if (form !== previousForm) {
    sourceDrafts.clear(); usbManual.value = false
  } else if (proto !== previousProtocol) {
    sourceDrafts.set(previousProtocol, form.source)
    form.source = sourceDrafts.get(proto) ?? ''
    probeState.value = 'idle'; probeResult.value = null
  }
  if (proto !== 'usb') return
  const originalSource = form.source
  usbLoading.value = true
  try {
    const res = await getUsbDevices()
    if (disposed || request !== usbRequest || form !== cameraForm.value || proto !== form.protocol) return
    usbDevices.value = res.devices
    if (!form.source && !originalSource && usbDevices.value.length > 0) {
      form.source = String(usbDevices.value[0].index)
    }
    usbManual.value = !usbDevices.value.some(device => String(device.index) === form.source)
    if (usbManual.value && !form.source) form.source = '0'
  } catch {
    if (disposed || request !== usbRequest || form !== cameraForm.value || proto !== form.protocol) return
    usbDevices.value = []
    usbError.value = '设备列表暂不可用，可手动填写设备索引；0 表示第一台 USB 相机。'
    usbManual.value = true
    if (!form.source) form.source = '0'
  } finally {
    if (!disposed && request === usbRequest) usbLoading.value = false
  }
})
onBeforeUnmount(() => { disposed = true; usbRequest++ })

function sourcePlaceholder(): string {
  switch (cameraForm.value.protocol) {
    case 'usb':
      return '选择 USB 摄像头'
    case 'file':
      return 'data/video.mp4'
    case 'gige':
      return '192.0.2.11'
    default:
      return 'rtsp://192.0.2.100:554/stream'
  }
}

async function fetchData() {
  loading.value = true
  try {
    const res = await getCameras()
    cameras.value = res.cameras || []
    loadError.value = ''
    loadErrorStatus.value = null
  } catch (e) {
    const apiError = toApiError(e, '摄像头列表加载失败')
    loadError.value = apiError.message
    loadErrorStatus.value = apiError.httpStatus
    if ([403, 404].includes(apiError.httpStatus ?? 0)) cameras.value = []
  } finally {
    loading.value = false
    initialLoading.value = false
  }
}

async function fetchRegions() {
  regionsLoading.value = true
  try {
    const res = await getRegions()
    regionOptions.value = (res.regions || []).map((region: any) => ({
      id: region.id,
      name: region.name,
    }))
  } catch (e) {
    message.error(extractErrorMessage(e, '加载区域列表失败'))
  } finally {
    regionsLoading.value = false
  }
}

const {} = useWebSocket<CameraSummary | CameraSummary[]>({
  topics: ['cameras'],
  onMessage(topic, data) {
    if (topic !== 'cameras') return

    if (data && typeof data === 'object' && !Array.isArray(data) && data.camera_id) {
      const idx = cameras.value.findIndex((camera: any) => camera.camera_id === data.camera_id)
      if (idx >= 0) {
        cameras.value[idx] = { ...cameras.value[idx], ...data }
      } else {
        cameras.value.push(data)
      }
      return
    }

    if (Array.isArray(data)) {
      cameras.value = data
    }
  },
  fallbackPoll: fetchData,
  fallbackInterval: 10000,
})

onMounted(() => {
  fetchData()
  fetchRegions()
})

async function handleStart(id: string) {
  operatingCameraId.value = id
  try {
    await startCamera(id)
    await fetchData()
  } catch (e) {
    message.error(extractErrorMessage(e, '摄像头启动失败'))
  } finally {
    operatingCameraId.value = null
  }
}

async function handleStop(id: string) {
  operatingCameraId.value = id
  try {
    await stopCamera(id)
    await fetchData()
  } catch (e) {
    message.error(extractErrorMessage(e, '摄像头停止失败'))
  } finally {
    operatingCameraId.value = null
  }
}

async function handleDelete(id: string) {
  deletingCameraId.value = id
  try {
    await deleteCamera(id)
    message.success('摄像头已删除')
    await fetchData()
  } catch (e) {
    message.error(extractErrorMessage(e, '删除失败'))
  } finally {
    deletingCameraId.value = null
  }
}

function openAddModal() {
  editingId.value = null
  cameraForm.value = defaultForm()
  modalVisible.value = true
}

async function openEditModal(cameraId: string) {
  editingId.value = cameraId
  modalLoading.value = true
  modalVisible.value = true
  try {
    const cfg = await getCameraConfig(cameraId)
    cameraForm.value = {
      camera_id: cfg.camera_id,
      name: cfg.name,
      region_id: cfg.region_id || undefined,
      source: cfg.source,
      protocol: cfg.protocol,
      fps_target: cfg.fps_target,
      resolution: cfg.resolution.join(','),
      gige_exposure: cfg.gige_exposure || 0,
      gige_gain: cfg.gige_gain || 0,
      gige_pixel_format: cfg.gige_pixel_format || 'Mono8',
      gige_capture_script: cfg.gige_capture_script || '',
    }
  } catch (e) {
    message.error(extractErrorMessage(e, '获取配置失败'))
    modalVisible.value = false
  } finally {
    modalLoading.value = false
  }
}

// 痛点 8: post-save connectivity probe state shown above the modal form
const probeState = ref<'idle' | 'testing' | 'done'>('idle')
const probeResult = ref<ConnTestResult | null>(null)

async function runProbe(cameraId: string) {
  probeState.value = 'testing'
  probeResult.value = null
  try {
    probeResult.value = await testCameraConnection(cameraId)
  } catch {
    probeResult.value = { ok: false, error: 'probe_request_failed' }
  } finally {
    probeState.value = 'done'
  }
}

async function runDraftProbe() {
  if (!cameraForm.value.source) {
    message.warning('请先填写视频源')
    return
  }
  probeState.value = 'testing'
  probeResult.value = null
  try {
    probeResult.value = await testCameraConnectionDraft({
      source: cameraForm.value.source,
      protocol: cameraForm.value.protocol,
    })
  } catch {
    probeResult.value = { ok: false, error: 'probe_request_failed' }
  } finally {
    probeState.value = 'done'
  }
}

async function handleSubmit() {
  try {
    const form = new FormData()
    Object.entries(cameraForm.value).forEach(([key, value]) => {
      if (key === 'region_id') {
        form.append(key, value == null ? '' : String(value))
        return
      }
      if (value !== '' && value !== null && value !== undefined) {
        form.append(key, String(value))
      }
    })

    const cameraId = cameraForm.value.camera_id
    if (editingId.value) {
      const res = await updateCamera(editingId.value, form)
      message.success(res.message || '配置已更新')
    } else {
      await addCamera(form)
      message.success('摄像头已添加')
    }

    fetchData()

    // 痛点 8: live-probe the camera before closing the modal so the user
    // immediately sees whether the source is reachable.
    if (cameraId) {
      await runProbe(cameraId)
      if (probeResult.value?.ok) {
        modalVisible.value = false
        cameraForm.value = defaultForm()
        editingId.value = null
        probeState.value = 'idle'
        probeResult.value = null
      }
      // Connection failed → keep modal open so the user sees the reason
    } else {
      modalVisible.value = false
      cameraForm.value = defaultForm()
      editingId.value = null
    }
  } catch (e) {
    message.error(extractErrorMessage(e, editingId.value ? '更新失败' : '添加失败'))
  }
}

const columns = [
  { title: '状态', key: 'status', width: 80 },
  { title: '健康度', key: 'health', width: 100 },
  { title: '运行模式', key: 'pipeline_mode', width: 110 },
  { title: '摄像头 ID', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '名称', dataIndex: 'name', key: 'name' },
  { title: '区域', dataIndex: 'region_name', key: 'region_name', width: 140 },
  {
    title: '已采集帧',
    key: 'frames',
    customRender: ({ record }: any) => record.stats?.frames_captured || 0,
  },
  {
    title: '延迟',
    key: 'latency',
    customRender: ({ record }: any) => (
      record.stats ? `${record.stats.avg_latency_ms?.toFixed(1)}ms` : '-'
    ),
  },
  {
    title: '告警数',
    key: 'alerts',
    customRender: ({ record }: any) => record.stats?.alerts_emitted || 0,
  },
  { title: '操作', key: 'action', width: 260 },
]

function openCameraDetail(cameraId: string) {
  router.push(`/cameras/${cameraId}`)
}

function cameraRowProps(record: CameraSummary) {
  return {
    role: 'link',
    tabindex: 0,
    'aria-label': `查看摄像头 ${record.name || record.camera_id} 详情`,
    onClick: () => openCameraDetail(record.camera_id),
    onKeydown: (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target?.closest('button, a, input, select, textarea, [role="button"]')) return
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        openCameraDetail(record.camera_id)
      }
    },
  }
}
</script>

<template>
  <PageShell
    class="cameras-page"
    title="摄像头"
    description="查看视频源、运行健康度和检测链状态。"
    :busy="loading"
  >
    <template #actions>
      <Button v-if="canManageCameraConfig" type="primary" @click="openAddModal">
        <PlusOutlined /> 新增摄像头
      </Button>
    </template>

    <Modal
      v-model:open="modalVisible"
      :title="editingId ? '编辑摄像头' : '新增摄像头'"
      :ok-text="editingId ? '保存' : '添加'"
      cancel-text="取消"
      :confirm-loading="modalLoading"
      @ok="handleSubmit"
    >
      <Form layout="vertical" style="margin-top: 16px">
        <ConnectionTestResult
          v-if="probeState !== 'idle'"
          :state="probeState"
          :result="probeResult"
          style="margin-bottom: 12px"
        />
        <Form.Item label="摄像头 ID" required>
          <Input v-model:value="cameraForm.camera_id" placeholder="cam_02" :disabled="!!editingId" />
        </Form.Item>
        <Form.Item label="名称" required>
          <Input v-model:value="cameraForm.name" placeholder="仓库入口摄像头" />
        </Form.Item>
        <Form.Item label="区域">
          <Select
            v-model:value="cameraForm.region_id"
            :loading="regionsLoading"
            allow-clear
            placeholder="请选择区域"
          >
            <Select.Option v-for="region in regionOptions" :key="region.id" :value="region.id">
              {{ region.name }}
            </Select.Option>
          </Select>
        </Form.Item>
        <Form.Item label="相机类型" required>
          <Select v-model:value="cameraForm.protocol" data-test="camera-type">
            <Select.Option value="usb">USB 相机</Select.Option>
            <Select.Option value="rtsp">网络相机（RTSP）</Select.Option>
            <Select.Option value="gige">工业相机（GigE Vision）</Select.Option>
            <Select.Option value="file">视频文件</Select.Option>
          </Select>
        </Form.Item>
        <Form.Item :label="cameraForm.protocol === 'usb' ? 'USB 设备' : cameraForm.protocol === 'rtsp' ? '网络视频流地址' : '视频源'" required>
          <Select
            v-if="cameraForm.protocol === 'usb' && !usbManual"
            v-model:value="cameraForm.source"
            :loading="usbLoading"
            :placeholder="usbLoading ? '正在检测...' : '选择 USB 摄像头'"
            :not-found-content="usbLoading ? '检测中...' : '未检测到 USB 摄像头'"
          >
            <Select.Option v-for="device in usbDevices" :key="device.index" :value="String(device.index)">
              {{ device.name }} ({{ device.width }}x{{ device.height }})
            </Select.Option>
          </Select>
          <InputNumber v-else-if="cameraForm.protocol === 'usb'" data-test="usb-device-index"
            :value="cameraForm.source === '' ? undefined : Number(cameraForm.source)" :min="0" :precision="0"
            @update:value="value => cameraForm.source = value == null ? '' : String(value)" />
          <Input v-else v-model:value="cameraForm.source" :placeholder="sourcePlaceholder()" />
          <template v-if="cameraForm.protocol === 'usb'">
            <Button size="small" type="link" :disabled="usbLoading" @click="usbManual = !usbManual">{{ usbManual ? '从设备列表选择' : '手动填写设备索引' }}</Button>
            <p class="source-help">设备位于运行 Argus 的电脑上；0 表示第一台 USB 相机，不是浏览网页电脑的设备。</p>
            <p v-if="usbError" role="status">{{ usbError }}</p>
          </template>
          <p v-else-if="cameraForm.protocol === 'rtsp'" class="source-help">填写摄像头的完整 RTSP 流地址，不是摄像头管理网页地址。</p>
          <Button
            size="small"
            style="margin-top: 6px"
            :loading="probeState === 'testing'"
            @click="runDraftProbe"
          >
            测试连接
          </Button>
        </Form.Item>

        <Space>
          <Form.Item label="目标帧率">
            <InputNumber v-model:value="cameraForm.fps_target" :min="1" :max="120" />
          </Form.Item>
          <Form.Item label="分辨率">
            <Select v-model:value="cameraForm.resolution" style="width: 140px">
              <Select.Option value="1920,1080">1920x1080</Select.Option>
              <Select.Option value="1280,720">1280x720</Select.Option>
              <Select.Option value="640,480">640x480</Select.Option>
            </Select>
          </Form.Item>
        </Space>

        <template v-if="cameraForm.protocol === 'gige'">
          <Typography.Text type="secondary" style="display: block; margin: 8px 0 12px">GigE Vision 参数</Typography.Text>
          <Space>
            <Form.Item label="曝光 (μs, 0=自动)">
              <InputNumber v-model:value="cameraForm.gige_exposure" :min="0" :max="1000000" style="width: 140px" />
            </Form.Item>
            <Form.Item label="增益 (dB, 0=自动)">
              <InputNumber v-model:value="cameraForm.gige_gain" :min="0" :max="48" :step="0.5" style="width: 120px" />
            </Form.Item>
          </Space>
          <Space>
            <Form.Item label="像素格式">
              <Select v-model:value="cameraForm.gige_pixel_format" style="width: 160px">
                <Select.Option value="Mono8">Mono8 (灰度)</Select.Option>
                <Select.Option value="BayerBG8">BayerBG8</Select.Option>
                <Select.Option value="BayerGB8">BayerGB8</Select.Option>
                <Select.Option value="BayerGR8">BayerGR8</Select.Option>
                <Select.Option value="BayerRG8">BayerRG8</Select.Option>
              </Select>
            </Form.Item>
            <Form.Item label="采集脚本路径">
              <Input v-model:value="cameraForm.gige_capture_script" placeholder="/home/user/scripts/gst_camera.sh" style="width: 300px" />
            </Form.Item>
          </Space>
        </template>
      </Form>
    </Modal>

    <StatusBanner
      v-if="refreshError"
      class="cameras-page__refresh-error"
      status="error"
      title="刷新失败，当前显示上次成功获取的数据"
      :description="refreshError"
    >
      <template #action>
        <Button
          size="small"
          aria-label="重试摄像头列表"
          @click="fetchData"
        >
          重试
        </Button>
      </template>
    </StatusBanner>

    <AsyncState
      :loading="initialLoading"
      :refreshing="loading && !initialLoading"
      :error="initialError"
      :forbidden="loadErrorStatus === 403"
      :not-found="loadErrorStatus === 404"
      :forbidden-description="loadError || undefined"
      :not-found-description="loadError || undefined"
      :empty="!initialLoading && !loadError && cameras.length === 0"
      skeleton-type="table"
      :skeleton-rows="6"
      empty-title="尚未配置摄像头"
      empty-description="添加视频源后，才能开始预览和检测。"
      :empty-action-text="canManageCameraConfig ? '新增摄像头' : ''"
      @retry="fetchData"
      @empty-action="openAddModal"
    >
      <ResponsiveTable
        :columns="columns"
        :data-source="cameras"
        :loading="loading"
        :pagination="false"
        row-key="camera_id"
        :row-class-name="() => 'camera-row'"
        :custom-row="cameraRowProps"
        :min-width="1040"
        accessible-label="摄像头列表，可横向滚动"
        style="cursor: pointer"
      >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'status'">
          <Badge :status="record.connected ? 'success' : 'default'" />
          {{ record.connected ? '在线' : '离线' }}
        </template>
        <template v-else-if="column.key === 'health'">
          <HealthBadge :health="record.health" :connected="record.connected" />
        </template>
        <template v-else-if="column.key === 'pipeline_mode'">
          <ModeBadge :mode="record.pipeline_mode" />
          <span v-if="!record.pipeline_mode || record.pipeline_mode === 'active'" style="color: var(--ink-3); font-size: 12px">—</span>
        </template>
        <template v-else-if="column.key === 'region_name'">
          {{ record.region_name || '-' }}
        </template>
        <template v-else-if="column.key === 'action'">
          <Space @click.stop>
            <Button
              v-if="!record.connected && canOperateCamera"
              type="primary"
              size="small"
              :loading="operatingCameraId === record.camera_id"
              :aria-label="`启动摄像头 ${record.name || record.camera_id}`"
              @click="handleStart(record.camera_id)"
            >
              启动
            </Button>
            <Button
              v-else-if="record.connected && canOperateCamera"
              danger
              size="small"
              :loading="operatingCameraId === record.camera_id"
              :aria-label="`停止摄像头 ${record.name || record.camera_id}`"
              @click="handleStop(record.camera_id)"
            >
              停止
            </Button>
            <Button
              v-if="canManageCameraConfig"
              size="small"
              :aria-label="`编辑摄像头 ${record.name || record.camera_id}`"
              @click="openEditModal(record.camera_id)"
            >
              <template #icon><EditOutlined /></template>
            </Button>
            <Button size="small" :aria-label="`查看摄像头 ${record.name || record.camera_id} 详情`" @click="openCameraDetail(record.camera_id)">
              详情
            </Button>
            <DangerousActionConfirm
              v-if="canDeleteCamera"
              title="确认删除摄像头"
              :description="`确定要删除摄像头 ${record.name || record.camera_id} 吗？删除后需重新添加。`"
              action-label="删除"
              :aria-label="`删除摄像头 ${record.name || record.camera_id}`"
              :loading="deletingCameraId === record.camera_id"
              confirm-text="删除"
              @confirm="handleDelete(record.camera_id)"
            >
              <template #icon><DeleteOutlined /></template>
            </DangerousActionConfirm>
          </Space>
        </template>
      </template>
      </ResponsiveTable>
    </AsyncState>
  </PageShell>
</template>

<style scoped>
.cameras-page__refresh-error {
  margin-bottom: 12px;
}

:deep(.camera-row:focus-visible) {
  outline: 3px solid rgba(22, 119, 255, 0.45);
  outline-offset: -3px;
}
</style>
