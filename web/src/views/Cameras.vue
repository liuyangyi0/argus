<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { Badge, Button, Form, Input, InputNumber, Modal, Select, Space, Table, Typography, message } from 'ant-design-vue'
import { DeleteOutlined, EditOutlined, PlusOutlined } from '@ant-design/icons-vue'

import { addCamera, deleteCamera, getCameraConfig, getCameras, getRegions, getUsbDevices, startCamera, stopCamera, updateCamera } from '../api'
import { useWebSocket } from '../composables/useWebSocket'
import { extractErrorMessage } from '../utils/error'

defineOptions({ name: 'CamerasPage' })

const router = useRouter()
const cameras = ref<any[]>([])
const loading = ref(true)

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
const regionOptions = ref<{ id: number; name: string }[]>([])
const regionsLoading = ref(false)

watch(() => cameraForm.value.protocol, async (proto) => {
  if (proto !== 'usb') return

  usbLoading.value = true
  try {
    const res = await getUsbDevices()
    usbDevices.value = res.devices ?? res
    if (usbDevices.value.length > 0 && !editingId.value) {
      cameraForm.value.source = String(usbDevices.value[0].index)
    }
  } catch {
    usbDevices.value = []
  } finally {
    usbLoading.value = false
  }
})

function sourcePlaceholder(): string {
  switch (cameraForm.value.protocol) {
    case 'usb':
      return '选择 USB 摄像头'
    case 'file':
      return 'data/video.mp4'
    case 'gige':
      return '192.168.66.223'
    default:
      return 'rtsp://admin:pass@192.168.1.100:554/stream'
  }
}

async function fetchData() {
  try {
    const res = await getCameras()
    cameras.value = res.cameras || []
  } catch {
  } finally {
    loading.value = false
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

const {} = useWebSocket({
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
  await startCamera(id)
  fetchData()
}

async function handleStop(id: string) {
  await stopCamera(id)
  fetchData()
}

function handleDelete(id: string) {
  Modal.confirm({
    title: '确认删除',
    content: `确定要删除摄像头 "${id}" 吗？删除后需重新添加。`,
    okText: '删除',
    okType: 'danger',
    cancelText: '取消',
    async onOk() {
      try {
        await deleteCamera(id)
        message.success('摄像头已删除')
        fetchData()
      } catch (e) {
        message.error(extractErrorMessage(e, '删除失败'))
      }
    },
  })
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

    if (editingId.value) {
      const res = await updateCamera(editingId.value, form)
      message.success(res.message || '配置已更新')
    } else {
      await addCamera(form)
      message.success('摄像头已添加')
    }

    modalVisible.value = false
    cameraForm.value = defaultForm()
    editingId.value = null
    fetchData()
  } catch (e) {
    message.error(extractErrorMessage(e, editingId.value ? '更新失败' : '添加失败'))
  }
}

const columns = [
  { title: '状态', key: 'status', width: 80 },
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
</script>

<template>
  <main class="glass" style=" padding: 24px; border-radius: var(--r-lg); min-width: 0; display: flex; flex-direction: column; flex: 1;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px">
      <Typography.Title :level="3" style="margin: 0">摄像头</Typography.Title>
      <Button type="primary" @click="openAddModal">
        <PlusOutlined /> 新增摄像头
      </Button>
    </div>

    <Modal
      v-model:open="modalVisible"
      :title="editingId ? '编辑摄像头' : '新增摄像头'"
      :ok-text="editingId ? '保存' : '添加'"
      cancel-text="取消"
      :confirm-loading="modalLoading"
      @ok="handleSubmit"
    >
      <Form layout="vertical" style="margin-top: 16px">
        <Form.Item label="摄像头 ID" required>
          <Input v-model:value="cameraForm.camera_id" placeholder="cam_02" :disabled="!!editingId" />
        </Form.Item>
        <Form.Item label="名称" required>
          <Input v-model:value="cameraForm.name" placeholder="反应堆厂房入口" />
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
        <Form.Item label="视频源" required>
          <Select
            v-if="cameraForm.protocol === 'usb'"
            v-model:value="cameraForm.source"
            :loading="usbLoading"
            :placeholder="usbLoading ? '正在检测...' : '选择 USB 摄像头'"
            :not-found-content="usbLoading ? '检测中...' : '未检测到 USB 摄像头'"
          >
            <Select.Option v-for="device in usbDevices" :key="device.index" :value="String(device.index)">
              {{ device.name }} ({{ device.width }}x{{ device.height }})
            </Select.Option>
          </Select>
          <Input v-else v-model:value="cameraForm.source" :placeholder="sourcePlaceholder()" />
        </Form.Item>

        <Space>
          <Form.Item label="协议">
            <Select v-model:value="cameraForm.protocol" style="width: 140px">
              <Select.Option value="rtsp">RTSP</Select.Option>
              <Select.Option value="usb">USB</Select.Option>
              <Select.Option value="gige">GigE Vision</Select.Option>
              <Select.Option value="file">文件</Select.Option>
            </Select>
          </Form.Item>
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

    <Table
      :columns="columns"
      :data-source="cameras"
      :loading="loading"
      :pagination="false"
      row-key="camera_id"
      :row-class-name="() => 'camera-row'"
      :custom-row="(record: any) => ({ onClick: () => router.push(`/cameras/${record.camera_id}`) })"
      style="cursor: pointer"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'status'">
          <Badge :status="record.connected ? 'success' : 'default'" />
          {{ record.connected ? '在线' : '离线' }}
        </template>
        <template v-else-if="column.key === 'region_name'">
          {{ record.region_name || '-' }}
        </template>
        <template v-else-if="column.key === 'action'">
          <Space @click.stop>
            <Button v-if="!record.connected" type="primary" size="small" @click="handleStart(record.camera_id)">
              启动
            </Button>
            <Button v-else danger size="small" @click="handleStop(record.camera_id)">
              停止
            </Button>
            <Button size="small" @click="openEditModal(record.camera_id)">
              <template #icon><EditOutlined /></template>
            </Button>
            <Button size="small" @click="router.push(`/cameras/${record.camera_id}`)">
              详情
            </Button>
            <Button size="small" danger @click="handleDelete(record.camera_id)">
              <template #icon><DeleteOutlined /></template>
            </Button>
          </Space>
        </template>
      </template>
    </Table>
  </main>
</template>
