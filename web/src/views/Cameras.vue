<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { useRouter } from 'vue-router'

defineOptions({ name: 'CamerasPage' })
import { Table, Badge, Button, Typography, Space, Modal, Form, Input, Select, InputNumber, message } from 'ant-design-vue'
import { PlusOutlined, DeleteOutlined } from '@ant-design/icons-vue'
import { getCameras, startCamera, stopCamera, getUsbDevices, addCamera, deleteCamera } from '../api'
import { useWebSocket } from '../composables/useWebSocket'

const router = useRouter()
const cameras = ref<any[]>([])
const loading = ref(true)
const addModalVisible = ref(false)
const addForm = ref({
  camera_id: '',
  name: '',
  source: '',
  protocol: 'rtsp',
  fps_target: 5,
  resolution: '1920,1080',
})

const usbDevices = ref<{ index: number; name: string; width: number; height: number }[]>([])
const usbLoading = ref(false)

watch(() => addForm.value.protocol, async (proto) => {
  if (proto === 'usb') {
    usbLoading.value = true
    try {
      const res = await getUsbDevices()
      usbDevices.value = res
      if (res.length > 0) {
        addForm.value.source = String(res[0].index)
      }
    } catch { usbDevices.value = [] }
    finally { usbLoading.value = false }
  }
})

async function fetchData() {
  try {
    const res = await getCameras()
    cameras.value = res.cameras || []
  } catch {
    // Network error or timeout — keep existing data
  } finally {
    loading.value = false
  }
}

const { } = useWebSocket({
  topics: ['cameras'],
  onMessage(topic, data) {
    if (topic === 'cameras') {
      // Backend pushes a single camera status object, not the full list.
      // Merge into existing list by camera_id instead of replacing.
      if (data && typeof data === 'object' && !Array.isArray(data) && data.camera_id) {
        const idx = cameras.value.findIndex((c: any) => c.camera_id === data.camera_id)
        if (idx >= 0) {
          cameras.value[idx] = { ...cameras.value[idx], ...data }
        } else {
          cameras.value.push(data)
        }
      } else if (Array.isArray(data)) {
        cameras.value = data
      }
    }
  },
  fallbackPoll: fetchData,
  fallbackInterval: 10000,
})

onMounted(fetchData)

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
      } catch (e: any) {
        message.error(e.message || '删除失败')
      }
    },
  })
}

async function handleAddCamera() {
  try {
    const form = new FormData()
    Object.entries(addForm.value).forEach(([k, v]) => form.append(k, String(v)))
    await addCamera(form)
    message.success('摄像头已添加')
    addModalVisible.value = false
    addForm.value = { camera_id: '', name: '', source: '', protocol: 'rtsp', fps_target: 5, resolution: '1920,1080' }
    fetchData()
  } catch (e: any) {
    message.error(e.message || '添加失败')
  }
}

const columns = [
  {
    title: '状态',
    key: 'status',
    width: 80,
  },
  { title: '摄像头 ID', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '名称', dataIndex: 'name', key: 'name' },
  {
    title: '已采集帧',
    key: 'frames',
    customRender: ({ record }: any) => record.stats?.frames_captured || 0,
  },
  {
    title: '延迟',
    key: 'latency',
    customRender: ({ record }: any) =>
      record.stats ? `${record.stats.avg_latency_ms?.toFixed(1)}ms` : '—',
  },
  {
    title: '告警数',
    key: 'alerts',
    customRender: ({ record }: any) => record.stats?.alerts_emitted || 0,
  },
  {
    title: '操作',
    key: 'action',
    width: 200,
  },
]
</script>

<template>
  <div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px">
      <Typography.Title :level="3" style="margin: 0">摄像头</Typography.Title>
      <Button type="primary" @click="addModalVisible = true">
        <PlusOutlined /> 新增摄像头
      </Button>
    </div>

    <!-- Add Camera Modal -->
    <Modal v-model:open="addModalVisible" title="新增摄像头" @ok="handleAddCamera" ok-text="添加" cancel-text="取消">
      <Form layout="vertical" style="margin-top: 16px">
        <Form.Item label="摄像头 ID" required>
          <Input v-model:value="addForm.camera_id" placeholder="cam_02" />
        </Form.Item>
        <Form.Item label="名称" required>
          <Input v-model:value="addForm.name" placeholder="反应堆厂房入口" />
        </Form.Item>
        <Form.Item label="视频源" required>
          <Select
            v-if="addForm.protocol === 'usb'"
            v-model:value="addForm.source"
            :loading="usbLoading"
            :placeholder="usbLoading ? '正在检测...' : '选择 USB 摄像头'"
            :not-found-content="usbLoading ? '检测中...' : '未检测到 USB 摄像头'"
          >
            <Select.Option v-for="d in usbDevices" :key="d.index" :value="String(d.index)">
              {{ d.name }} ({{ d.width }}x{{ d.height }})
            </Select.Option>
          </Select>
          <Input v-else v-model:value="addForm.source" :placeholder="addForm.protocol === 'file' ? 'data/video.mp4' : 'rtsp://admin:pass@192.168.1.100:554/stream'" />
        </Form.Item>
        <Space>
          <Form.Item label="协议">
            <Select v-model:value="addForm.protocol" style="width: 120px">
              <Select.Option value="rtsp">RTSP</Select.Option>
              <Select.Option value="usb">USB</Select.Option>
              <Select.Option value="file">文件</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item label="目标帧率">
            <InputNumber v-model:value="addForm.fps_target" :min="1" :max="30" />
          </Form.Item>
          <Form.Item label="分辨率">
            <Select v-model:value="addForm.resolution" style="width: 140px">
              <Select.Option value="1920,1080">1920x1080</Select.Option>
              <Select.Option value="1280,720">1280x720</Select.Option>
              <Select.Option value="640,480">640x480</Select.Option>
            </Select>
          </Form.Item>
        </Space>
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
        <template v-if="column.key === 'action'">
          <Space @click.stop>
            <Button v-if="!record.connected" type="primary" size="small" @click="handleStart(record.camera_id)">
              启动
            </Button>
            <Button v-else danger size="small" @click="handleStop(record.camera_id)">
              停止
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
  </div>
</template>
