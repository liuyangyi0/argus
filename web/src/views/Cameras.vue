<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { Table, Badge, Button, Typography, Space } from 'ant-design-vue'
import { getCameras, startCamera, stopCamera } from '../api'

const router = useRouter()
const cameras = ref<any[]>([])
const loading = ref(true)
let timer: ReturnType<typeof setInterval>

async function fetchData() {
  try {
    const res = await getCameras()
    cameras.value = res.data
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchData()
  timer = setInterval(fetchData, 10000)
})
onUnmounted(() => clearInterval(timer))

async function handleStart(id: string) {
  await startCamera(id)
  fetchData()
}

async function handleStop(id: string) {
  await stopCamera(id)
  fetchData()
}

const columns = [
  {
    title: '状态',
    key: 'status',
    width: 80,
    customRender: ({ record }: any) => {
      return record.connected ? '● 在线' : '○ 离线'
    },
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
    </div>

    <Table
      :columns="columns"
      :data-source="cameras"
      :loading="loading"
      :pagination="false"
      row-key="camera_id"
      :row-class-name="() => 'camera-row'"
      @row:click="(record: any) => router.push(`/cameras/${record.camera_id}`)"
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
          </Space>
        </template>
      </template>
    </Table>
  </div>
</template>
