<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  Table, Tag, Button, Space, Typography, Select, Drawer,
  Descriptions, Image, Divider, message,
} from 'ant-design-vue'
import { getAlerts, getCameras, acknowledgeAlert, markFalsePositive } from '../api'
import { useWebSocket } from '../composables/useWebSocket'
import ReplayPlayer from '../components/ReplayPlayer.vue'

const alerts = ref<any[]>([])
const cameras = ref<any[]>([])
const loading = ref(true)
const filters = ref({ camera_id: '', severity: '' })
const detailVisible = ref(false)
const selectedAlert = ref<any>(null)

async function fetchData() {
  try {
    const params: Record<string, any> = { limit: 100 }
    if (filters.value.camera_id) params.camera_id = filters.value.camera_id
    if (filters.value.severity) params.severity = filters.value.severity
    const [a, c] = await Promise.all([getAlerts(params), getCameras()])
    alerts.value = a.data
    cameras.value = c.data
  } finally {
    loading.value = false
  }
}

useWebSocket({
  topics: ['alerts'],
  onMessage(topic, data) {
    if (topic === 'alerts') alerts.value = data
  },
  fallbackPoll: fetchData,
  fallbackInterval: 15000,
})

onMounted(fetchData)

function showDetail(record: any) {
  selectedAlert.value = record
  detailVisible.value = true
}

async function handleAcknowledge(id: string) {
  await acknowledgeAlert(id)
  message.success('已确认')
  fetchData()
  detailVisible.value = false
}

async function handleFalsePositive(id: string) {
  await markFalsePositive(id)
  message.success('已标记误报')
  fetchData()
  detailVisible.value = false
}

const severityColor: Record<string, string> = {
  high: 'red', medium: 'orange', low: 'gold', info: 'blue',
}
const severityLabel: Record<string, string> = {
  high: '高', medium: '中', low: '低', info: '提示',
}

const columns = [
  {
    title: '严重度',
    key: 'severity',
    width: 100,
    filters: [
      { text: '高', value: 'high' },
      { text: '中', value: 'medium' },
      { text: '低', value: 'low' },
      { text: '提示', value: 'info' },
    ],
  },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id', width: 120 },
  { title: '区域', dataIndex: 'zone_id', key: 'zone_id', width: 100 },
  {
    title: '异常分数',
    key: 'score',
    width: 100,
  },
  {
    title: '状态',
    key: 'status',
    width: 120,
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
      <Typography.Title :level="3" style="margin: 0">告警中心</Typography.Title>
      <Space>
        <Select
          v-model:value="filters.camera_id"
          placeholder="全部摄像头"
          allow-clear
          style="width: 160px"
          @change="fetchData"
        >
          <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
            {{ cam.camera_id }}
          </Select.Option>
        </Select>
        <Select
          v-model:value="filters.severity"
          placeholder="全部严重度"
          allow-clear
          style="width: 120px"
          @change="fetchData"
        >
          <Select.Option value="high">高</Select.Option>
          <Select.Option value="medium">中</Select.Option>
          <Select.Option value="low">低</Select.Option>
          <Select.Option value="info">提示</Select.Option>
        </Select>
      </Space>
    </div>

    <Table
      :columns="columns"
      :data-source="alerts"
      :loading="loading"
      row-key="alert_id"
      :pagination="{ pageSize: 20, showTotal: (total: number) => `共 ${total} 条` }"
      @row:click="(record: any) => showDetail(record)"
      style="cursor: pointer"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'severity'">
          <Tag :color="severityColor[record.severity]">{{ severityLabel[record.severity] || record.severity }}</Tag>
        </template>
        <template v-if="column.key === 'score'">
          {{ record.anomaly_score?.toFixed(3) }}
        </template>
        <template v-if="column.key === 'status'">
          <Tag v-if="record.acknowledged" color="green">已确认</Tag>
          <Tag v-else-if="record.false_positive" color="orange">误报</Tag>
          <Tag v-else color="default">待处理</Tag>
        </template>
        <template v-if="column.key === 'action'">
          <Space @click.stop>
            <Button
              v-if="!record.acknowledged && !record.false_positive"
              type="primary"
              size="small"
              @click="handleAcknowledge(record.alert_id)"
            >确认</Button>
            <Button
              v-if="!record.false_positive"
              size="small"
              @click="handleFalsePositive(record.alert_id)"
            >误报</Button>
          </Space>
        </template>
      </template>
    </Table>

    <!-- Alert Detail Drawer -->
    <Drawer
      v-model:open="detailVisible"
      :title="`告警详情 — ${selectedAlert?.alert_id?.slice(0, 24)}`"
      :width="selectedAlert?.has_recording ? 900 : 640"
      placement="right"
    >
      <template v-if="selectedAlert">
        <!-- Replay Player (when recording exists) -->
        <ReplayPlayer
          v-if="selectedAlert.has_recording"
          :alert-id="selectedAlert.alert_id"
          style="margin-bottom: 16px"
        />

        <!-- Static snapshot fallback -->
        <div v-if="selectedAlert.snapshot_path && !selectedAlert.has_recording" style="margin-bottom: 24px; text-align: center">
          <Image
            :src="`/api/alerts/${selectedAlert.alert_id}/image/composite`"
            :fallback="`/api/alerts/${selectedAlert.alert_id}/image/snapshot`"
            style="max-width: 100%; border-radius: 8px"
          />
        </div>
        <Descriptions :column="1" bordered size="small">
          <Descriptions.Item label="告警 ID">
            <span style="font-family: monospace; font-size: 12px">{{ selectedAlert.alert_id }}</span>
          </Descriptions.Item>
          <Descriptions.Item label="摄像头">{{ selectedAlert.camera_id }}</Descriptions.Item>
          <Descriptions.Item label="区域">{{ selectedAlert.zone_id }}</Descriptions.Item>
          <Descriptions.Item label="严重度">
            <Tag :color="severityColor[selectedAlert.severity]">
              {{ severityLabel[selectedAlert.severity] }}
            </Tag>
          </Descriptions.Item>
          <Descriptions.Item label="异常分数">{{ selectedAlert.anomaly_score?.toFixed(4) }}</Descriptions.Item>
          <Descriptions.Item label="状态">
            <Tag v-if="selectedAlert.acknowledged" color="green">已确认</Tag>
            <Tag v-else-if="selectedAlert.false_positive" color="orange">误报</Tag>
            <Tag v-else color="default">待处理</Tag>
          </Descriptions.Item>
          <Descriptions.Item label="备注" v-if="selectedAlert.notes">{{ selectedAlert.notes }}</Descriptions.Item>
        </Descriptions>
        <Divider />
        <Space>
          <Button
            v-if="!selectedAlert.acknowledged"
            type="primary"
            @click="handleAcknowledge(selectedAlert.alert_id)"
          >确认真实</Button>
          <Button
            v-if="!selectedAlert.false_positive"
            @click="handleFalsePositive(selectedAlert.alert_id)"
          >标记误报</Button>
        </Space>
      </template>
    </Drawer>
  </div>
</template>
