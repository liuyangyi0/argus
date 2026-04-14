<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { Card, Button, Tag, Typography, Space, Modal, Badge, message } from 'ant-design-vue'
import { getAlerts, updateAlertWorkflow, getHealth, getTasks } from '../api'
import { formatRelativeTime } from '../utils/time'
import { useWebSocket } from '../composables/useWebSocket'

const props = defineProps<{
  health?: any
}>()

const router = useRouter()
const alerts = ref<any[]>([])
const internalHealth = ref<any>(null)
const tasks = ref<any[]>([])
const highFlash = ref(false)

const healthData = computed(() => props.health ?? internalHealth.value)

async function fetchData() {
  try {
    const fetches: Promise<any>[] = [
      getAlerts({ limit: 20 }),
      getTasks(),
    ]
    if (!props.health) fetches.push(getHealth())

    const results = await Promise.all(fetches)
    alerts.value = results[0].alerts
    tasks.value = (results[1].data || results[1])?.tasks || []
    if (!props.health && results[2]) internalHealth.value = results[2]
  } catch { /* silent */ }
}

useWebSocket({
  topics: ['alerts', 'health', 'tasks'],
  onMessage(topic, data) {
    if (topic === 'alerts') alerts.value = data
    else if (topic === 'health' && !props.health) internalHealth.value = data
    else if (topic === 'tasks') tasks.value = data
  },
  fallbackPoll: fetchData,
  fallbackInterval: 15000,
})

onMounted(fetchData)

const pendingAlerts = computed(() =>
  alerts.value.filter((a: any) => a.workflow_status === 'new' || a.workflow_status === 'acknowledged')
)

const highAlerts = computed(() => pendingAlerts.value.filter((a: any) => a.severity === 'high'))
const mediumAlerts = computed(() => pendingAlerts.value.filter((a: any) => a.severity === 'medium'))
const lowAlerts = computed(() => pendingAlerts.value.filter((a: any) =>
  a.severity === 'low' || a.severity === 'info'
))

// Flash HIGH card when new high alert arrives
watch(() => highAlerts.value.length, (newLen, oldLen) => {
  if (newLen > (oldLen ?? 0)) {
    highFlash.value = true
    setTimeout(() => { highFlash.value = false }, 1500)
  }
})

const connectedCount = computed(() =>
  healthData.value?.cameras?.filter((c: any) => c.connected).length || 0
)
const totalCameras = computed(() => healthData.value?.cameras?.length || 0)

const severityColor: Record<string, string> = {
  high: '#ef4444', medium: '#f97316', low: '#f59e0b', info: '#3b82f6',
}
const severityLabel: Record<string, string> = {
  high: '高', medium: '中', low: '低', info: '提示',
}

// Short relative time: formatRelativeTime(ts, true) → "5s", "3m", "2h"

async function quickAction(alertId: string, severity: string, action: string) {
  if (severity === 'high') {
    router.push(`/alerts?id=${alertId}`)
    return
  }
  if (severity === 'medium') {
    Modal.confirm({
      title: '确认操作',
      content: `确定要将此 MEDIUM 级告警标记为"${action === 'acknowledged' ? '已确认' : '误报'}"吗？`,
      okText: '确定',
      cancelText: '取消',
      async onOk() {
        await updateAlertWorkflow(alertId, { status: action, confirmed: true })
        message.success('已处理')
        fetchData()
      },
    })
    return
  }
  // LOW / INFO: quick action
  await updateAlertWorkflow(alertId, { status: action })
  message.success('已处理')
  fetchData()
}
</script>

<template>
  <div style="width: 280px; flex-shrink: 0; display: flex; flex-direction: column; gap: 10px; overflow-y: auto; padding: 0 0 0 12px">
    <!-- Pending alerts header -->
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0 4px">
      <Typography.Text strong style="font-size: 13px">待处理告警</Typography.Text>
      <Badge
        :count="pendingAlerts.length"
        :number-style="{ backgroundColor: pendingAlerts.length > 0 ? '#3b82f6' : '#4a5568', fontSize: '11px', minWidth: '18px', height: '18px', lineHeight: '18px' }"
        :show-zero="true"
      />
    </div>

    <!-- System status -->
    <Card size="small" style="background: var(--bg)">
      <div style="display: flex; justify-content: space-between; align-items: center">
        <Typography.Text type="secondary">摄像头</Typography.Text>
        <Badge :status="connectedCount === totalCameras && totalCameras > 0 ? 'success' : 'warning'" />
      </div>
      <Typography.Title :level="4" style="margin: 4px 0 0">
        {{ connectedCount }}/{{ totalCameras }} 在线
      </Typography.Title>
    </Card>

    <!-- HIGH alerts -->
    <Card
      v-if="highAlerts.length > 0"
      size="small"
      title="高级告警"
      :style="{
        background: 'var(--bg)',
        borderColor: highFlash ? '#ef4444' : '#ef444488',
        boxShadow: highFlash ? '0 0 12px rgba(239,68,68,0.4)' : 'none',
        transition: 'border-color 0.3s, box-shadow 0.3s',
      }"
    >
      <div v-for="a in highAlerts.slice(0, 5)" :key="a.alert_id" style="margin-bottom: 10px; padding: 6px; border-radius: 4px; background: rgba(239,68,68,0.1)">
        <!-- Thumbnail -->
        <div v-if="a.snapshot_path" style="margin-bottom: 6px; border-radius: 3px; overflow: hidden; background: #000">
          <img
            :src="`/api/alerts/${a.alert_id}/image/snapshot`"
            style="width: 100%; height: 48px; object-fit: cover; display: block"
            loading="lazy"
          />
        </div>
        <!-- Info row -->
        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px">
          <Tag color="red" style="margin: 0; font-size: 10px; padding: 0 4px; line-height: 18px">高</Tag>
          <Typography.Text style="font-size: 12px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">
            {{ a.camera_id }}
          </Typography.Text>
          <Typography.Text type="secondary" style="font-size: 10px; flex-shrink: 0">
            {{ formatRelativeTime(a.timestamp || a.created_at, true) }}
          </Typography.Text>
          <Typography.Text :style="{ fontSize: '11px', color: '#ef4444', fontWeight: 600, flexShrink: 0 }">
            {{ (a.anomaly_score || 0).toFixed(2) }}
          </Typography.Text>
        </div>
        <Button type="primary" size="small" block danger @click="quickAction(a.alert_id, 'high', '')">
          查看详情
        </Button>
      </div>
    </Card>

    <!-- MEDIUM alerts -->
    <Card v-if="mediumAlerts.length > 0" size="small" title="中级告警" style="background: var(--bg)">
      <div v-for="a in mediumAlerts.slice(0, 5)" :key="a.alert_id" style="margin-bottom: 8px; padding: 6px; border-radius: 4px; background: rgba(249,115,22,0.06)">
        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px">
          <Tag color="orange" style="margin: 0; font-size: 10px; padding: 0 4px; line-height: 18px">中</Tag>
          <Typography.Text style="font-size: 12px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">
            {{ a.camera_id }}
          </Typography.Text>
          <Typography.Text type="secondary" style="font-size: 10px; flex-shrink: 0">
            {{ formatRelativeTime(a.timestamp || a.created_at, true) }}
          </Typography.Text>
          <Typography.Text :style="{ fontSize: '11px', color: '#f97316', fontWeight: 600, flexShrink: 0 }">
            {{ (a.anomaly_score || 0).toFixed(2) }}
          </Typography.Text>
        </div>
        <Space size="small">
          <Button size="small" @click="quickAction(a.alert_id, 'medium', 'acknowledged')">确认</Button>
          <Button size="small" @click="quickAction(a.alert_id, 'medium', 'false_positive')">误报</Button>
        </Space>
      </div>
    </Card>

    <!-- LOW / INFO alerts -->
    <Card v-if="lowAlerts.length > 0" size="small" title="低级/提示" style="background: var(--bg)">
      <div v-for="a in lowAlerts.slice(0, 8)" :key="a.alert_id" style="margin-bottom: 6px; display: flex; align-items: center; gap: 6px">
        <Tag :color="severityColor[a.severity]" style="margin: 0; min-width: 28px; text-align: center; font-size: 10px; padding: 0 4px; line-height: 18px">
          {{ severityLabel[a.severity] }}
        </Tag>
        <Typography.Text style="font-size: 12px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">
          {{ a.camera_id }}
        </Typography.Text>
        <Typography.Text type="secondary" style="font-size: 10px; flex-shrink: 0">
          {{ formatRelativeTime(a.timestamp || a.created_at, true) }}
        </Typography.Text>
        <Space size="small" style="flex-shrink: 0">
          <Button size="small" type="text" style="font-size: 11px; padding: 0 4px" @click="quickAction(a.alert_id, a.severity, 'acknowledged')">V</Button>
          <Button size="small" type="text" style="font-size: 11px; padding: 0 4px" @click="quickAction(a.alert_id, a.severity, 'false_positive')">X</Button>
        </Space>
      </div>
    </Card>

    <!-- Pending tasks -->
    <Card v-if="tasks.length > 0" size="small" title="待办任务" style="background: var(--bg)">
      <div v-for="t in tasks.slice(0, 5)" :key="t.id || t.task_id" style="margin-bottom: 4px">
        <Typography.Text style="font-size: 12px">{{ t.description || t.name || t.task_id }}</Typography.Text>
      </div>
    </Card>

    <!-- Empty state -->
    <div v-if="pendingAlerts.length === 0" style="text-align: center; padding: 24px 0; color: #4a5568">
      <Typography.Text type="secondary">暂无待处理告警</Typography.Text>
    </div>
  </div>
</template>
