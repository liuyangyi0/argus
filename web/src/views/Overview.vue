<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { Row, Col, Card, Statistic, Alert, Tag, List, Badge, Typography } from 'ant-design-vue'
import { getHealth, getCameras, getAlerts } from '../api'
import { useWebSocket } from '../composables/useWebSocket'

const router = useRouter()
const health = ref<any>(null)
const cameras = ref<any[]>([])
const alerts = ref<any[]>([])
const loading = ref(true)

async function fetchData() {
  try {
    const [h, c, a] = await Promise.all([
      getHealth(),
      getCameras(),
      getAlerts({ limit: 10 }),
    ])
    health.value = h.data
    cameras.value = c.data
    alerts.value = a.data
  } catch (e) {
    console.error('Overview fetch error', e)
  } finally {
    loading.value = false
  }
}

const { } = useWebSocket({
  topics: ['health', 'cameras', 'alerts'],
  onMessage(topic, data) {
    if (topic === 'health') health.value = data
    else if (topic === 'cameras') cameras.value = data
    else if (topic === 'alerts') alerts.value = data
  },
  fallbackPoll: fetchData,
  fallbackInterval: 15000,
})

onMounted(fetchData)

const connected = () => cameras.value.filter((c: any) => c.connected).length
const severityColor: Record<string, string> = {
  high: '#ef4444', medium: '#f97316', low: '#f59e0b', info: '#3b82f6',
}
</script>

<template>
  <div>
    <Typography.Title :level="3" style="margin-bottom: 24px">系统总览</Typography.Title>

    <!-- Health Banner -->
    <Alert
      v-if="health"
      :type="health.status === 'healthy' ? 'success' : health.status === 'degraded' ? 'warning' : 'error'"
      :message="health.status === 'healthy' ? '系统运行正常 — 所有摄像头已连接' : health.status === 'degraded' ? '系统降级运行' : '系统异常'"
      show-icon
      style="margin-bottom: 24px"
    />

    <!-- Stats -->
    <Row :gutter="16" style="margin-bottom: 24px">
      <Col :span="6">
        <Card>
          <Statistic title="摄像头在线" :value="`${connected()}/${cameras.length}`" :value-style="{ color: '#3b82f6' }" />
        </Card>
      </Col>
      <Col :span="6">
        <Card>
          <Statistic title="累计告警" :value="health?.total_alerts || 0" :value-style="{ color: '#10b981' }" />
        </Card>
      </Col>
      <Col :span="6">
        <Card>
          <Statistic title="运行时间" :value="health ? `${Math.floor((health.uptime_seconds || 0) / 60)}分钟` : '—'" />
        </Card>
      </Col>
      <Col :span="6">
        <Card>
          <Statistic title="Python" :value="health?.python_version || '—'" />
        </Card>
      </Col>
    </Row>

    <!-- Camera Grid -->
    <Card title="摄像头概览" style="margin-bottom: 24px">
      <Row :gutter="16">
        <Col v-for="cam in cameras" :key="cam.camera_id" :span="8">
          <Card
            hoverable
            size="small"
            style="margin-bottom: 12px"
            @click="router.push(`/cameras/${cam.camera_id}`)"
          >
            <template #title>
              <Badge :status="cam.connected ? 'success' : 'default'" />
              {{ cam.camera_id }} — {{ cam.name }}
            </template>
            <div style="color: #888; font-size: 13px">
              <span v-if="cam.stats">
                帧数: {{ cam.stats.frames_captured }} |
                延迟: {{ cam.stats.avg_latency_ms?.toFixed(1) }}ms |
                告警: {{ cam.stats.alerts_emitted }}
              </span>
              <span v-else>{{ cam.connected ? '在线' : '离线' }}</span>
            </div>
          </Card>
        </Col>
      </Row>
      <div v-if="cameras.length === 0" style="text-align: center; color: #666; padding: 32px">
        暂无摄像头
      </div>
    </Card>

    <!-- Recent Alerts -->
    <Card title="最近告警">
      <List :data-source="alerts" :loading="loading" size="small">
        <template #renderItem="{ item }">
          <List.Item>
            <List.Item.Meta>
              <template #title>
                <Tag :color="severityColor[item.severity] || '#666'">{{ item.severity?.toUpperCase() }}</Tag>
                {{ item.camera_id }} · {{ item.zone_id }}
              </template>
              <template #description>
                分数: {{ item.anomaly_score?.toFixed(3) }}
              </template>
            </List.Item.Meta>
          </List.Item>
        </template>
        <template #header v-if="alerts.length === 0 && !loading">
          <div style="text-align: center; color: #10b981; padding: 16px">
            ✓ 暂无告警 — 一切正常
          </div>
        </template>
      </List>
    </Card>
  </div>
</template>
