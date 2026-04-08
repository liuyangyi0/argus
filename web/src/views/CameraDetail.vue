<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Card, Steps, Tabs, Descriptions, Badge, Button, Typography, Space, Statistic, Row, Col, Empty } from 'ant-design-vue'
import { ArrowLeftOutlined } from '@ant-design/icons-vue'
import { getCameraDetail } from '../api'
import { useGo2RTC } from '../composables/useGo2RTC'

const route = useRoute()
const router = useRouter()
const cameraId = route.params.id as string
const camera = ref<any>(null)
const loading = ref(true)
let pollTimer: ReturnType<typeof setInterval> | null = null

async function fetchCamera() {
  try {
    camera.value = await getCameraDetail(cameraId) || null
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchCamera()
  pollTimer = setInterval(fetchCamera, 5000)
  startStream()
})

onUnmounted(() => {
  if (pollTimer !== null) {
    clearInterval(pollTimer)
    pollTimer = null
  }
  stopStream()
})

// go2rtc WebRTC/MSE player
const { videoRef, status: streamStatus, start: startStream, stop: stopStream } = useGo2RTC(cameraId)
const mjpegUrl = computed(() => `/api/cameras/${cameraId}/stream`)

const lifecycleStep = computed(() => {
  if (!camera.value) return 0
  const stages = camera.value.stages
  if (stages && Array.isArray(stages)) {
    const activeIdx = stages.findIndex((s: any) => s.status === 'active')
    if (activeIdx >= 0) return activeIdx
    // All completed
    if (stages.every((s: any) => s.status === 'completed')) return stages.length - 1
    return 0
  }
  // Fallback when stages not available
  const c = camera.value
  if (!c.connected) return 0
  if (!c.stats || c.stats.frames_captured === 0) return 0
  if (c.stats.frames_analyzed === 0) return 2
  return 4
})

function formatLabel(key: string) {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function formatValue(value: any): string {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (Array.isArray(value)) return value.length ? value.map(formatValue).join(', ') : '-'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function flattenEntries(value: any, prefix = ''): Array<{ key: string; label: string; value: string }> {
  if (value === null || value === undefined) return []

  if (typeof value !== 'object' || Array.isArray(value)) {
    return [{ key: prefix, label: formatLabel(prefix || 'value'), value: formatValue(value) }]
  }

  return Object.entries(value).flatMap(([key, nested]) => {
    const nextPrefix = prefix ? `${prefix}.${key}` : key
    if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
      return flattenEntries(nested, nextPrefix)
    }
    return [{ key: nextPrefix, label: formatLabel(nextPrefix), value: formatValue(nested) }]
  })
}

const basicEntries = computed(() => flattenEntries({
  camera_id: camera.value?.camera_id,
  name: camera.value?.name,
  connected: camera.value?.connected,
  running: camera.value?.running,
}))

const runtimeEntries = computed(() => flattenEntries({
  stats: camera.value?.stats,
  runtime: camera.value?.runtime,
  runner: camera.value?.runner,
}))

const healthEntries = computed(() => flattenEntries(camera.value?.health))
const detectorEntries = computed(() => flattenEntries(camera.value?.detector))
const configEntries = computed(() => flattenEntries(camera.value?.config))
</script>

<template>
  <div>
    <!-- Header -->
    <Space style="margin-bottom: 24px" align="center">
      <Button @click="router.push('/cameras')">
        <ArrowLeftOutlined /> 返回
      </Button>
      <Typography.Title :level="3" style="margin: 0">
        <Badge :status="camera?.connected ? 'success' : 'default'" />
        {{ cameraId }}
        <span style="color: #888; font-weight: normal; font-size: 16px; margin-left: 8px">
          {{ camera?.name || '' }}
        </span>
      </Typography.Title>
    </Space>

    <!-- Pipeline Stepper -->
    <Card style="margin-bottom: 24px">
      <Steps :current="lifecycleStep" size="small">
        <Steps.Step title="采集" description="基线帧" />
        <Steps.Step title="基线审查" description="质量验证" />
        <Steps.Step title="训练" description="模型训练" />
        <Steps.Step title="发布" description="部署上线" />
        <Steps.Step title="推理" description="运行监控" />
      </Steps>
    </Card>

    <!-- Tabs -->
    <Tabs>
      <Tabs.TabPane key="live" tab="实时画面">
        <Card>
          <div v-if="camera?.connected" style="text-align: center; position: relative">
            <!-- WebRTC / MSE via go2rtc — always in DOM so videoRef is never null -->
            <video
              ref="videoRef"
              autoplay
              muted
              playsinline
              v-show="streamStatus === 'playing' || streamStatus === 'connecting'"
              style="max-width: 100%; border-radius: 8px; background: #000"
            />
            <!-- MJPEG fallback -->
            <img
              v-if="streamStatus === 'fallback'"
              :src="mjpegUrl"
              style="max-width: 100%; border-radius: 8px; background: #000"
              alt="实时画面"
            />
            <div v-if="streamStatus === 'connecting'" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #999; font-size: 14px">
              连接中...
            </div>
          </div>
          <div v-else style="text-align: center; padding: 64px; color: #666">
            摄像头离线
          </div>
        </Card>
        <Row :gutter="16" style="margin-top: 16px" v-if="camera?.stats">
          <Col :span="6"><Card><Statistic title="已采集帧" :value="camera.stats.frames_captured" /></Card></Col>
          <Col :span="6"><Card><Statistic title="已分析帧" :value="camera.stats.frames_analyzed" /></Card></Col>
          <Col :span="6"><Card><Statistic title="告警数" :value="camera.stats.alerts_emitted" /></Card></Col>
          <Col :span="6"><Card><Statistic title="平均延迟" :value="`${camera.stats.avg_latency_ms?.toFixed(1)}ms`" /></Card></Col>
        </Row>
      </Tabs.TabPane>

      <Tabs.TabPane key="info" tab="摄像头信息">
        <Space direction="vertical" style="width: 100%" :size="16">
          <Card title="基础信息">
            <Descriptions v-if="basicEntries.length > 0" :column="2" bordered size="small">
              <Descriptions.Item v-for="item in basicEntries" :key="item.key" :label="item.label">
                {{ item.value }}
              </Descriptions.Item>
            </Descriptions>
            <Empty v-else description="暂无基础信息" />
          </Card>

          <Card title="运行状态">
            <Descriptions v-if="runtimeEntries.length > 0" :column="2" bordered size="small">
              <Descriptions.Item v-for="item in runtimeEntries" :key="item.key" :label="item.label">
                {{ item.value }}
              </Descriptions.Item>
            </Descriptions>
            <Empty v-else description="暂无运行状态" />
          </Card>

          <Card title="健康信息">
            <Descriptions v-if="healthEntries.length > 0" :column="2" bordered size="small">
              <Descriptions.Item v-for="item in healthEntries" :key="item.key" :label="item.label">
                {{ item.value }}
              </Descriptions.Item>
            </Descriptions>
            <Empty v-else description="暂无健康信息" />
          </Card>

          <Card title="检测器参数">
            <Descriptions v-if="detectorEntries.length > 0" :column="2" bordered size="small">
              <Descriptions.Item v-for="item in detectorEntries" :key="item.key" :label="item.label">
                {{ item.value }}
              </Descriptions.Item>
            </Descriptions>
            <Empty v-else description="暂无检测器参数" />
          </Card>

          <Card title="摄像头配置">
            <Descriptions v-if="configEntries.length > 0" :column="2" bordered size="small">
              <Descriptions.Item v-for="item in configEntries" :key="item.key" :label="item.label">
                {{ item.value }}
              </Descriptions.Item>
            </Descriptions>
            <Empty v-else description="暂无摄像头配置" />
          </Card>
        </Space>
      </Tabs.TabPane>
    </Tabs>
  </div>
</template>
