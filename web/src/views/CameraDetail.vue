<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Card, Steps, Tabs, Descriptions, Badge, Button, Typography, Space, Statistic, Row, Col } from 'ant-design-vue'
import { ArrowLeftOutlined } from '@ant-design/icons-vue'
import { getCameras } from '../api'

const route = useRoute()
const router = useRouter()
const cameraId = route.params.id as string
const camera = ref<any>(null)
const loading = ref(true)

async function fetchCamera() {
  try {
    const res = await getCameras()
    camera.value = res.data.find((c: any) => c.camera_id === cameraId) || null
  } finally {
    loading.value = false
  }
}

onMounted(fetchCamera)

// const snapshotUrl = computed(() => `/api/cameras/${cameraId}/snapshot?t=${Date.now()}`)
const streamUrl = computed(() => `/api/cameras/${cameraId}/stream`)

const lifecycleStep = computed(() => {
  if (!camera.value) return 0
  const c = camera.value
  if (!c.connected) return 0
  if (!c.stats || c.stats.frames_captured === 0) return 0
  if (c.stats.frames_analyzed === 0) return 2
  return 4
})
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
          <div v-if="camera?.connected" style="text-align: center">
            <img
              :src="streamUrl"
              style="max-width: 100%; border-radius: 8px; background: #000"
              alt="实时画面"
            />
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
        <Card>
          <Descriptions :column="2" bordered size="small">
            <Descriptions.Item label="摄像头 ID">{{ cameraId }}</Descriptions.Item>
            <Descriptions.Item label="名称">{{ camera?.name }}</Descriptions.Item>
            <Descriptions.Item label="状态">
              <Badge :status="camera?.connected ? 'success' : 'default'" />
              {{ camera?.connected ? '在线' : '离线' }}
            </Descriptions.Item>
            <Descriptions.Item label="运行中">{{ camera?.running ? '是' : '否' }}</Descriptions.Item>
          </Descriptions>
        </Card>
      </Tabs.TabPane>
    </Tabs>
  </div>
</template>
