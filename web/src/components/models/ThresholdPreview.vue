<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { Card, Slider, Select, Statistic, Row, Col, Space, Typography, Spin } from 'ant-design-vue'
import VChart from 'vue-echarts'
import { getThresholdPreview } from '../../api/models'

const props = defineProps<{ cameras: Array<{ camera_id: string; name?: string }> }>()

const selectedCamera = ref<string | undefined>(undefined)
const threshold = ref(0.5)
const days = ref(7)
const loading = ref(false)
const previewData = ref<any>(null)

async function fetchPreview() {
  loading.value = true
  try {
    previewData.value = await getThresholdPreview({
      camera_id: selectedCamera.value,
      threshold: threshold.value,
      days: days.value,
    })
  } catch {
    previewData.value = null
  } finally {
    loading.value = false
  }
}

// Debounce threshold changes
let timer: ReturnType<typeof setTimeout> | null = null
function onThresholdChange(val: number) {
  threshold.value = val
  if (timer) clearTimeout(timer)
  timer = setTimeout(fetchPreview, 300)
}

watch([selectedCamera, days], fetchPreview)
onMounted(fetchPreview)

const chartOption = ref({})
watch(previewData, (data) => {
  if (!data?.histogram) {
    chartOption.value = {}
    return
  }
  const bins = data.histogram.map((h: any) => h.bin_start.toFixed(2))
  const counts = data.histogram.map((h: any) => h.count)
  const thresholdIdx = Math.floor(threshold.value * 50)

  chartOption.value = {
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: bins,
      name: '异常分数',
      axisLabel: { interval: 9, fontSize: 10 },
    },
    yAxis: { type: 'value', name: '帧数' },
    series: [{
      type: 'bar',
      data: counts.map((c: number, i: number) => ({
        value: c,
        itemStyle: { color: i >= thresholdIdx ? '#ff4d4f' : '#1890ff' },
      })),
    }],
    markLine: {
      data: [{ xAxis: thresholdIdx }],
    },
    grid: { left: 50, right: 20, top: 30, bottom: 40 },
  }
}, { deep: true })
</script>

<template>
  <div>
    <Card title="阈值调节预览" size="small">
      <template #extra>
        <Space>
          <Select
            v-model:value="selectedCamera"
            placeholder="全部摄像头"
            allow-clear
            size="small"
            style="width: 160px"
          >
            <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
              {{ cam.name || cam.camera_id }}
            </Select.Option>
          </Select>
          <Select v-model:value="days" size="small" style="width: 100px">
            <Select.Option :value="1">1 天</Select.Option>
            <Select.Option :value="7">7 天</Select.Option>
            <Select.Option :value="30">30 天</Select.Option>
          </Select>
        </Space>
      </template>

      <div style="margin-bottom: 16px">
        <Typography.Text>阈值: <strong>{{ threshold.toFixed(2) }}</strong></Typography.Text>
        <Slider
          :value="threshold"
          :min="0"
          :max="1"
          :step="0.01"
          :tooltip-visible="true"
          @change="onThresholdChange"
          :marks="{ 0.3: '0.3', 0.5: '0.5', 0.7: '0.7', 0.9: '0.9' }"
        />
      </div>

      <Spin :spinning="loading">
        <Row :gutter="16" v-if="previewData" style="margin-bottom: 16px">
          <Col :span="6">
            <Statistic title="总帧数" :value="previewData.total_frames" />
          </Col>
          <Col :span="6">
            <Statistic
              title="超过阈值"
              :value="previewData.above_threshold"
              value-style="color: #ff4d4f"
            />
          </Col>
          <Col :span="6">
            <Statistic
              title="告警率"
              :value="(previewData.alert_rate * 100).toFixed(3) + '%'"
            />
          </Col>
          <Col :span="6">
            <Statistic
              title="P95 分数"
              :value="previewData.percentiles?.p95?.toFixed(4)"
            />
          </Col>
        </Row>

        <VChart
          v-if="previewData?.histogram"
          :option="chartOption"
          style="height: 240px; width: 100%"
          autoresize
        />

        <div v-if="previewData && !previewData.total_frames" style="text-align: center; padding: 40px; color: #999">
          暂无推理记录 — 系统运行后将显示分数分布
        </div>
      </Spin>
    </Card>
  </div>
</template>
