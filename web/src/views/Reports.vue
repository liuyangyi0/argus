<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Card, Statistic, Row, Col, Select, Spin, Typography, Empty, Button, Radio, message } from 'ant-design-vue'
import { DownloadOutlined } from '@ant-design/icons-vue'
import VChart from 'vue-echarts'
import { getReportStats, getDailyTrend, getSeverityDist, getCameraDist, getFPTrend, downloadComplianceReport } from '../api/reports'
import { SEVERITY_COLORS } from '../utils/colors'

const loading = ref(true)
const days = ref(30)
const stats = ref<any>(null)
const trendData = ref<any>(null)
const severityData = ref<any>(null)
const cameraData = ref<any>(null)
const fpData = ref<any>(null)

// Compliance report controls
const complianceDays = ref(30)
const complianceFormat = ref<'csv' | 'pdf'>('csv')
const complianceLoading = ref(false)

async function fetchAll() {
  loading.value = true
  try {
    const [s, t, sev, cam, fp] = await Promise.all([
      getReportStats(),
      getDailyTrend(days.value),
      getSeverityDist(),
      getCameraDist(),
      getFPTrend(days.value),
    ])
    stats.value = s
    trendData.value = t
    severityData.value = sev
    cameraData.value = cam
    fpData.value = fp
  } catch { /* interceptor handles */ }
  finally { loading.value = false }
}

async function handleDownloadCompliance() {
  complianceLoading.value = true
  try {
    await downloadComplianceReport(complianceDays.value, complianceFormat.value)
    message.success('报告下载已开始')
  } catch (e: any) {
    message.error(e.message || '生成合规报告失败')
  } finally {
    complianceLoading.value = false
  }
}

function handleDaysChange() { fetchAll() }
onMounted(fetchAll)

// ── ECharts options ──

const trendOption = computed(() => {
  if (!trendData.value) return {}
  const d = trendData.value
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { data: ['高', '中', '低', '提示'], textStyle: { color: 'var(--ink-3)' } },
    grid: { left: 40, right: 16, top: 40, bottom: 24 },
    xAxis: { type: 'category' as const, data: d.labels, axisLabel: { color: 'var(--ink-4)' } },
    yAxis: { type: 'value' as const, axisLabel: { color: 'var(--ink-4)' }, minInterval: 1 },
    series: [
      { name: '高', type: 'bar' as const, stack: 'total', data: d.high, color: SEVERITY_COLORS.high },
      { name: '中', type: 'bar' as const, stack: 'total', data: d.medium, color: SEVERITY_COLORS.medium },
      { name: '低', type: 'bar' as const, stack: 'total', data: d.low, color: SEVERITY_COLORS.low },
      { name: '提示', type: 'bar' as const, stack: 'total', data: d.info, color: SEVERITY_COLORS.info },
    ],
  }
})

const severityOption = computed(() => {
  if (!severityData.value) return {}
  const d = severityData.value
  return {
    tooltip: { trigger: 'item' as const },
    legend: { bottom: 0, textStyle: { color: 'var(--ink-3)' } },
    series: [{
      type: 'pie' as const,
      radius: ['40%', '70%'],
      label: { color: 'var(--ink-3)' },
      data: [
        { value: d.high, name: '高', itemStyle: { color: SEVERITY_COLORS.high } },
        { value: d.medium, name: '中', itemStyle: { color: SEVERITY_COLORS.medium } },
        { value: d.low, name: '低', itemStyle: { color: SEVERITY_COLORS.low } },
        { value: d.info, name: '提示', itemStyle: { color: SEVERITY_COLORS.info } },
      ],
    }],
  }
})

const cameraOption = computed(() => {
  if (!cameraData.value?.cameras?.length) return {}
  const cams = cameraData.value.cameras
  return {
    tooltip: { trigger: 'axis' as const },
    grid: { left: 100, right: 24, top: 8, bottom: 24 },
    xAxis: { type: 'value' as const, axisLabel: { color: 'var(--ink-4)' }, minInterval: 1 },
    yAxis: { type: 'category' as const, data: cams.map((c: any) => c.camera_id), axisLabel: { color: 'var(--ink-4)' } },
    series: [{ type: 'bar' as const, data: cams.map((c: any) => c.count), color: '#4fc3f7' }],
  }
})

const fpOption = computed(() => {
  if (!fpData.value) return {}
  const d = fpData.value
  return {
    tooltip: { trigger: 'axis' as const, formatter: '{b}: {c}%' },
    grid: { left: 40, right: 16, top: 16, bottom: 24 },
    xAxis: { type: 'category' as const, data: d.labels, axisLabel: { color: 'var(--ink-4)' } },
    yAxis: { type: 'value' as const, max: 100, axisLabel: { color: 'var(--ink-4)', formatter: '{value}%' } },
    series: [{
      type: 'line' as const,
      data: d.rates,
      smooth: true,
      areaStyle: { color: 'rgba(217,119,6,0.1)' },
      lineStyle: { color: SEVERITY_COLORS.medium },
      itemStyle: { color: SEVERITY_COLORS.medium },
    }],
  }
})
</script>

<template>
  <main class="glass" style="margin: 12px; padding: 24px; border-radius: var(--r-lg); min-width: 0; display: flex; flex-direction: column; flex: 1;">
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;">
      <Typography.Title :level="3" style="margin: 0; color: var(--ink)">报表统计</Typography.Title>
      <Select v-model:value="days" style="width: 120px" @change="handleDaysChange">
        <Select.Option :value="7">最近 7 天</Select.Option>
        <Select.Option :value="14">最近 14 天</Select.Option>
        <Select.Option :value="30">最近 30 天</Select.Option>
        <Select.Option :value="60">最近 60 天</Select.Option>
        <Select.Option :value="90">最近 90 天</Select.Option>
      </Select>
    </div>

    <!-- Compliance report download -->
    <Card size="small" style="margin-bottom: 24px;">
      <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
        <Typography.Text strong style="color: var(--ink); white-space: nowrap;">生成合规报告</Typography.Text>
        <Select v-model:value="complianceDays" style="width: 120px" size="small">
          <Select.Option :value="7">最近 7 天</Select.Option>
          <Select.Option :value="14">最近 14 天</Select.Option>
          <Select.Option :value="30">最近 30 天</Select.Option>
          <Select.Option :value="60">最近 60 天</Select.Option>
          <Select.Option :value="90">最近 90 天</Select.Option>
        </Select>
        <Radio.Group v-model:value="complianceFormat" size="small">
          <Radio.Button value="csv">CSV</Radio.Button>
          <Radio.Button value="pdf">PDF</Radio.Button>
        </Radio.Group>
        <Button type="primary" size="small" :loading="complianceLoading" @click="handleDownloadCompliance">
          <template #icon><DownloadOutlined /></template>
          下载报告
        </Button>
      </div>
    </Card>

    <Spin :spinning="loading">
      <!-- Summary stats -->
      <Row :gutter="16" style="margin-bottom: 24px" v-if="stats">
        <Col :span="6"><Card size="small"><Statistic title="告警总数" :value="stats.total_alerts" /></Card></Col>
        <Col :span="6"><Card size="small"><Statistic title="高严重度" :value="stats.by_severity?.high ?? 0" :value-style="{ color: SEVERITY_COLORS.high }" /></Card></Col>
        <Col :span="6"><Card size="small"><Statistic title="误报率" :value="stats.false_positive_rate" suffix="%" :value-style="{ color: '#d97706' }" /></Card></Col>
        <Col :span="6"><Card size="small"><Statistic title="确认率" :value="stats.acknowledged_rate" suffix="%" :value-style="{ color: '#15a34a' }" /></Card></Col>
      </Row>

      <!-- Daily trend -->
      <Card title="每日告警趋势" size="small" style="margin-bottom: 16px">
        <VChart v-if="trendData" :option="trendOption" :autoresize="true" style="height: 280px" />
        <Empty v-else description="暂无数据" />
      </Card>

      <!-- Severity + Camera distribution side by side -->
      <Row :gutter="16" style="margin-bottom: 16px">
        <Col :span="12">
          <Card title="按严重度分布" size="small">
            <VChart v-if="severityData" :option="severityOption" :autoresize="true" style="height: 260px" />
            <Empty v-else description="暂无数据" />
          </Card>
        </Col>
        <Col :span="12">
          <Card title="按摄像头分布" size="small">
            <VChart v-if="cameraData?.cameras?.length" :option="cameraOption" :autoresize="true" style="height: 260px" />
            <Empty v-else description="暂无数据" />
          </Card>
        </Col>
      </Row>

      <!-- FP trend -->
      <Card title="误报率趋势" size="small">
        <VChart v-if="fpData" :option="fpOption" :autoresize="true" style="height: 220px" />
        <Empty v-else description="暂无数据" />
      </Card>
    </Spin>
  </main>
</template>
