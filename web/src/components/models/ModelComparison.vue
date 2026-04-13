<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import {
  Card, Select, Button, Space, Statistic, Tag, Slider, Spin,
  Row, Col, Empty, message, Segmented,
} from 'ant-design-vue'
import {
  CameraOutlined, ThunderboltOutlined, ReloadOutlined,
} from '@ant-design/icons-vue'
import { use } from 'echarts/core'
import { LineChart, BarChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import {
  GridComponent, TooltipComponent, LegendComponent,
  DataZoomComponent, MarkLineComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'
import {
  getModelRegistry, getShadowReport,
  getABScores, getABDistribution, runABLiveCompare,
} from '../../api'
import { STAGE_MAP } from '../../composables/useModelState'
import { extractErrorMessage } from '../../utils/error'
import { useThemeStore } from '../../stores/theme'

use([CanvasRenderer, LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, MarkLineComponent])

const themeStore = useThemeStore()

const props = defineProps<{
  cameras: any[]
}>()

const route = useRoute()

// ── Selection state ──
const selectedCamera = ref<string>('')

// Pre-fill from query params (linked from ModelTable "A/B 详细对比")
onMounted(() => {
  const qCamera = route.query.camera as string | undefined
  const qShadow = route.query.shadow as string | undefined
  if (qCamera) selectedCamera.value = qCamera
  // Shadow will be set after model list loads via the watch on selectedCamera,
  // but if a specific shadow is requested, override after models load.
  if (qShadow) {
    const unwatch = watch(shadowModels, (models) => {
      if (models.some((m: any) => m.model_version_id === qShadow)) {
        selectedShadow.value = qShadow
      }
      unwatch()
    })
  }
})
const shadowModels = ref<any[]>([])
const selectedShadow = ref<string>('')
const loadingModels = ref(false)

// ── Data state ──
const shadowReport = ref<any>(null)
const abScores = ref<any[]>([])
const abDistribution = ref<any>(null)
const liveResult = ref<any>(null)
const loadingReport = ref(false)
const loadingScores = ref(false)
const loadingDistribution = ref(false)
const loadingLive = ref(false)

// ── Chart config ──
const chartView = ref<string>('时序')
const daysRange = ref(7)

// ── Heatmap overlay ──
const heatmapOpacity = ref(60)

// ── Aggregated loading ──
const isAnyLoading = computed(() => loadingReport.value || loadingScores.value || loadingDistribution.value)

// ── Refresh deduplication ──
let refreshInProgress = false

// ── Auto-refresh ──
const autoRefresh = ref(false)
let refreshTimer: ReturnType<typeof setInterval> | null = null

watch(autoRefresh, (on) => {
  if (refreshTimer) { clearInterval(refreshTimer); refreshTimer = null }
  if (on && selectedShadow.value) {
    refreshTimer = setInterval(loadAllData, 30000) // 30s interval
  }
})

onUnmounted(() => {
  if (refreshTimer) clearInterval(refreshTimer)
})

// Auto-load shadow models when camera changes
watch(selectedCamera, async (camId) => {
  if (!camId) {
    shadowModels.value = []
    selectedShadow.value = ''
    return
  }
  loadingModels.value = true
  try {
    const res = await getModelRegistry(camId)
    shadowModels.value = (res.models || []).filter(
      (m: any) => m.stage === 'shadow' || m.stage === 'canary'
    )
    selectedShadow.value = shadowModels.value.length > 0 ? shadowModels.value[0].model_version_id : ''
  } catch (e: any) {
    message.error('获取模型列表失败')
  } finally {
    loadingModels.value = false
  }
})

// Auto-load data when shadow model changes
watch(selectedShadow, () => {
  if (selectedShadow.value) {
    loadAllData()
  }
})

async function loadAllData() {
  if (!selectedShadow.value) return
  if (refreshInProgress) return  // deduplicate concurrent refreshes
  refreshInProgress = true
  try {
    await Promise.allSettled([loadReport(), loadScores(), loadDistribution()])
  } finally {
    refreshInProgress = false
  }
}

async function loadReport() {
  loadingReport.value = true
  try {
    shadowReport.value = await getShadowReport(selectedShadow.value, {
      camera_id: selectedCamera.value || undefined,
      days: daysRange.value,
    })
  } catch (e) { message.error(extractErrorMessage(e, '加载影子报告失败')); shadowReport.value = null }
  finally { loadingReport.value = false }
}

async function loadScores() {
  loadingScores.value = true
  try {
    const res = await getABScores(selectedShadow.value, {
      camera_id: selectedCamera.value || undefined,
      days: daysRange.value,
      limit: 500,
    })
    abScores.value = res.scores || []
  } catch (e) { message.error(extractErrorMessage(e, '加载评分数据失败')); abScores.value = [] }
  finally { loadingScores.value = false }
}

async function loadDistribution() {
  loadingDistribution.value = true
  try {
    abDistribution.value = await getABDistribution(selectedShadow.value, {
      camera_id: selectedCamera.value || undefined,
      days: daysRange.value,
    })
  } catch (e) { message.error(extractErrorMessage(e, '加载分布数据失败')); abDistribution.value = null }
  finally { loadingDistribution.value = false }
}

async function runLiveCompare() {
  if (!selectedCamera.value || !selectedShadow.value) {
    message.warning('请选择摄像头和影子模型')
    return
  }
  loadingLive.value = true
  liveResult.value = null
  try {
    liveResult.value = await runABLiveCompare(selectedCamera.value, selectedShadow.value)
  } catch (e: any) {
    message.error(extractErrorMessage(e, '实时对比失败'))
  } finally {
    loadingLive.value = false
  }
}

// ── ECharts: Score time-series ──
const scoreChartOption = computed(() => {
  const data = abScores.value
  if (data.length === 0) return null

  const timeLabels = data.map((d: any) => {
    if (!d.t) return ''
    // Format: HH:mm or MM-DD HH:mm
    const dt = new Date(d.t)
    const h = String(dt.getHours()).padStart(2, '0')
    const m = String(dt.getMinutes()).padStart(2, '0')
    const mo = String(dt.getMonth() + 1).padStart(2, '0')
    const day = String(dt.getDate()).padStart(2, '0')
    return daysRange.value <= 1 ? `${h}:${m}` : `${mo}-${day} ${h}:${m}`
  })

  return {
    backgroundColor: 'transparent',
    grid: { left: 50, right: 20, top: 16, bottom: 60 },
    tooltip: {
      trigger: 'axis' as const,
      backgroundColor: themeStore.isDark ? '#1a1a2e' : '#ffffff',
      borderColor: themeStore.isDark ? '#2d2d4a' : '#e5e7eb',
      textStyle: { color: themeStore.isDark ? '#e2e8f0' : '#111827', fontSize: 12 },
      formatter: (params: any) => {
        if (!Array.isArray(params) || params.length === 0) return ''
        let html = `<div style="font-size:11px;color:#8890a0;margin-bottom:4px">${params[0].axisValue}</div>`
        for (const p of params) {
          const val = typeof p.value === 'number' ? p.value.toFixed(4) : '--'
          html += `<div style="display:flex;align-items:center;gap:6px;margin:2px 0"><span style="width:8px;height:8px;border-radius:50%;background:${p.color};flex-shrink:0"></span><span style="font-size:12px">${p.seriesName}: <b>${val}</b></span></div>`
        }
        return html
      },
    },
    legend: {
      bottom: 6,
      textStyle: { color: '#8890a0', fontSize: 11 },
      icon: 'circle',
      itemWidth: 8, itemHeight: 8,
    },
    dataZoom: [{ type: 'inside' as const, start: 0, end: 100 }],
    xAxis: {
      type: 'category' as const,
      data: timeLabels,
      axisLine: { lineStyle: { color: themeStore.isDark ? '#2d2d4a' : '#e5e7eb' } },
      axisLabel: { color: '#4a5568', fontSize: 10, rotate: 30 },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value' as const,
      min: 0, max: 1, splitNumber: 4,
      axisLine: { show: false },
      axisLabel: { color: '#4a5568', fontSize: 10, formatter: (v: number) => v.toFixed(1) },
      splitLine: { lineStyle: { color: themeStore.isDark ? '#1e1e36' : '#f1f5f9' } },
    },
    series: [
      {
        name: '生产模型',
        type: 'line',
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 2, color: '#52c41a' },
        areaStyle: {
          color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [{ offset: 0, color: '#52c41a30' }, { offset: 1, color: '#52c41a05' }] },
        },
        data: data.map((d: any) => d.production),
        markLine: {
          silent: true, symbol: 'none',
          lineStyle: { type: 'dashed', color: '#ef444466', width: 1 },
          data: [{ yAxis: 0.7, label: { show: true, formatter: '告警阈值', color: '#ef444488', fontSize: 10, position: 'insideEndTop' as const } }],
        },
      },
      {
        name: '影子模型',
        type: 'line',
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 2, color: '#1890ff' },
        areaStyle: {
          color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [{ offset: 0, color: '#1890ff30' }, { offset: 1, color: '#1890ff05' }] },
        },
        data: data.map((d: any) => d.shadow),
      },
    ],
    animation: true,
    animationDuration: 500,
  }
})

// ── ECharts: Score distribution ──
const distChartOption = computed(() => {
  const dist = abDistribution.value
  if (!dist || !dist.shadow_counts || dist.shadow_counts.length === 0) return null

  const bins = dist.shadow_counts.length
  const labels = Array.from({ length: bins }, (_, i) => (i / bins).toFixed(2) + '-' + ((i + 1) / bins).toFixed(2))

  return {
    backgroundColor: 'transparent',
    grid: { left: 50, right: 20, top: 16, bottom: 60 },
    tooltip: {
      trigger: 'axis' as const,
      backgroundColor: themeStore.isDark ? '#1a1a2e' : '#ffffff',
      borderColor: themeStore.isDark ? '#2d2d4a' : '#e5e7eb',
      textStyle: { color: themeStore.isDark ? '#e2e8f0' : '#111827', fontSize: 12 },
    },
    legend: {
      bottom: 6,
      textStyle: { color: '#8890a0', fontSize: 11 },
      icon: 'roundRect',
      itemWidth: 12, itemHeight: 8,
    },
    xAxis: {
      type: 'category' as const,
      data: labels,
      axisLine: { lineStyle: { color: themeStore.isDark ? '#2d2d4a' : '#e5e7eb' } },
      axisLabel: { color: '#4a5568', fontSize: 9, rotate: 45, interval: Math.max(0, Math.floor(bins / 8) - 1) },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value' as const,
      axisLine: { show: false },
      axisLabel: { color: '#4a5568', fontSize: 10 },
      splitLine: { lineStyle: { color: themeStore.isDark ? '#1e1e36' : '#f1f5f9' } },
    },
    series: [
      {
        name: '生产模型',
        type: 'bar',
        barGap: '10%',
        itemStyle: { color: '#52c41a', borderRadius: [2, 2, 0, 0] },
        data: dist.production_counts,
      },
      {
        name: '影子模型',
        type: 'bar',
        itemStyle: { color: '#1890ff', borderRadius: [2, 2, 0, 0] },
        data: dist.shadow_counts,
      },
    ],
    animation: true,
    animationDuration: 500,
  }
})

// ── Computed helpers ──
const fpDelta = computed(() => shadowReport.value?.false_positive_delta ?? 0)
const fpDeltaColor = computed(() => fpDelta.value > 0 ? '#ff4d4f' : fpDelta.value < 0 ? '#52c41a' : '#8890a0')
</script>

<template>
  <div>
    <!-- Selector bar -->
    <Card size="small" style="margin-bottom: 16px">
      <Row :gutter="16" align="middle">
        <Col :span="6">
          <div style="font-size: 12px; color: #8890a0; margin-bottom: 4px">摄像头</div>
          <Select
            v-model:value="selectedCamera"
            placeholder="选择摄像头"
            style="width: 100%"
            :loading="loadingModels"
            allowClear
          >
            <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
              {{ cam.camera_id }} — {{ cam.name || cam.camera_id }}
            </Select.Option>
          </Select>
        </Col>
        <Col :span="7">
          <div style="font-size: 12px; color: #8890a0; margin-bottom: 4px">影子/金丝雀模型</div>
          <Select
            v-model:value="selectedShadow"
            placeholder="选择待对比模型"
            style="width: 100%"
            :disabled="!selectedCamera || shadowModels.length === 0"
          >
            <Select.Option v-for="m in shadowModels" :key="m.model_version_id" :value="m.model_version_id">
              <Tag :color="(STAGE_MAP[m.stage] || { color: 'default' }).color" style="margin-right: 4px">
                {{ (STAGE_MAP[m.stage] || { text: m.stage }).text }}
              </Tag>
              <span style="font-family: monospace; font-size: 12px">{{ m.model_version_id }}</span>
            </Select.Option>
          </Select>
        </Col>
        <Col :span="3">
          <div style="font-size: 12px; color: #8890a0; margin-bottom: 4px">时间范围</div>
          <Select v-model:value="daysRange" style="width: 100%" @change="loadAllData">
            <Select.Option :value="1">1 天</Select.Option>
            <Select.Option :value="3">3 天</Select.Option>
            <Select.Option :value="7">7 天</Select.Option>
            <Select.Option :value="14">14 天</Select.Option>
            <Select.Option :value="30">30 天</Select.Option>
          </Select>
        </Col>
        <Col :span="8" style="display: flex; gap: 8px; align-items: flex-end; padding-top: 18px">
          <Button type="primary" :disabled="!selectedShadow" :loading="isAnyLoading" @click="loadAllData">
            <template #icon><ReloadOutlined /></template>
            刷新
          </Button>
          <Button :disabled="!selectedShadow" :loading="loadingLive" @click="runLiveCompare">
            <template #icon><CameraOutlined /></template>
            抓帧对比
          </Button>
          <Button
            :disabled="!selectedShadow"
            :type="autoRefresh ? 'primary' : 'default'"
            :ghost="autoRefresh"
            @click="autoRefresh = !autoRefresh"
          >
            {{ autoRefresh ? '停止刷新' : '自动 30s' }}
          </Button>
        </Col>
      </Row>
    </Card>

    <!-- Empty state -->
    <div v-if="!selectedShadow" style="padding: 60px 0">
      <Empty description="选择一个摄像头和影子模型以开始 A/B 对比" />
    </div>

    <template v-else>
      <!-- Summary metrics -->
      <Row :gutter="16" style="margin-bottom: 16px">
        <Col :span="4">
          <Card size="small">
            <Statistic title="采样总数" :value="shadowReport?.total_samples ?? '-'" :loading="loadingReport" />
          </Card>
        </Col>
        <Col :span="4">
          <Card size="small">
            <Statistic
              title="生产告警率"
              :value="shadowReport ? (shadowReport.production_alert_rate * 100).toFixed(1) + '%' : '-'"
              :value-style="{ color: '#52c41a' }"
              :loading="loadingReport"
            />
          </Card>
        </Col>
        <Col :span="4">
          <Card size="small">
            <Statistic
              title="影子告警率"
              :value="shadowReport ? (shadowReport.shadow_alert_rate * 100).toFixed(1) + '%' : '-'"
              :value-style="{ color: '#1890ff' }"
              :loading="loadingReport"
            />
          </Card>
        </Col>
        <Col :span="4">
          <Card size="small">
            <Statistic
              title="误报差异"
              :value="fpDelta > 0 ? '+' + fpDelta : String(fpDelta)"
              :value-style="{ color: fpDeltaColor }"
              :loading="loadingReport"
            />
          </Card>
        </Col>
        <Col :span="4">
          <Card size="small">
            <Statistic
              title="平均分数偏差"
              :value="shadowReport?.avg_score_divergence?.toFixed(4) ?? '-'"
              :loading="loadingReport"
            />
          </Card>
        </Col>
        <Col :span="4">
          <Card size="small">
            <Statistic
              title="影子推理延迟"
              :value="shadowReport?.avg_shadow_latency_ms ? shadowReport.avg_shadow_latency_ms.toFixed(1) + ' ms' : '-'"
              :loading="loadingReport"
            />
          </Card>
        </Col>
      </Row>

      <!-- Score chart (ECharts) -->
      <Card size="small" style="margin-bottom: 16px">
        <template #title>
          <div style="display: flex; justify-content: space-between; align-items: center">
            <span>分数对比</span>
            <Segmented v-model:value="chartView" :options="['时序', '分布']" size="small" />
          </div>
        </template>
        <Spin :spinning="loadingScores || loadingDistribution">
          <div v-show="chartView === '时序'" style="height: 280px">
            <VChart
              v-if="scoreChartOption"
              :option="scoreChartOption"
              :autoresize="true"
              style="width: 100%; height: 100%"
            />
            <div v-else style="height: 100%; display: flex; align-items: center; justify-content: center; color: #4a5568">
              暂无分数数据
            </div>
          </div>
          <div v-show="chartView === '分布'" style="height: 280px">
            <VChart
              v-if="distChartOption"
              :option="distChartOption"
              :autoresize="true"
              style="width: 100%; height: 100%"
            />
            <div v-else style="height: 100%; display: flex; align-items: center; justify-content: center; color: #4a5568">
              暂无分布数据
            </div>
          </div>
        </Spin>
      </Card>

      <!-- Live A/B heatmap comparison -->
      <Card size="small" style="margin-bottom: 16px">
        <template #title>
          <div style="display: flex; justify-content: space-between; align-items: center">
            <Space>
              <ThunderboltOutlined />
              <span>实时热力图对比</span>
            </Space>
            <Space>
              <span style="font-size: 12px; color: #8890a0">热力图透明度</span>
              <Slider
                v-model:value="heatmapOpacity"
                :min="0" :max="100" :step="5"
                style="width: 120px"
              />
              <Button size="small" :loading="loadingLive" @click="runLiveCompare">
                <template #icon><CameraOutlined /></template>
                抓帧对比
              </Button>
            </Space>
          </div>
        </template>

        <Spin :spinning="loadingLive">
          <div v-if="liveResult" class="ab-heatmap-grid">
            <!-- Original frame -->
            <div class="ab-panel">
              <div class="ab-panel-header">原始帧</div>
              <div class="ab-image-container">
                <img
                  v-if="liveResult.original_frame"
                  :src="'data:image/jpeg;base64,' + liveResult.original_frame"
                  class="ab-image"
                />
              </div>
            </div>

            <!-- Production heatmap -->
            <div class="ab-panel">
              <div class="ab-panel-header">
                <Tag color="green">生产模型</Tag>
                <span v-if="liveResult.production_score != null" style="font-family: monospace">
                  分数: {{ liveResult.production_score }}
                </span>
                <span v-if="liveResult.production_latency_ms" style="color: #8890a0; font-size: 12px; margin-left: 8px">
                  {{ liveResult.production_latency_ms }} ms
                </span>
              </div>
              <div class="ab-image-container">
                <img
                  v-if="liveResult.original_frame"
                  :src="'data:image/jpeg;base64,' + liveResult.original_frame"
                  class="ab-image"
                />
                <img
                  v-if="liveResult.production_heatmap"
                  :src="'data:image/jpeg;base64,' + liveResult.production_heatmap"
                  class="ab-heatmap-overlay"
                  :style="{ opacity: heatmapOpacity / 100 }"
                />
                <div v-if="liveResult.production_error" class="ab-error">
                  {{ liveResult.production_error }}
                </div>
              </div>
            </div>

            <!-- Shadow heatmap -->
            <div class="ab-panel">
              <div class="ab-panel-header">
                <Tag color="blue">影子模型</Tag>
                <span v-if="liveResult.shadow_score != null" style="font-family: monospace">
                  分数: {{ liveResult.shadow_score }}
                </span>
                <span v-if="liveResult.shadow_latency_ms" style="color: #8890a0; font-size: 12px; margin-left: 8px">
                  {{ liveResult.shadow_latency_ms }} ms
                </span>
              </div>
              <div class="ab-image-container">
                <img
                  v-if="liveResult.original_frame"
                  :src="'data:image/jpeg;base64,' + liveResult.original_frame"
                  class="ab-image"
                />
                <img
                  v-if="liveResult.shadow_heatmap"
                  :src="'data:image/jpeg;base64,' + liveResult.shadow_heatmap"
                  class="ab-heatmap-overlay"
                  :style="{ opacity: heatmapOpacity / 100 }"
                />
                <div v-if="liveResult.shadow_error" class="ab-error">
                  {{ liveResult.shadow_error }}
                </div>
              </div>
            </div>
          </div>
          <div v-else style="text-align: center; padding: 40px; color: #666">
            点击「抓帧对比」按钮获取实时热力图
          </div>
        </Spin>
      </Card>
    </template>
  </div>
</template>

<style scoped>
.ab-heatmap-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 12px;
}

.ab-panel {
  background: var(--bg);
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid #2a2a3e;
}

.ab-panel-header {
  padding: 8px 12px;
  font-size: 13px;
  display: flex;
  align-items: center;
  gap: 8px;
  border-bottom: 1px solid #2a2a3e;
}

.ab-image-container {
  position: relative;
  aspect-ratio: 16 / 9;
  background: #111;
}

.ab-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.ab-heatmap-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  mix-blend-mode: screen;
  pointer-events: none;
}

.ab-error {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ff4d4f;
  font-size: 13px;
  background: rgba(0, 0, 0, 0.6);
}

@media (max-width: 1200px) {
  .ab-heatmap-grid {
    grid-template-columns: 1fr 1fr;
  }
}

@media (max-width: 768px) {
  .ab-heatmap-grid {
    grid-template-columns: 1fr;
  }
}
</style>
