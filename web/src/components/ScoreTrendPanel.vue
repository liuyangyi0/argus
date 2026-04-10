<script setup lang="ts">
/**
 * ScoreTrendPanel — Real-time multi-camera anomaly score trend chart.
 *
 * Renders an ECharts area chart with one series per camera, updated
 * in real-time via WebSocket or polling. Inspired by Frigate NVR's
 * detection timeline and Grafana's time-series panels.
 */
import { computed, ref, watch } from 'vue'
import { Typography } from 'ant-design-vue'
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  MarkLineComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'
import type { CameraTileData } from './VideoTile.vue'
import { useThemeStore } from '../stores/theme'

const themeStore = useThemeStore()

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, MarkLineComponent])

const props = defineProps<{
  cameras: CameraTileData[]
}>()

// Color palette for camera lines (Frigate-inspired)
const COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#10b981', '#f97316']

// Accumulate score history per camera (ring buffer, max 120 points = ~2 min at 1/sec)
const MAX_POINTS = 120
const history = ref<Record<string, { time: string; score: number }[]>>({})
const timeLabels = ref<string[]>([])

// Narrow computed to only track camera IDs + scores (avoids deep reactivity on entire camera objects)
const scoreSnapshot = computed(() =>
  props.cameras.map(c => c.camera_id + ':' + (c.current_score ?? 0).toFixed(4)).join('|')
)

watch(scoreSnapshot, () => {
  const cams = props.cameras
  if (!cams || cams.length === 0) return

  const now = new Date()
  const timeStr = now.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })

  timeLabels.value.push(timeStr)
  if (timeLabels.value.length > MAX_POINTS) timeLabels.value.shift()

  for (const cam of cams) {
    if (!history.value[cam.camera_id]) {
      history.value[cam.camera_id] = []
    }
    history.value[cam.camera_id].push({
      time: timeStr,
      score: cam.current_score ?? 0,
    })
    if (history.value[cam.camera_id].length > MAX_POINTS) {
      history.value[cam.camera_id].shift()
    }
  }
})

const chartOption = computed(() => {
  const cams = props.cameras || []
  const labels = timeLabels.value

  const series = cams.map((cam, idx) => ({
    name: cam.name || cam.camera_id,
    type: 'line' as const,
    smooth: true,
    symbol: 'none',
    sampling: 'lttb' as const,
    lineStyle: { width: 1.5, color: COLORS[idx % COLORS.length] },
    areaStyle: {
      color: {
        type: 'linear' as const,
        x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [
          { offset: 0, color: COLORS[idx % COLORS.length] + '30' },
          { offset: 1, color: COLORS[idx % COLORS.length] + '05' },
        ],
      },
    },
    data: (history.value[cam.camera_id] || []).map(h => h.score),
    markLine: idx === 0 ? {
      silent: true,
      symbol: 'none',
      lineStyle: { type: 'dashed' as const, color: '#ef444466', width: 1 },
      data: [{ yAxis: 0.7, label: { show: true, formatter: '告警阈值', color: '#ef444488', fontSize: 10, position: 'insideEndTop' as const } }],
    } : undefined,
  }))

  return {
    backgroundColor: 'transparent',
    grid: { left: 45, right: 16, top: 12, bottom: 50 },
    tooltip: {
      trigger: 'axis' as const,
      backgroundColor: themeStore.isDark ? '#1a1a2e' : '#ffffff',
      borderColor: themeStore.isDark ? '#2d2d4a' : '#e5e7eb',
      textStyle: { color: themeStore.isDark ? '#e2e8f0' : '#111827', fontSize: 12 },
      formatter: (params: any) => {
        if (!Array.isArray(params) || params.length === 0) return ''
        let html = `<div style="font-size:11px;color:#8890a0;margin-bottom:4px">${params[0].axisValue}</div>`
        for (const p of params) {
          const color = p.color || '#fff'
          const val = typeof p.value === 'number' ? p.value.toFixed(3) : '--'
          html += `<div style="display:flex;align-items:center;gap:6px;font-size:12px;margin:2px 0"><span style="width:8px;height:8px;border-radius:50%;background:${color};flex-shrink:0"></span>${p.seriesName}: <b>${val}</b></div>`
        }
        return html
      },
    },
    legend: {
      bottom: 6,
      textStyle: { color: themeStore.isDark ? '#6b7280' : '#9ca3af', fontSize: 11 },
      icon: 'circle',
      itemWidth: 8,
      itemHeight: 8,
      itemGap: 12,
    },
    xAxis: {
      type: 'category' as const,
      data: labels,
      axisLine: { lineStyle: { color: themeStore.isDark ? '#2d2d4a' : '#e5e7eb' } },
      axisLabel: { color: themeStore.isDark ? '#4a5568' : '#9ca3af', fontSize: 10, interval: 'auto' },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value' as const,
      min: 0,
      max: 1,
      splitNumber: 4,
      axisLine: { show: false },
      axisLabel: { color: themeStore.isDark ? '#4a5568' : '#9ca3af', fontSize: 10, formatter: (v: number) => v.toFixed(1) },
      splitLine: { lineStyle: { color: themeStore.isDark ? '#1e1e36' : '#f1f5f9' } },
    },
    dataZoom: [{
      type: 'inside' as const,
      start: labels.length > 60 ? 50 : 0,
      end: 100,
    }],
    series,
    animation: false,
  }
})

const hasData = computed(() => timeLabels.value.length > 1)
</script>

<template>
  <div class="trend-panel">
    <div class="trend-header">
      <Typography.Text strong style="font-size: 13px; color: var(--argus-text)">异常分数趋势</Typography.Text>
      <Typography.Text type="secondary" style="font-size: 11px">实时更新 · 最近 2 分钟</Typography.Text>
    </div>
    <div class="trend-chart-container">
      <VChart
        v-if="hasData"
        :option="chartOption"
        :autoresize="true"
        style="width: 100%; height: 100%"
      />
      <div v-else class="trend-empty">
        <div class="trend-empty-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#4a5568" stroke-width="1.5">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
        </div>
        <span>等待数据...</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.trend-panel {
  background: var(--argus-card-bg);
  border: 1px solid var(--argus-border);
  border-radius: 8px;
  overflow: hidden;
  flex-shrink: 0;
}

.trend-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 14px;
  border-bottom: 1px solid var(--argus-hover-bg);
}

.trend-chart-container {
  height: 120px;
  padding: 4px;
}

.trend-empty {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: #4a5568;
  font-size: 12px;
}

.trend-empty-icon {
  opacity: 0.5;
}
</style>
