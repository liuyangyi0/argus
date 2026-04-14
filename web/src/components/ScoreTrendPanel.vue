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

  const series = cams.map((cam, idx) => {
    const baseColor = COLORS[idx % COLORS.length]
    return {
      name: cam.name || cam.camera_id,
      type: 'line' as const,
      smooth: true,
      symbol: 'none',
      sampling: 'lttb' as const,
      lineStyle: { width: 2, color: baseColor },
      areaStyle: {
        color: {
          type: 'linear' as const,
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: baseColor + '29' }, // ~16% opacity to match rgba(..., 0.16)
            { offset: 1, color: baseColor + '00' }, // transparent
          ],
        },
      },
      data: (history.value[cam.camera_id] || []).map(h => h.score),
      markLine: idx === 0 ? {
        silent: true,
        symbol: 'none',
        lineStyle: { type: 'dashed' as const, color: '#ef444466', width: 1 },
        data: [{ yAxis: 0.7, label: { show: false } }], // Hide label to keep it clean
      } : undefined,
    }
  })

  return {
    backgroundColor: 'transparent',
    grid: { top: 10, right: 0, bottom: 0, left: 0 },
    tooltip: {
      trigger: 'axis' as const,
      backgroundColor: 'rgba(255, 255, 255, 0.9)',
      borderColor: 'rgba(10, 10, 15, 0.1)',
      textStyle: { color: '#0a0a0c', fontSize: 12 },
      formatter: (params: any) => {
        if (!Array.isArray(params) || params.length === 0) return ''
        let html = `<div style="font-size:11px;color:#65656e;margin-bottom:4px">${params[0].axisValue}</div>`
        for (const p of params) {
          const color = p.color || '#fff'
          const val = typeof p.value === 'number' ? p.value.toFixed(3) : '--'
          html += `<div style="display:flex;align-items:center;gap:6px;font-size:12px;margin:2px 0"><span style="width:8px;height:8px;border-radius:50%;background:${color};flex-shrink:0"></span><span style="color:#18181c">${p.seriesName}:</span> <b>${val}</b></div>`
        }
        return html
      },
    },
    legend: {
      show: false, // Hide legend for extreme minimalism as per design intent
    },
    xAxis: {
      type: 'category' as const,
      data: labels,
      show: false,
    },
    yAxis: {
      type: 'value' as const,
      min: 0,
      max: 1,
      show: false,
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
  <div class="trend-panel glass-panel">
    <div class="trend-header">
      <Typography.Text strong class="text-h3">异常分数趋势</Typography.Text>
      <Typography.Text type="secondary" class="text-micro">实时更新 · 最近 2 分钟</Typography.Text>
    </div>
    <div class="trend-chart-container">
      <VChart
        v-if="hasData"
        :option="chartOption"
        :autoresize="true"
        style="width: 100%; height: 100%; position: relative; z-index: 1;"
      />
      <div v-else class="trend-empty">
        <div class="trend-empty-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
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
  /* background, border and shadow are handled by .glass-panel from global utilities */
  overflow: hidden;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
}

.trend-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  border-bottom: 1px solid var(--line);
}

.trend-chart-container {
  height: 140px; /* Slight height increase to feature the clean chart */
  padding: 0;    /* Remove padding to let chart reach edges */
  position: relative;
}

.trend-empty {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: var(--ink-4);
  font-size: 12px;
}

.trend-empty-icon {
  opacity: 0.5;
}
</style>
