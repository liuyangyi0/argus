<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { Card, Slider, Statistic, Row, Col, Spin, Typography, Alert, Button, Empty } from 'ant-design-vue'
import { ReloadOutlined, ThunderboltOutlined, InfoCircleOutlined } from '@ant-design/icons-vue'
import { use } from 'echarts/core'
import { LineChart, ScatterChart } from 'echarts/charts'
import { CanvasRenderer } from 'echarts/renderers'
import { GridComponent, TooltipComponent, MarkLineComponent, LegendComponent, MarkPointComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import { getTrainingMetrics } from '../../api/training'
import ConfusionMatrix from './ConfusionMatrix.vue'
import type { TrainingMetricsResponse } from '../../types/api'
import { extractErrorMessage } from '../../utils/error'

use([
  CanvasRenderer, LineChart, ScatterChart,
  GridComponent, TooltipComponent, MarkLineComponent, LegendComponent, MarkPointComponent,
])

// Empty.PRESENTED_IMAGE_SIMPLE is a function returning a VNode; exposing via
// a const so template bindings stay reactive-friendly under <script setup>.
const emptyImage = Empty.PRESENTED_IMAGE_SIMPLE

const props = defineProps<{ recordId: number | null }>()

const loading = ref(false)
const data = ref<TrainingMetricsResponse | null>(null)
const threshold = ref(0.7)
const errorMsg = ref<string | null>(null)

async function load() {
  if (props.recordId == null) return
  loading.value = true
  errorMsg.value = null
  try {
    const res = await getTrainingMetrics(props.recordId)
    data.value = res
    if (res.threshold_used != null) {
      threshold.value = res.threshold_used
    }
  } catch (e) {
    errorMsg.value = extractErrorMessage(e, '加载指标失败')
    data.value = null
  } finally {
    loading.value = false
  }
}

watch(() => props.recordId, load, { immediate: true })

// Client-side re-computation at the slider threshold
const liveMetrics = computed(() => {
  if (!data.value?.has_labeled_eval || !data.value.scores || !data.value.labels) {
    return null
  }
  const scores = data.value.scores
  const labels = data.value.labels
  let tp = 0, fp = 0, fn = 0, tn = 0
  for (let i = 0; i < scores.length; i++) {
    const pred = scores[i] >= threshold.value ? 1 : 0
    const actual = labels[i]
    if (pred === 1 && actual === 1) tp++
    else if (pred === 1 && actual === 0) fp++
    else if (pred === 0 && actual === 1) fn++
    else tn++
  }
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0
  const fpr = fp + tn > 0 ? fp / (fp + tn) : 0
  return {
    precision, recall, f1, fpr, tp, fp, fn, tn,
    confusion_matrix: { tp, fp, fn, tn },
  }
})

// ROC curve: computed client-side by sweeping thresholds over the scores array
const rocPoints = computed(() => {
  if (!data.value?.scores || !data.value.labels) return { fprs: [] as number[], tprs: [] as number[] }
  const scores = data.value.scores
  const labels = data.value.labels
  const uniq = Array.from(new Set([...scores, 0, 1])).sort((a, b) => b - a)
  const nPos = labels.filter((l) => l === 1).length
  const nNeg = labels.length - nPos
  if (nPos === 0 || nNeg === 0) return { fprs: [0, 1], tprs: [0, 1] }
  const fprs: number[] = []
  const tprs: number[] = []
  for (const t of uniq) {
    let tp = 0, fp = 0
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] >= t) {
        if (labels[i] === 1) tp++
        else fp++
      }
    }
    fprs.push(fp / nNeg)
    tprs.push(tp / nPos)
  }
  return { fprs, tprs }
})

const prChartOption = computed(() => {
  const curve = data.value?.pr_curve
  if (!curve || !curve.recalls?.length) return {}
  const points = curve.recalls.map((r, i) => [r, curve.precisions[i]])
  // Highlight the point corresponding to the current slider threshold
  const lm = liveMetrics.value
  const markPoint = lm ? {
    data: [{
      name: '当前阈值',
      value: [lm.recall, lm.precision],
      coord: [lm.recall, lm.precision],
      symbol: 'pin',
      symbolSize: 40,
      itemStyle: { color: '#e5484d' },
      label: {
        formatter: `t=${threshold.value.toFixed(2)}`,
        fontSize: 10,
        position: 'top',
      },
    }],
  } : undefined
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (p: any) => `recall: ${p[0].value[0].toFixed(3)}<br/>precision: ${p[0].value[1].toFixed(3)}`,
    },
    grid: { left: 50, right: 20, top: 30, bottom: 40 },
    xAxis: { type: 'value', name: 'Recall', min: 0, max: 1 },
    yAxis: { type: 'value', name: 'Precision', min: 0, max: 1 },
    series: [{
      type: 'line',
      data: points,
      smooth: false,
      symbol: 'circle',
      symbolSize: 3,
      lineStyle: { color: '#2563eb', width: 2 },
      areaStyle: { color: 'rgba(37, 99, 235, 0.1)' },
      markPoint,
    }],
  }
})

const rocChartOption = computed(() => {
  const { fprs, tprs } = rocPoints.value
  if (!fprs.length) return {}
  const points = fprs.map((f, i) => [f, tprs[i]])
  // Sort by FPR for proper line rendering
  points.sort((a, b) => a[0] - b[0])
  // Reference diagonal (random classifier)
  const diag = [[0, 0], [1, 1]]
  const lm = liveMetrics.value
  const markPoint = lm ? {
    data: [{
      name: '当前阈值',
      value: [lm.fpr, lm.recall],
      coord: [lm.fpr, lm.recall],
      symbol: 'pin',
      symbolSize: 40,
      itemStyle: { color: '#e5484d' },
      label: {
        formatter: `t=${threshold.value.toFixed(2)}`,
        fontSize: 10,
        position: 'top',
      },
    }],
  } : undefined
  return {
    tooltip: {
      trigger: 'axis',
      formatter: (p: any) => p
        .filter((x: any) => x.seriesName === 'ROC')
        .map((x: any) => `FPR: ${x.value[0].toFixed(3)}<br/>TPR: ${x.value[1].toFixed(3)}`)
        .join('<br/>'),
    },
    legend: { data: ['ROC', '随机基线'], top: 0 },
    grid: { left: 50, right: 20, top: 30, bottom: 40 },
    xAxis: { type: 'value', name: 'FPR', min: 0, max: 1 },
    yAxis: { type: 'value', name: 'TPR (Recall)', min: 0, max: 1 },
    series: [
      {
        name: 'ROC',
        type: 'line',
        data: points,
        smooth: false,
        symbol: 'circle',
        symbolSize: 2,
        lineStyle: { color: '#15a34a', width: 2 },
        areaStyle: { color: 'rgba(21, 163, 74, 0.1)' },
        markPoint,
      },
      {
        name: '随机基线',
        type: 'line',
        data: diag,
        symbol: 'none',
        lineStyle: { color: '#999', type: 'dashed', width: 1 },
      },
    ],
  }
})

function applyOptimalThreshold() {
  if (data.value?.optimal_f1_threshold != null) {
    threshold.value = data.value.optimal_f1_threshold
  }
}

function applyStoredThreshold() {
  if (data.value?.threshold_used != null) {
    threshold.value = data.value.threshold_used
  }
}
</script>

<template>
  <Spin :spinning="loading">
    <Alert v-if="errorMsg" type="error" :message="errorMsg" style="margin-bottom: 12px" />

    <div v-if="!data || !data.has_labeled_eval" class="no-data">
      <Empty :image="emptyImage">
        <template #description>
          <Typography.Title :level="5" style="margin-bottom: 4px">
            本次训练未产生真实评估数据
          </Typography.Title>
          <Typography.Text type="secondary">
            {{ data?.message || '后端检测到 data/validation/ 或 data/foe_objects/ 样本不足，评估步骤已被跳过（详见后端日志 validation.real_labeled_skipped / validation.synthetic_skipped）。' }}
          </Typography.Text>
        </template>
      </Empty>

      <Alert
        type="info"
        show-icon
        style="margin-top: 16px; text-align: left"
      >
        <template #icon><InfoCircleOutlined /></template>
        <template #message>
          <strong>如何生成评估指标</strong>
        </template>
        <template #description>
          <div style="margin-top: 8px">
            <Typography.Paragraph style="margin-bottom: 6px">
              真实评估数据由两类目录驱动，后端每次训练完自动读取：
            </Typography.Paragraph>
            <ul style="margin: 0 0 8px 18px; padding: 0; line-height: 1.8">
              <li>
                <strong>真实人工标注（最推荐）</strong>
                <ul style="margin: 2px 0 0 18px; padding: 0">
                  <li>
                    阳性样本（确认异物）：
                    <code>data/validation/&lt;camera_id&gt;/confirmed/</code>
                  </li>
                  <li>
                    阴性样本（被否决的误报）：
                    <code>data/baselines/&lt;camera_id&gt;/false_positives/</code>
                  </li>
                  <li>每个目录至少 <strong>10</strong> 张 PNG/JPG，才会启用 P/R/F1/AUROC/PR-AUC。</li>
                </ul>
              </li>
              <li style="margin-top: 6px">
                <strong>合成 FOE 对象（可选，补充）</strong>
                <ul style="margin: 2px 0 0 18px; padding: 0">
                  <li>
                    透明背景的异物 PNG：<code>data/foe_objects/*.png</code>
                  </li>
                  <li>启用 holdout AUROC + synthetic recall 两个步骤。</li>
                </ul>
              </li>
            </ul>
            <Typography.Text type="secondary" style="font-size: 12px">
              两类目录缺失时训练依然会完成，但指标图表无数据。详见后端模块
              <code>src/argus/anomaly/training_validator.py</code> 的模块 docstring
              与日志关键字 <code>validation.real_labeled_skipped</code>。
            </Typography.Text>
          </div>
        </template>
      </Alert>
    </div>

    <div v-else>
      <!-- Summary stats -->
      <Row :gutter="16" style="margin-bottom: 16px">
        <Col :span="4">
          <Statistic
            title="Precision (当前阈值)"
            :value="liveMetrics ? (liveMetrics.precision * 100).toFixed(1) + '%' : '-'"
            :value-style="{ color: '#2563eb' }"
          />
        </Col>
        <Col :span="4">
          <Statistic
            title="Recall (当前阈值)"
            :value="liveMetrics ? (liveMetrics.recall * 100).toFixed(1) + '%' : '-'"
            :value-style="{ color: '#15a34a' }"
          />
        </Col>
        <Col :span="4">
          <Statistic
            title="F1 (当前阈值)"
            :value="liveMetrics ? liveMetrics.f1.toFixed(4) : '-'"
            :value-style="{ color: '#a855f7' }"
          />
        </Col>
        <Col :span="4">
          <Statistic
            title="FPR (当前阈值)"
            :value="liveMetrics ? (liveMetrics.fpr * 100).toFixed(2) + '%' : '-'"
            :value-style="{ color: '#e5484d' }"
          />
        </Col>
        <Col :span="4">
          <Statistic
            title="AUROC"
            :value="data.stored_metrics?.auroc?.toFixed(4) ?? '-'"
          />
        </Col>
        <Col :span="4">
          <Statistic
            title="PR-AUC"
            :value="data.stored_metrics?.pr_auc?.toFixed(4) ?? '-'"
          />
        </Col>
      </Row>

      <!-- Threshold slider -->
      <Card size="small" style="margin-bottom: 16px">
        <template #title>
          <span>阈值滑块</span>
          <span style="margin-left: 12px; font-size: 12px; color: #666">
            当前: <strong>{{ threshold.toFixed(3) }}</strong>
            ·
            样本: {{ data.sample_count }} 张（正 {{ data.metrics_at_threshold?.n_positive }} / 负 {{ data.metrics_at_threshold?.n_negative }}）
          </span>
        </template>
        <template #extra>
          <Button size="small" @click="applyStoredThreshold" v-if="data.threshold_used != null">
            <template #icon><ReloadOutlined /></template>
            训练阈值 {{ data.threshold_used.toFixed(3) }}
          </Button>
          <Button
            size="small"
            type="primary"
            @click="applyOptimalThreshold"
            v-if="data.optimal_f1_threshold != null"
            style="margin-left: 8px"
          >
            <template #icon><ThunderboltOutlined /></template>
            最优 F1 阈值 {{ data.optimal_f1_threshold.toFixed(3) }}
          </Button>
        </template>
        <Slider
          v-model:value="threshold"
          :min="0"
          :max="1"
          :step="0.005"
          :marks="{ 0: '0', 0.25: '0.25', 0.5: '0.5', 0.7: '0.7', 0.9: '0.9', 1: '1' }"
        />
      </Card>

      <!-- PR + ROC curves side by side -->
      <Row :gutter="16" style="margin-bottom: 16px">
        <Col :span="12">
          <Card size="small" title="Precision-Recall 曲线">
            <VChart :option="prChartOption" style="height: 320px; width: 100%" autoresize />
          </Card>
        </Col>
        <Col :span="12">
          <Card size="small" title="ROC 曲线">
            <VChart :option="rocChartOption" style="height: 320px; width: 100%" autoresize />
          </Card>
        </Col>
      </Row>

      <!-- Confusion matrix at current threshold -->
      <Card size="small" title="混淆矩阵（随阈值实时更新）">
        <ConfusionMatrix :matrix="liveMetrics?.confusion_matrix" />
      </Card>
    </div>
  </Spin>
</template>

<style scoped>
.no-data {
  padding: 32px 24px;
  background: rgba(0, 0, 0, 0.02);
  border-radius: 6px;
}
.no-data :deep(.ant-empty) {
  margin-bottom: 0;
}
.no-data code {
  background: rgba(0, 0, 0, 0.05);
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 12px;
  font-family: Consolas, 'Courier New', monospace;
}
</style>
