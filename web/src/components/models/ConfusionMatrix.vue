<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  matrix: { tp: number; fp: number; fn: number; tn: number } | null | undefined
  title?: string
}>()

const total = computed(() => {
  if (!props.matrix) return 0
  return props.matrix.tp + props.matrix.fp + props.matrix.fn + props.matrix.tn
})

function pct(n: number): string {
  if (!total.value) return '-'
  return `${((n / total.value) * 100).toFixed(1)}%`
}
</script>

<template>
  <div v-if="matrix" class="cm-wrap">
    <div v-if="title" class="cm-title">{{ title }}</div>
    <table class="cm-table">
      <thead>
        <tr>
          <th></th>
          <th class="header">预测: 异常</th>
          <th class="header">预测: 正常</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th class="header">实际: 异常</th>
          <td class="cell tp">
            <div class="label">TP</div>
            <div class="value">{{ matrix.tp }}</div>
            <div class="pct">{{ pct(matrix.tp) }}</div>
          </td>
          <td class="cell fn">
            <div class="label">FN</div>
            <div class="value">{{ matrix.fn }}</div>
            <div class="pct">{{ pct(matrix.fn) }}</div>
          </td>
        </tr>
        <tr>
          <th class="header">实际: 正常</th>
          <td class="cell fp">
            <div class="label">FP</div>
            <div class="value">{{ matrix.fp }}</div>
            <div class="pct">{{ pct(matrix.fp) }}</div>
          </td>
          <td class="cell tn">
            <div class="label">TN</div>
            <div class="value">{{ matrix.tn }}</div>
            <div class="pct">{{ pct(matrix.tn) }}</div>
          </td>
        </tr>
      </tbody>
    </table>
    <div class="cm-summary">总样本: {{ total }}</div>
  </div>
  <div v-else class="cm-empty">暂无混淆矩阵数据</div>
</template>

<style scoped>
.cm-wrap {
  display: inline-block;
}
.cm-title {
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--ink, #222);
}
.cm-table {
  border-collapse: collapse;
  background: rgba(255, 255, 255, 0.02);
}
.cm-table th.header {
  padding: 8px 12px;
  font-weight: 500;
  background: rgba(0, 0, 0, 0.04);
  text-align: center;
  font-size: 13px;
}
.cm-table td.cell {
  padding: 12px 24px;
  text-align: center;
  border: 1px solid rgba(0, 0, 0, 0.12);
  min-width: 120px;
}
.cm-table td.cell .label {
  font-size: 11px;
  font-weight: 500;
  opacity: 0.6;
}
.cm-table td.cell .value {
  font-size: 22px;
  font-weight: 700;
  margin: 4px 0;
}
.cm-table td.cell .pct {
  font-size: 12px;
  opacity: 0.7;
}
.cm-table td.tp { background: rgba(21, 163, 74, 0.15); }
.cm-table td.tn { background: rgba(21, 163, 74, 0.08); }
.cm-table td.fp { background: rgba(229, 72, 77, 0.15); }
.cm-table td.fn { background: rgba(229, 72, 77, 0.12); }
.cm-summary {
  font-size: 12px;
  opacity: 0.65;
  margin-top: 6px;
}
.cm-empty {
  padding: 16px;
  text-align: center;
  color: #999;
  font-size: 13px;
}
</style>
