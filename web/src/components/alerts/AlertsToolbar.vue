<script setup lang="ts">
import { Select } from 'ant-design-vue'
import { storeToRefs } from 'pinia'
import { useAlertStore } from '../../stores/useAlertStore'

const alertStore = useAlertStore()
const { activeCount, resolvedCount, totalAlerts, filters, cameras } = storeToRefs(alertStore)

let _filterTimer: ReturnType<typeof setTimeout> | null = null
function debouncedFetchData() {
  if (_filterTimer) clearTimeout(_filterTimer)
  _filterTimer = setTimeout(() => alertStore.fetchData(), 300)
}

function buildExportParams() {
  const params = new URLSearchParams()
  if (filters.value.camera_id) params.set('camera_id', filters.value.camera_id)
  if (filters.value.severity) params.set('severity', filters.value.severity)
  return params.toString() ? '?' + params.toString() : ''
}

function handleExportCSV() {
  window.open(`/api/alerts/export-csv${buildExportParams()}`, '_blank')
}

function handleExportPDF() {
  window.open(`/api/alerts/export-pdf${buildExportParams()}`, '_blank')
}
</script>

<template>
  <div class="alerts-header">
    <div class="alerts-header-top">
      <div>
        <h2 class="alerts-title">告警<span class="alerts-accent">中心</span></h2>
      </div>
      <div class="alerts-stats">
        <span>待处理活跃 <b class="mono">{{ String(activeCount).padStart(2, '0') }}</b></span>
        <span>今日已解决 <b class="mono">{{ String(resolvedCount).padStart(2, '0') }}</b></span>
        <span>历史总计 <b class="mono">{{ String(totalAlerts).padStart(2, '0') }}</b></span>
      </div>
    </div>
    <div class="alerts-filters">
      <Select
        v-model:value="filters.camera_id"
        placeholder="全部点位"
        allow-clear
        size="small"
        style="width: 140px; border-radius: 6px;"
        @change="debouncedFetchData"
      >
        <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
          {{ cam.camera_id }}
        </Select.Option>
      </Select>
      
      <div class="alerts-segment">
        <button :class="['alerts-segment-btn', { on: !filters.severity }]" @click="filters.severity = ''; alertStore.fetchData()">全部等级</button>
        <button :class="['alerts-segment-btn', { on: filters.severity === 'high' }]" @click="filters.severity = 'high'; alertStore.fetchData()">高危</button>
        <button :class="['alerts-segment-btn', { on: filters.severity === 'medium' }]" @click="filters.severity = 'medium'; alertStore.fetchData()">中危</button>
        <button :class="['alerts-segment-btn', { on: filters.severity === 'low' }]" @click="filters.severity = 'low'; alertStore.fetchData()">低危</button>
      </div>

      <div class="alerts-actions-end">
        <button class="alerts-btn" @click="handleExportCSV">导出 CSV ⭳</button>
        <button class="alerts-btn" @click="handleExportPDF">导出 PDF ⭳</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ── Header ── */
.alerts-header {
  padding: 18px 20px 0;
  flex-shrink: 0;
  border-bottom: 1px solid var(--line-2);
}
.alerts-header-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 14px;
}
.alerts-eyebrow {
  font-size: 11.5px;
  font-weight: 500;
  color: var(--ink-4);
  letter-spacing: .08em;
  margin-bottom: 4px;
}
.alerts-title {
  font-size: 24px;
  font-weight: 800;
  margin: 0;
  letter-spacing: .04em;
  color: var(--ink-2);
}
.alerts-accent {
  color: #3b82f6;
}
.alerts-stats {
  display: flex;
  gap: 16px;
  font-size: 11.5px;
  font-weight: 500;
  color: var(--ink-4);
}
.alerts-stats b {
  color: var(--ink-2);
  margin-left: 6px;
  font-size: 14px;
}

/* ── Filter & Actions ── */
.alerts-filters {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px 0;
  flex-wrap: nowrap;
}

.alerts-segment {
  display: inline-flex;
  background: rgba(10, 10, 15, .04);
  padding: 4px;
  border-radius: 8px;
  gap: 4px;
}
.alerts-segment-btn {
  padding: 4px 16px;
  border: none;
  background: transparent;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
  color: var(--ink-4);
  cursor: pointer;
  white-space: nowrap;
  transition: all .2s cubic-bezier(0.4, 0, 0.2, 1);
}
.alerts-segment-btn:hover:not(.on) {
  color: var(--ink-2);
}
.alerts-segment-btn.on {
  background: #fff;
  color: var(--ink-2);
  box-shadow: 0 1px 3px rgba(0,0,0,.08), 0 1px 0 rgba(0,0,0,.02);
}

.alerts-actions-end {
  margin-left: auto;
  display: flex;
  gap: 8px;
}

.alerts-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 14px;
  border: 1px solid var(--line-2);
  background: rgba(255, 255, 255, 0.6);
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
  white-space: nowrap;
  cursor: pointer;
  color: var(--ink-4);
  transition: all .2s;
}
.alerts-btn:hover {
  background: #fff;
  border-color: #cbd5e1;
  color: var(--ink-2);
}
</style>
