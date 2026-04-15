<script setup lang="ts">
import { Select } from 'ant-design-vue'
import { storeToRefs } from 'pinia'
import { useAlertStore } from '../../stores/useAlertStore'

defineOptions({ name: 'AlertsToolbar' })

const store = useAlertStore()
const { cameras, filters, totalAlerts, activeCount, resolvedCount } = storeToRefs(store)

let _filterTimer: ReturnType<typeof setTimeout> | null = null
function debouncedFetchData() {
  if (_filterTimer) clearTimeout(_filterTimer)
  _filterTimer = setTimeout(() => store.fetchData(), 300)
}

function setSeverity(value: string) {
  filters.value.severity = value
  store.fetchData()
}

function buildExportParams(): string {
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
        <h2 class="alerts-title">告警中心</h2>
      </div>
      <div class="alerts-stats">
        <span>活跃 <b>{{ String(activeCount).padStart(2, '0') }}</b></span>
        <span>已解决 <b>{{ String(resolvedCount).padStart(2, '0') }}</b></span>
        <span>总计 <b>{{ String(totalAlerts).padStart(2, '0') }}</b></span>
      </div>
    </div>
    <div class="alerts-filters">
      <Select
        v-model:value="filters.camera_id"
        placeholder="全部摄像头"
        allow-clear
        size="small"
        style="width: 130px"
        @change="debouncedFetchData"
      >
        <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
          {{ cam.camera_id }}
        </Select.Option>
      </Select>
      <div class="alerts-chip-group">
        <button :class="['alerts-chip', { on: !filters.severity }]" @click="setSeverity('')">全部</button>
        <button :class="['alerts-chip', { on: filters.severity === 'high' }]" @click="setSeverity('high')">高</button>
        <button :class="['alerts-chip', { on: filters.severity === 'medium' }]" @click="setSeverity('medium')">中</button>
        <button :class="['alerts-chip', { on: filters.severity === 'low' }]" @click="setSeverity('low')">低</button>
      </div>
      <div class="alerts-chip-group alerts-chip-group--end">
        <button class="alerts-chip alerts-chip--export" @click="handleExportCSV">CSV ↓</button>
        <button class="alerts-chip alerts-chip--export" @click="handleExportPDF">PDF ↓</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.alerts-header {
  padding: 18px 20px 0;
  flex-shrink: 0;
  border-bottom: 1px solid var(--argus-border);
}
.alerts-header-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 14px;
}
.alerts-title {
  font-size: 22px;
  font-weight: 700;
  margin: 0;
  color: var(--argus-text);
}
.alerts-stats {
  display: flex;
  gap: 16px;
  font-size: 13px;
  color: var(--argus-text-muted);
}
.alerts-stats b {
  color: var(--argus-text);
  margin-left: 4px;
}

.alerts-filters {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 0;
}
.alerts-chip-group {
  display: flex;
  gap: 4px;
}
.alerts-chip-group--end {
  margin-left: auto;
}
.alerts-chip {
  padding: 4px 14px;
  border: 1px solid var(--argus-border);
  background: transparent;
  color: var(--argus-text-muted);
  font-size: 13px;
  cursor: pointer;
  transition: all .15s;
  border-radius: 4px;
}
.alerts-chip:hover {
  border-color: #3b82f6;
  color: var(--argus-text);
}
.alerts-chip.on {
  border-color: #3b82f6;
  color: #3b82f6;
  background: rgba(59, 130, 246, .08);
}
.alerts-chip--export {
  border-radius: 4px;
  background: var(--argus-card-bg-solid, rgba(255,255,255,0.6));
}
</style>
