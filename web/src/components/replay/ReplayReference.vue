<script setup lang="ts">
import { computed, inject } from 'vue'
import { Select, Slider } from 'ant-design-vue'
import type { useReplayController } from '../../composables/useReplayController'

const ctrl = inject<ReturnType<typeof useReplayController>>('replayCtrl')!

const offsetLabel = computed(() => {
  const v = ctrl.referenceOffsetSeconds.value
  const sign = v > 0 ? '+' : ''
  return `${sign}${v.toFixed(1)}s`
})

function resetOffset() {
  ctrl.referenceOffsetSeconds.value = 0
}
</script>

<template>
  <div class="replay-ref glass-panel">
    <div class="replay-ref-label">
      <span>历史对照 · Reference</span>
      <Select
        :value="ctrl.selectedRefOption.value"
        size="small"
        class="ref-select"
        style="flex: 1; min-width: 100px"
        @change="ctrl.onRefOptionChange"
      >
        <Select.Option value="yesterday">昨天</Select.Option>
        <Select.Option value="last_week">上周</Select.Option>
        <Select.Option value="prev_week">两周前</Select.Option>
        <Select.Option value="custom">手动...</Select.Option>
      </Select>
    </div>
    <div class="replay-ref-img">
      <img v-if="ctrl.referenceFrame.value" :src="ctrl.referenceFrame.value" style="width: 100%; height: 100%; object-fit: contain; display: block" />
      <span v-else class="replay-ref-empty">{{ ctrl.loadingRef.value ? '加载中...' : '无参考数据' }}</span>
    </div>
    <div class="replay-ref-slider">
      <div class="replay-ref-slider-header">
        <span>偏移</span>
        <button
          class="replay-ref-offset-value"
          type="button"
          :disabled="ctrl.referenceOffsetSeconds.value === 0"
          @click="resetOffset"
          title="点击复位到触发时刻"
        >{{ offsetLabel }}</button>
      </div>
      <Slider
        :value="ctrl.referenceOffsetSeconds.value"
        :min="-30"
        :max="30"
        :step="0.1"
        :tooltip-visible="false"
        @change="(v: number) => (ctrl.referenceOffsetSeconds.value = v)"
      />
    </div>
    <div v-if="ctrl.referenceDate.value" class="replay-ref-date">{{ ctrl.referenceDate.value }}</div>
  </div>
</template>

<style scoped>
.replay-ref {
  width: 220px;
  flex-shrink: 0;
  border-left: 1px solid var(--line-2);
  display: flex;
  flex-direction: column;
  border-radius: 0; /* Override glass-panel default radius if needed, or keep to fit */
}
.replay-ref-label {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 10px;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 9px;
  color: var(--ink-4);
  letter-spacing: .12em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--line-2);
}
.replay-ref-img {
  flex: 1;
  background: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 120px;
}
.replay-ref-empty {
  font-size: 11px;
  color: var(--ink-4);
}
.replay-ref-date {
  padding: 4px 10px;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 10px;
  color: var(--ink-4);
  letter-spacing: .1em;
  border-top: 1px solid var(--line-2);
}

.replay-ref-slider {
  padding: 6px 10px 4px;
  border-top: 1px solid var(--line-2);
}
.replay-ref-slider-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 9px;
  color: var(--ink-4);
  letter-spacing: .12em;
  text-transform: uppercase;
  margin-bottom: 2px;
}
.replay-ref-offset-value {
  background: transparent;
  border: none;
  padding: 0;
  cursor: pointer;
  color: #3b82f6;
  font-family: inherit;
  font-size: 10px;
  letter-spacing: .08em;
}
.replay-ref-offset-value[disabled] {
  color: var(--ink-4);
  cursor: default;
}

:deep(.ref-select .ant-select-selector) {
  background: var(--bg) !important;
  border-color: var(--line-2) !important;
  color: var(--ink-2) !important;
}

:deep(.replay-ref-slider .ant-slider) {
  margin: 2px 4px;
}
</style>
