<script setup lang="ts">
import { inject, computed } from 'vue'
import SignalTrack from '../SignalTrack.vue'
import type { useReplayController } from '../../composables/useReplayController'

const ctrl = inject<ReturnType<typeof useReplayController>>('replayCtrl')!

const signals = ctrl.signals

const simplexData = computed(() => signals.value?.simplex_scores?.map((s: any) => s ?? 0) || [])
const hasSimplexData = computed(() => signals.value?.simplex_scores?.some((s: any) => s != null) || false)
const yoloPersonsData = computed(() => signals.value?.yolo_persons?.map((p: any) => p.count || 0) || [])
const keyFrames = computed(() => signals.value?.key_frames || [])

function getOperatorActionLeft(action: any) {
  if (!signals.value?.timestamps?.length) return '0%'
  const t0 = signals.value.timestamps[0]
  const tn = signals.value.timestamps[signals.value.timestamps.length - 1]
  const pct = ((action.timestamp - t0) / (tn - t0)) * 100
  return `${pct}%`
}
</script>

<template>
  <div>
    <!-- Signal tracks -->
    <div v-if="signals.value && ctrl.metadata.value.severity !== 'low' && ctrl.metadata.value.severity !== 'info'" class="replay-signals">
      <div v-if="signals.value.anomaly_scores" class="sig-wrap" style="border-left-color: #ef4444">
        <SignalTrack
          :data="signals.value.anomaly_scores"
          :current-index="ctrl.currentIndex.value"
          label="Dinomaly"
          color="#ef4444"
          :height="32"
          @seek="ctrl.seekTo"
        />
      </div>
      
      <div v-if="hasSimplexData" class="sig-wrap" style="border-left-color: #f97316">
        <SignalTrack
          :data="simplexData"
          :current-index="ctrl.currentIndex.value"
          label="Simplex"
          color="#f97316"
          :height="32"
          @seek="ctrl.seekTo"
        />
      </div>
      
      <template v-for="(values, zone) in (signals.value.cusum_evidence || {})" :key="zone">
        <div class="sig-wrap" style="border-left-color: #8b5cf6">
          <SignalTrack
            :data="values"
            :current-index="ctrl.currentIndex.value"
            :label="'CUSUM'"
            color="#8b5cf6"
            :height="32"
            @seek="ctrl.seekTo"
          />
        </div>
      </template>

      <div v-if="signals.value.yolo_persons" class="sig-wrap" style="border-left-color: #10b981">
        <SignalTrack
          :data="yoloPersonsData"
          :current-index="ctrl.currentIndex.value"
          label="YOLO人员"
          color="#10b981"
          :height="24"
          @seek="ctrl.seekTo"
        />
      </div>

      <!-- Operator action track -->
      <div v-if="signals.value.operator_actions?.length" class="sig-wrap" style="border-left-color: #f59e0b; height: 20px; position: relative; overflow: hidden">
        <span class="operator-label">操作员</span>
        <div
          v-for="(action, idx) in signals.value.operator_actions"
          :key="idx"
          class="operator-action-mark"
          :style="{ left: getOperatorActionLeft(action) }"
          :title="`${action.user}: ${action.action}`"
        />
      </div>
    </div>

    <!-- Key frames -->
    <div v-if="keyFrames.length > 0" class="replay-keyframes">
      <span class="kf-label">关键帧</span>
      <button
        v-for="kf in keyFrames" :key="kf.index"
        :class="['kf-tag', kf.type === 'trigger' ? 'kf-purple' : '']"
        @click="ctrl.seekTo(kf.index)"
      >{{ kf.label }} · #{{ kf.index }}</button>
      <button class="kf-tag kf-add" @click="ctrl.handlePinFrame">+ 标记当前帧</button>
    </div>
  </div>
</template>

<style scoped>
.replay-signals {
  background: var(--bg);
  padding: 4px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.sig-wrap {
  border-left: 3px solid;
  background: var(--glass);
  border-radius: 0 4px 4px 0;
  overflow: hidden;
}
.operator-label {
  font-size: 10px;
  position: absolute;
  left: 8px; top: 2px;
  color: var(--ink-4);
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: .1em;
}
.operator-action-mark {
  position: absolute;
  top: 2px;
  width: 8px;
  height: 16px;
  background: #f59e0b;
  border-radius: 2px;
  cursor: pointer;
}

/* Keyframes */
.replay-keyframes {
  padding: 8px 12px;
  background: var(--bg);
  border-top: 1px solid var(--line-2);
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.kf-label {
  font-size: 10px;
  color: var(--ink-4);
  text-transform: uppercase;
  margin-right: 4px;
}
.kf-tag {
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 10px;
  font-family: 'JetBrains Mono', monospace;
  background: var(--line-2);
  color: var(--ink-2);
  border: none;
  cursor: pointer;
}
.kf-tag:hover { background: var(--line-3); }
.kf-purple {
  background: rgba(139,92,246,.15);
  color: #8b5cf6;
  border: 1px solid rgba(139,92,246,.3);
}
.kf-add {
  background: transparent;
  border: 1px dashed var(--line-3);
  color: var(--ink-4);
}
.kf-add:hover {
  color: #3b82f6;
  border-color: #3b82f6;
}
</style>
