<script setup lang="ts">
import { inject, computed } from 'vue'
import SignalTrack from '../SignalTrack.vue'
import type {
  ReplaySignalsState,
  useReplayController,
} from '../../composables/useReplayController'

interface CandidateEvent extends Record<string, unknown> {
  frame_index?: unknown
  candidate_id?: unknown
  candidate_type?: unknown
  decision?: unknown
  suppression_reason?: unknown
  evidence?: unknown
}

type OperatorAction = NonNullable<ReplaySignalsState['operator_actions']>[number]

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

const ctrl = inject<ReturnType<typeof useReplayController>>('replayCtrl')!

const signals = ctrl.signals

const simplexData = computed(() => signals.value?.simplex_scores?.map((score) => score ?? 0) || [])
const hasSimplexData = computed(() => signals.value?.simplex_scores?.some((score) => score != null) || false)
const yoloPersonsData = computed(() => signals.value?.yolo_persons?.map((person) => person.count || 0) || [])
const keyFrames = computed(() => signals.value?.key_frames || [])
const candidateEvents = computed<CandidateEvent[]>(() => {
  const flat = signals.value?.candidate_events
  if (Array.isArray(flat) && flat.length > 0) return flat.filter(isRecord)
  const perFrame = signals.value?.candidate_records
  if (!Array.isArray(perFrame)) return []
  return perFrame.flatMap((records, frameIndex) => {
    if (!Array.isArray(records)) return []
    return records.flatMap((record) => (
      isRecord(record) ? [{ ...record, frame_index: frameIndex }] : []
    ))
  })
})
const nearbyCandidateEvents = computed(() => {
  const current = ctrl.currentIndex.value
  return candidateEvents.value
    .filter((event) => Math.abs(Number(event.frame_index ?? -999999) - current) <= 2)
    .slice(-6)
})

function candidateTypeLabel(value: unknown) {
  if (value === 'persistent_change') return '固定场景变化'
  if (value === 'frame_anomaly') return '帧异常'
  if (value === 'fast_motion') return '高速飞溅'
  if (value === 'stationary_track') return '静止轨迹'
  if (value === 'frame_change') return '画面变化'
  return String(value || '-')
}

function decisionLabel(value: unknown) {
  if (value === 'alert') return '告警'
  if (value === 'candidate') return '候选'
  if (value === 'info') return '信息'
  if (value === 'suppressed') return '已抑制'
  if (value === 'skipped') return '已跳过'
  if (value === 'pending') return '等待确认'
  if (value === 'normal') return '正常'
  return String(value || '-')
}

function candidateScore(event: CandidateEvent) {
  const evidence = isRecord(event.evidence) ? event.evidence : {}
  const persistentEvidence = isRecord(evidence.persistent_change)
    ? evidence.persistent_change
    : null
  const persistent = persistentEvidence
    || (event.candidate_type === 'persistent_change' ? evidence : null)
  const score = persistent?.score ?? evidence.score ?? evidence.anomaly_score
  return typeof score === 'number' && Number.isFinite(score) ? score.toFixed(3) : '-'
}

function candidateReason(event: CandidateEvent): string {
  const evidence = isRecord(event.evidence) ? event.evidence : {}
  const reason = event.suppression_reason || evidence.suppression_reason || ''
  return typeof reason === 'string' ? reason : String(reason)
}

function getOperatorActionLeft(action: OperatorAction) {
  if (!signals.value?.timestamps || signals.value.timestamps.length < 2) return '0%'
  const t0 = signals.value.timestamps[0]
  const tn = signals.value.timestamps[signals.value.timestamps.length - 1]
  const duration = tn - t0
  if (!Number.isFinite(duration) || duration <= 0) return '0%'
  const pct = Math.max(0, Math.min(100, ((action.timestamp - t0) / duration) * 100))
  return `${pct}%`
}
</script>

<template>
  <div>
    <!-- Signal tracks -->
    <div v-if="signals && ctrl.metadata.value?.severity !== 'low' && ctrl.metadata.value?.severity !== 'info'" class="replay-signals">
      <div v-if="signals.anomaly_scores" class="sig-wrap" style="border-left-color: #ef4444">
        <SignalTrack
          :data="signals.anomaly_scores"
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
      
      <template v-for="(values, zone) in (signals.cusum_evidence || {})" :key="zone">
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

      <div v-if="signals.yolo_persons" class="sig-wrap" style="border-left-color: #10b981">
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
      <div v-if="signals.operator_actions?.length" class="sig-wrap" style="border-left-color: #f59e0b; height: 20px; position: relative; overflow: hidden">
        <span class="operator-label">操作员</span>
        <div
          v-for="(action, idx) in signals.operator_actions"
          :key="idx"
          class="operator-action-mark"
          :style="{ left: getOperatorActionLeft(action) }"
          :title="`${action.user}: ${action.action}`"
        />
      </div>
    </div>

    <div v-if="candidateEvents.length > 0" class="replay-candidates" data-test="replay-candidates">
      <div class="candidate-header">
        <span>候选证据</span>
        <small>当前帧 ±2</small>
      </div>
      <div v-if="nearbyCandidateEvents.length === 0" class="candidate-empty">当前帧附近无候选记录</div>
      <button
        v-for="event in nearbyCandidateEvents"
        :key="`${event.frame_index}:${event.candidate_id || event.candidate_type}`"
        type="button"
        class="candidate-event"
        @click="ctrl.seekTo(Number(event.frame_index || 0))"
      >
        <span class="candidate-frame">#{{ event.frame_index }}</span>
        <span class="candidate-main">
          <b>{{ candidateTypeLabel(event.candidate_type) }}</b>
          <span>{{ decisionLabel(event.decision) }}</span>
          <em v-if="candidateReason(event)">{{ candidateReason(event) }}</em>
        </span>
        <span class="candidate-score">{{ candidateScore(event) }}</span>
      </button>
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

.replay-candidates {
  padding: 8px 12px;
  background: var(--bg);
  border-top: 1px solid var(--line-2);
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.candidate-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  color: var(--ink-3);
  font-size: 11px;
  font-weight: 700;
}
.candidate-header small {
  color: var(--ink-5);
  font-weight: 600;
}
.candidate-empty {
  color: var(--ink-5);
  font-size: 11px;
}
.candidate-event {
  border: 1px solid var(--line-2);
  background: var(--glass);
  border-radius: 6px;
  color: var(--ink-3);
  display: grid;
  grid-template-columns: 44px minmax(0, 1fr) 52px;
  gap: 8px;
  align-items: center;
  min-height: 32px;
  padding: 5px 8px;
  text-align: left;
  cursor: pointer;
}
.candidate-event:hover {
  border-color: #3b82f6;
}
.candidate-frame,
.candidate-score {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--ink-4);
}
.candidate-score {
  text-align: right;
}
.candidate-main {
  min-width: 0;
  display: flex;
  align-items: baseline;
  gap: 6px;
  overflow-wrap: anywhere;
}
.candidate-main b {
  color: var(--ink-2);
  font-size: 11.5px;
}
.candidate-main span,
.candidate-main em {
  color: var(--ink-4);
  font-size: 11px;
  font-style: normal;
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
