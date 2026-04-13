<script setup lang="ts">
import { ref, computed } from 'vue'
import { storeToRefs } from 'pinia'
import { Tag, Button, Typography, Tooltip, Segmented, Steps } from 'ant-design-vue'
import { CloseOutlined, ExportOutlined, CheckCircleOutlined, StopOutlined, DeleteOutlined } from '@ant-design/icons-vue'
import { useAlertStore } from '../../stores/useAlertStore'
import { scoreColor } from '../../utils/colors'
import ReplayPlayer from '../ReplayPlayer.vue'
import ImageCompareSlider from '../ImageCompareSlider.vue'
import AnnotationOverlay from '../AnnotationOverlay.vue'

const alertStore = useAlertStore()
const { selectedAlert } = storeToRefs(alertStore)

const imageMode = ref<'composite' | 'snapshot' | 'heatmap' | 'compare'>('composite')
const annotationMode = ref(false)

const emit = defineEmits<{ (e: 'close'): void }>()

function closeDetail() {
  alertStore.selectedAlert = null
  emit('close')
}

const severityColor: Record<string, string> = {
  high: 'red', medium: 'orange', low: 'gold', info: 'blue',
}
const severityLabel: Record<string, string> = {
  high: '高', medium: '中', low: '低', info: '提示',
}
const workflowLabel: Record<string, string> = {
  new: '待处理', acknowledged: '已确认', investigating: '调查中',
  resolved: '已解决', closed: '已关闭', false_positive: '误报', uncertain: '待定',
}

function formatTimestamp(ts: string | number | undefined): string {
  if (!ts) return '--'
  const date = typeof ts === 'string' ? new Date(ts) : new Date(ts * 1000)
  return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

const workflowSteps = ['new', 'acknowledged', 'investigating', 'resolved', 'closed']
const workflowStepIndex = computed(() => {
  if (!selectedAlert.value) return 0
  const status = selectedAlert.value.workflow_status
  if (status === 'false_positive') return -1
  return workflowSteps.indexOf(status)
})
</script>

<template>
  <div v-if="selectedAlert" class="detail-panel glass-panel">
    <!-- Detail Header -->
    <div class="detail-header glass-header">
      <span :class="['sev-badge', `sev-${selectedAlert.severity}`]">{{ severityLabel[selectedAlert.severity] }}</span>
      <div class="detail-crumb">
        <b>{{ selectedAlert.camera_id }}</b>
        <span class="detail-crumb-sep">/</span>
        {{ selectedAlert.zone_id || 'DEFAULT' }}
      </div>

      <div class="detail-header-actions">
        <Tag :color="severityColor[selectedAlert.severity] || 'default'" style="margin: 0; background: rgba(255,255,255,0.1); border: none; backdrop-filter: blur(4px);">
          {{ workflowLabel[selectedAlert.workflow_status] || '待处理' }}
        </Tag>
        <Tooltip title="导出证据包">
          <Button size="small" type="text" style="color: rgba(255,255,255,0.8)"><ExportOutlined /></Button>
        </Tooltip>
        <Button size="small" type="text" style="color: rgba(255,255,255,0.8)" @click="closeDetail"><CloseOutlined /></Button>
      </div>
    </div>

    <!-- Detail Content -->
    <div class="detail-content">
      <!-- Media Viewer (Edge-to-Edge) -->
      <div class="media-edge">
        <!-- Replay Player -->
        <ReplayPlayer
          v-if="selectedAlert.has_recording"
          :alert-id="selectedAlert.alert_id"
          class="edge-player"
        />

        <!-- Static snapshot fallback -->
        <div v-else-if="selectedAlert.snapshot_path" class="edge-image-viewer">
          <div class="image-mode-glass-bar">
            <Segmented
              v-model:value="imageMode"
              :options="[{ label: '叠加图', value: 'composite' }, { label: '原图', value: 'snapshot' }, { label: '热力图', value: 'heatmap' }]"
              size="small"
              class="glass-segmented"
            />
            <div class="score-glass-display">
              <span class="score-label">异常分数</span>
              <span class="score-val" :style="{ color: scoreColor(selectedAlert.anomaly_score) }">{{ selectedAlert.anomaly_score?.toFixed(4) }}</span>
            </div>
          </div>

          <div v-if="imageMode === 'compare'" class="compare-section-edge">
            <ImageCompareSlider
              :left-src="`/api/alerts/${selectedAlert.alert_id}/image/snapshot`"
              :right-src="`/api/alerts/${selectedAlert.alert_id}/image/heatmap`"
              left-label="原始帧" right-label="热力图" :width="640" :height="480"
            />
          </div>
          <div v-else class="image-viewer-edge">
            <img :src="`/api/alerts/${selectedAlert.alert_id}/image/${imageMode}`" class="alert-detail-img-edge" :alt="selectedAlert.alert_id" />
            <AnnotationOverlay v-if="annotationMode" :width="640" :height="480" :alert-id="selectedAlert.alert_id" :camera-id="selectedAlert.camera_id" />
          </div>

          <div class="image-action-glass-bar">
            <Button class="glass-btn" :type="imageMode === 'compare' ? 'primary' : 'default'" @click="imageMode = imageMode === 'compare' ? 'composite' : 'compare'">{{ imageMode === 'compare' ? '返回普通视图' : '对比滑块' }}</Button>
            <Button class="glass-btn" :type="annotationMode ? 'primary' : 'default'" @click="annotationMode = !annotationMode">{{ annotationMode ? '关闭标注' : '标注画框' }}</Button>
          </div>
        </div>

        <!-- No image placeholder -->
        <div v-else class="no-image-edge">
          <span class="no-img-text">无快照或录像数据</span>
        </div>
      </div>

      <!-- Metadata & Actions -->
      <div class="meta-section-padded">
        
        <div class="fancy-dict-title">告警详情 / DETAILS <span>#{{ selectedAlert.alert_id?.slice(-8).toUpperCase() }}</span></div>
        <div class="fancy-dict">
            <div class="dict-row"><span class="dict-k">监控点位</span><span class="dict-v">{{ selectedAlert.camera_id }} {{ selectedAlert.zone_id ? `/ ${selectedAlert.zone_id}` : '' }}</span></div>
            <div class="dict-row"><span class="dict-k">危险级别</span><span class="dict-v"><Tag :color="severityColor[selectedAlert.severity]" style="margin: 0; background: rgba(255,255,255,0.15); border: none;">{{ severityLabel[selectedAlert.severity] }}</Tag></span></div>
            <div class="dict-row"><span class="dict-k">判定置信度</span><span class="dict-v" :style="{ color: scoreColor(selectedAlert.anomaly_score), fontWeight: 600 }">{{ selectedAlert.anomaly_score?.toFixed(4) }}</span></div>
            <div class="dict-row"><span class="dict-k">触发时间</span><span class="dict-v">{{ formatTimestamp(selectedAlert.timestamp || selectedAlert.created_at) }}</span></div>
            <div v-if="selectedAlert.assigned_to" class="dict-row"><span class="dict-k">指派人员</span><span class="dict-v">{{ selectedAlert.assigned_to }}</span></div>
            <div v-if="selectedAlert.notes" class="dict-row"><span class="dict-k">备注内容</span><span class="dict-v dict-v-notes">{{ selectedAlert.notes }}</span></div>
        </div>

        <div class="action-workflow-area">
          <div class="action-buttons-modern">
            <Button v-if="selectedAlert.workflow_status === 'new'" type="primary" class="modern-btn btn-confirm" @click="alertStore.ackAlert(selectedAlert.alert_id)"><template #icon><CheckCircleOutlined /></template>确认真实</Button>
            <Button v-if="selectedAlert.workflow_status === 'new' || selectedAlert.workflow_status === 'acknowledged'" class="modern-btn btn-fp" @click="alertStore.fpAlert(selectedAlert.alert_id)"><template #icon><StopOutlined /></template>标记误报</Button>
            <Button danger class="modern-btn btn-del" @click="alertStore.delAlert(selectedAlert.alert_id); emit('close')"><template #icon><DeleteOutlined /></template>删除告警</Button>
          </div>

          <div class="workflow-modern-timeline">
            <div class="timeline-title">工作流程 / WORKFLOW</div>
            <Steps v-if="selectedAlert.workflow_status !== 'false_positive'" :current="workflowStepIndex" size="small" class="glass-steps">
              <Steps.Step title="待处理" />
              <Steps.Step title="已确认" />
              <Steps.Step title="调查中" />
              <Steps.Step title="已解决" />
              <Steps.Step title="已关闭" />
            </Steps>
            <div v-else class="fp-badge">已标记为误报</div>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<style scoped>
.detail-panel {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--glass);
  backdrop-filter: blur(20px);
}
.detail-header {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 12px 24px;
  background: rgba(0, 0, 0, 0.4);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  flex-shrink: 0;
}
.sev-badge {
  display: inline-block;
  padding: 2px 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .12em;
  color: #fff;
  text-transform: uppercase;
  border-radius: 4px;
}
.sev-high { background: rgba(185, 28, 28, 0.8); }
.sev-medium { background: rgba(217, 119, 6, 0.8); }
.sev-low { background: rgba(37, 99, 235, 0.8); }
.sev-info { background: rgba(75, 85, 99, 0.8); }
.detail-crumb {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  color: rgba(255,255,255,0.7);
  display: flex;
  align-items: center;
}
.detail-crumb b { color: #fff; font-weight: 500; }
.detail-crumb-sep { margin: 0 8px; color: rgba(255,255,255,0.3); }
.detail-header-actions { margin-left: auto; display: flex; align-items: center; gap: 8px; }

.detail-content { 
  flex: 1; 
  overflow-y: auto; 
  display: flex;
  flex-direction: column;
}

.media-edge {
  width: 100%;
  background: #000;
  position: relative;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 320px;
}
.edge-player {
  width: 100%;
  height: 100%;
}
.edge-image-viewer {
  width: 100%;
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
}
.image-viewer-edge {
  width: 100%;
  text-align: center;
  position: relative;
}
.alert-detail-img-edge {
  max-width: 100%;
  max-height: 50vh;
  object-fit: contain;
  display: block;
  margin: 0 auto;
}
.compare-section-edge {
  width: 100%;
  display: flex;
  justify-content: center;
  background: #111;
}

.image-mode-glass-bar {
  position: absolute;
  top: 16px;
  left: 16px;
  right: 16px;
  z-index: 10;
  display: flex;
  justify-content: space-between;
  pointer-events: none;
}
.image-mode-glass-bar > * { pointer-events: auto; }
.glass-segmented {
  background: rgba(0,0,0,0.6) !important;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.1);
}
.score-glass-display {
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(0,0,0,0.6);
  backdrop-filter: blur(10px);
  padding: 4px 12px;
  border-radius: 20px;
  border: 1px solid rgba(255,255,255,0.1);
}
.score-label { font-size: 11px; color: rgba(255,255,255,0.6); }
.score-val { font-weight: 700; font-size: 14px; font-family: 'JetBrains Mono', monospace; }

.image-action-glass-bar {
  position: absolute;
  bottom: 16px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 10;
  display: flex;
  gap: 8px;
  background: rgba(0,0,0,0.5);
  padding: 6px;
  border-radius: 8px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.1);
}
.glass-btn {
  background: rgba(255,255,255,0.1) !important;
  border-color: transparent !important;
  color: #fff !important;
}
.glass-btn:hover { background: rgba(255,255,255,0.2) !important; }
.glass-btn.ant-btn-primary { background: rgba(24, 144, 255, 0.6) !important; }

.no-image-edge {
  width: 100%;
  height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255,255,255,0.02);
}
.no-img-text { color: rgba(255,255,255,0.3); font-size: 14px; letter-spacing: 0.1em; }

.meta-section-padded {
  padding: 24px 32px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.fancy-dict-title {
  font-size: 12px;
  color: rgba(255,255,255,0.5);
  margin-bottom: 12px;
  letter-spacing: 0.05em;
  display: flex;
  justify-content: space-between;
}
.fancy-dict-title span { font-family: 'JetBrains Mono', monospace; color: rgba(255,255,255,0.3); }

.fancy-dict {
  display: flex;
  flex-direction: column;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.05);
  padding: 8px 16px;
}
.dict-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px dashed rgba(255, 255, 255, 0.1);
  font-size: 13px;
}
.dict-row:last-child { border-bottom: none; }
.dict-k { color: rgba(255,255,255,0.5); }
.dict-v { color: rgba(255,255,255,0.9); font-weight: 500; font-family: 'JetBrains Mono', monospace; }
.dict-v-notes { font-family: inherit; font-weight: 400; max-width: 300px; text-align: right; line-height: 1.5; color: rgba(255,255,255,0.7); }

.action-workflow-area {
  display: grid;
  grid-template-columns: 240px 1fr;
  gap: 24px;
}

.action-buttons-modern {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.modern-btn {
  height: 40px;
  border-radius: 20px;
  font-weight: 500;
  letter-spacing: 0.05em;
  border: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.btn-confirm { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: #fff; }
.btn-confirm:hover { background: linear-gradient(135deg, #34d399 0%, #10b981 100%); }
.btn-fp { background: rgba(255,255,255,0.1); color: #fff; backdrop-filter: blur(10px); }
.btn-fp:hover { background: rgba(255,255,255,0.15); }
.btn-del { background: rgba(220, 38, 38, 0.2); color: #fca5a5; backdrop-filter: blur(10px); }
.btn-del:hover { background: rgba(220, 38, 38, 0.3); color: #fff; }

.workflow-modern-timeline {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 12px;
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.timeline-title {
  font-size: 11px;
  color: rgba(255,255,255,0.4);
  margin-bottom: 20px;
  letter-spacing: 0.1em;
}
:deep(.ant-steps-item-title) { color: rgba(255,255,255,0.6) !important; font-size: 12px !important; }
:deep(.ant-steps-item-finish .ant-steps-item-title) { color: rgba(255,255,255,0.9) !important; }
:deep(.ant-steps-item-active .ant-steps-item-title) { color: #1890ff !important; font-weight: 600 !important; }
:deep(.ant-steps-item-wait .ant-steps-item-icon) { background-color: rgba(255,255,255,0.1) !important; border-color: transparent !important; }
:deep(.ant-steps-item-wait .ant-steps-item-icon > .ant-steps-icon) { color: rgba(255,255,255,0.3) !important; }
:deep(.ant-steps-item-process .ant-steps-item-icon) { background-color: transparent !important; border-color: #1890ff !important; }

.fp-badge {
  align-self: flex-start;
  padding: 6px 16px;
  background: rgba(245, 158, 11, 0.2);
  color: #fbbf24;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 500;
  border: 1px solid rgba(245, 158, 11, 0.3);
}
</style>
