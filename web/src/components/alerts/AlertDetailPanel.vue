<script setup lang="ts">
import { ref, computed } from 'vue'
import { storeToRefs } from 'pinia'
import {
  Tag, Button, Typography, Tooltip, Segmented, message, Modal,
} from 'ant-design-vue'
import {
  CloseOutlined,
  CheckCircleOutlined,
  StopOutlined,
  ExportOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  AppstoreOutlined,
} from '@ant-design/icons-vue'
import { useRouter } from 'vue-router'
import { useAlertStore } from '../../stores/useAlertStore'
import { scoreColor } from '../../utils/colors'
import { formatTimestamp } from '../../utils/time'
import ReplayPlayer from '../ReplayPlayer.vue'
import AnnotationOverlay from '../AnnotationOverlay.vue'
import ImageCompareSlider from '../ImageCompareSlider.vue'

defineOptions({ name: 'AlertDetailPanel' })

const emit = defineEmits<{
  (e: 'close'): void
}>()

const router = useRouter()
const store = useAlertStore()
const { selectedAlert } = storeToRefs(store)

const imageMode = ref<'composite' | 'snapshot' | 'heatmap' | 'compare'>('composite')
const annotationMode = ref(false)

const severityColor: Record<string, string> = {
  high: 'red', medium: 'orange', low: 'gold', info: 'blue',
}
const severityLabel: Record<string, string> = {
  high: '高', medium: '中', low: '低', info: '提示',
}
const categoryMeta: Record<string, { label: string; color: string }> = {
  projectile: { label: '抛射物', color: 'red' },
  static_foreign: { label: '静态异物', color: 'orange' },
  scene_change: { label: '场景变化', color: 'blue' },
  environmental: { label: '环境干扰', color: 'default' },
  person_intrusion: { label: '人员入侵', color: 'purple' },
  equipment_displacement: { label: '设备位移', color: 'gold' },
  unknown: { label: '未分类', color: 'default' },
}
const workflowLabel: Record<string, string> = {
  new: '待处理', acknowledged: '已确认', investigating: '调查中',
  resolved: '已解决', closed: '已关闭', false_positive: '误报', uncertain: '待定',
}
const workflowColor: Record<string, string> = {
  new: 'default', acknowledged: 'green', investigating: 'blue',
  resolved: 'cyan', closed: 'default', false_positive: 'orange', uncertain: 'gold',
}

// Pixels are hard to eyeball — kilo-pixels give a more scannable magnitude
// for the 分割结果 row. We intentionally don't convert to % of frame area,
// because the frame dimensions aren't persisted on the alert record.
function formatArea(px: number): string {
  if (px < 1000) return `${px} px`
  if (px < 1_000_000) return `${(px / 1000).toFixed(1)} K px`
  return `${(px / 1_000_000).toFixed(2)} M px`
}

// A "多机位回放" storyboard is only meaningful when there's another camera
// we can pair the alert with — either through the cross-camera correlation
// pipeline (correlation_partner) or because the event grouping has picked
// up related alerts from multiple cameras (event_group_count > 1).
const hasMultiCam = computed(() => {
  const a = selectedAlert.value
  if (!a) return false
  if (a.correlation_partner) return true
  if ((a.event_group_count ?? 0) > 1) return true
  return false
})

async function handleAcknowledge() {
  if (!selectedAlert.value) return
  try {
    await store.ackAlert(selectedAlert.value.alert_id)
    message.success('已确认')
  } catch (e: any) {
    message.error(e.response?.data?.error || '确认失败')
  }
}

async function handleFalsePositive() {
  if (!selectedAlert.value) return
  try {
    await store.fpAlert(selectedAlert.value.alert_id)
    message.success('已标记误报')
  } catch (e: any) {
    message.error(e.response?.data?.error || '标记失败')
  }
}

function handleDelete() {
  if (!selectedAlert.value) return
  const id = selectedAlert.value.alert_id
  Modal.confirm({
    title: '确认删除',
    content: '删除后无法恢复，确定要删除此条告警吗？',
    okText: '删除',
    okType: 'danger',
    cancelText: '取消',
    async onOk() {
      try {
        await store.delAlert(id)
        message.success('已删除')
        // Store clears selectedAlert on successful delete → panel collapses.
      } catch (e: any) {
        message.error(e.response?.data?.error || '删除失败')
      }
    },
  })
}
</script>

<template>
  <div v-if="selectedAlert" class="detail-panel">
    <div class="detail-header">
      <span :class="['sev-badge', `sev-${selectedAlert.severity}`]">
        {{ severityLabel[selectedAlert.severity] }}
      </span>
      <div class="detail-crumb">
        <b>{{ selectedAlert.camera_id }}</b>
        <span class="detail-crumb-sep">/</span>
        {{ selectedAlert.zone_id || 'DEFAULT' }}
        <span class="detail-crumb-sep">/</span>
        {{ formatTimestamp(selectedAlert.timestamp || selectedAlert.created_at) }}
      </div>

      <div class="detail-header-actions">
        <Tag
          :color="workflowColor[selectedAlert.workflow_status] || 'default'"
          style="margin: 0"
        >
          {{ workflowLabel[selectedAlert.workflow_status] || '待处理' }}
        </Tag>
        <Tooltip v-if="selectedAlert.has_recording" title="在独立页面查看录像回放">
          <Button
            size="small"
            type="text"
            style="color: var(--argus-text-muted)"
            @click="router.push(`/replay/${selectedAlert.alert_id}`)"
          >
            <template #icon><PlayCircleOutlined /></template>
            查看录像
          </Button>
        </Tooltip>
        <Tooltip v-if="hasMultiCam" title="在共享时间线上同步播放所有关联摄像头">
          <Button
            size="small"
            type="text"
            style="color: var(--argus-text-muted)"
            @click="router.push(`/replay/${selectedAlert.alert_id}/storyboard`)"
          >
            <template #icon><AppstoreOutlined /></template>
            多机位回放
          </Button>
        </Tooltip>
        <Tooltip title="导出证据包">
          <Button size="small" type="text" style="color: var(--argus-text-muted)">
            <template #icon><ExportOutlined /></template>
          </Button>
        </Tooltip>
        <Button size="small" type="text" style="color: var(--argus-text-muted)" @click="emit('close')">
          <template #icon><CloseOutlined /></template>
        </Button>
      </div>
    </div>

    <div class="detail-content">
      <ReplayPlayer
        v-if="selectedAlert.has_recording"
        :key="selectedAlert.alert_id"
        :alert-id="selectedAlert.alert_id"
        class="detail-section-spacing"
      />

      <div v-if="!selectedAlert.has_recording && selectedAlert.snapshot_path" class="detail-section-spacing">
        <div class="image-mode-bar">
          <Segmented
            v-model:value="imageMode"
            :options="[
              { label: '叠加图', value: 'composite' },
              { label: '原图', value: 'snapshot' },
              { label: '热力图', value: 'heatmap' },
            ]"
            size="small"
          />
          <div class="score-display-wrapper">
            <Typography.Text type="secondary" style="font-size: 11px">异常分数</Typography.Text>
            <div
              :style="{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '2px 10px',
                borderRadius: '12px',
                background: `${scoreColor(selectedAlert.anomaly_score)}22`,
                border: `1px solid ${scoreColor(selectedAlert.anomaly_score)}44`,
              }"
            >
              <span :style="{ color: scoreColor(selectedAlert.anomaly_score), fontWeight: 700, fontSize: '14px' }">
                {{ selectedAlert.anomaly_score?.toFixed(4) }}
              </span>
            </div>
          </div>
        </div>

        <div v-if="imageMode === 'compare'" class="compare-section">
          <ImageCompareSlider
            :left-src="`/api/alerts/${selectedAlert.alert_id}/image/snapshot`"
            :right-src="`/api/alerts/${selectedAlert.alert_id}/image/heatmap`"
            left-label="原始帧"
            right-label="热力图"
            :width="640"
            :height="480"
          />
          <Typography.Text type="secondary" style="font-size: 11px; margin-top: 4px; display: block">
            按住 Alt 悬停可放大区域
          </Typography.Text>
        </div>
        <div v-else class="image-viewer">
          <img
            :src="`/api/alerts/${selectedAlert.alert_id}/image/${imageMode}`"
            class="alert-detail-img"
            :alt="selectedAlert.alert_id"
          />
          <AnnotationOverlay
            v-if="annotationMode"
            :width="640"
            :height="480"
            :alert-id="selectedAlert.alert_id"
            :camera-id="selectedAlert.camera_id"
          />
        </div>
        <div class="image-action-bar">
          <Button size="small" :type="imageMode === 'compare' ? 'primary' : 'default'" @click="imageMode = imageMode === 'compare' ? 'composite' : 'compare'">
            {{ imageMode === 'compare' ? '返回普通视图' : '对比滑块' }}
          </Button>
          <Button size="small" :type="annotationMode ? 'primary' : 'default'" @click="annotationMode = !annotationMode">
            {{ annotationMode ? '关闭标注' : '标注画框' }}
          </Button>
        </div>
      </div>

      <div
        v-if="!selectedAlert.has_recording && !selectedAlert.snapshot_path"
        class="no-image-placeholder"
      >
        <Typography.Text type="secondary" style="font-size: 14px">
          无快照或录像数据
        </Typography.Text>
      </div>

      <div class="meta-grid">
        <div class="meta-panel">
          <div class="meta-panel-hd">
            <span>告警信息</span>
            <b>{{ selectedAlert.alert_id?.slice(-8) }}</b>
          </div>
          <div class="meta-panel-bd">
            <div class="meta-row">
              <span class="meta-k">摄像头</span>
              <span class="meta-v">{{ selectedAlert.camera_id }} / {{ selectedAlert.zone_id || '默认' }}</span>
            </div>
            <div class="meta-row">
              <span class="meta-k">严重度</span>
              <span class="meta-v">
                <Tag :color="severityColor[selectedAlert.severity]" style="margin: 0">{{ severityLabel[selectedAlert.severity] }}</Tag>
                <Tooltip v-if="selectedAlert.severity_adjusted_by_classifier" title="严重度已被 AI 分类器根据识别标签上调">
                  <Tag color="purple" style="margin-left: 4px; font-size: 10px">AI↑</Tag>
                </Tooltip>
              </span>
            </div>
            <div v-if="selectedAlert.category" class="meta-row">
              <span class="meta-k">分类</span>
              <span class="meta-v">
                <Tag :color="categoryMeta[selectedAlert.category]?.color || 'default'" style="margin: 0">
                  {{ categoryMeta[selectedAlert.category]?.label || selectedAlert.category }}
                </Tag>
              </span>
            </div>
            <div class="meta-row">
              <span class="meta-k">置信度</span>
              <span class="meta-v" :style="{ color: scoreColor(selectedAlert.anomaly_score), fontWeight: 600 }">{{ selectedAlert.anomaly_score?.toFixed(4) }}</span>
            </div>
            <div v-if="selectedAlert.classification_label" class="meta-row">
              <span class="meta-k">AI 分类</span>
              <span class="meta-v meta-v--classification">
                <Tag color="geekblue" style="margin: 0">{{ selectedAlert.classification_label }}</Tag>
                <span v-if="selectedAlert.classification_confidence != null" class="classification-conf">
                  {{ (selectedAlert.classification_confidence * 100).toFixed(1) }}%
                </span>
              </span>
            </div>
            <div v-if="selectedAlert.corroborated !== undefined && selectedAlert.corroborated !== null" class="meta-row">
              <span class="meta-k">
                <Tooltip title="跨相机相关性：另一个视野重叠的相机是否在同一位置确认了异常。未证实的告警严重度会被自动降级。">
                  <span>跨相机验证</span>
                </Tooltip>
              </span>
              <span class="meta-v meta-v--corroboration">
                <Tag v-if="selectedAlert.corroborated" color="green" style="margin: 0">
                  已证实
                </Tag>
                <Tag v-else color="volcano" style="margin: 0">
                  未证实 (已降级)
                </Tag>
                <span v-if="selectedAlert.correlation_partner" class="correlation-partner">
                  via {{ selectedAlert.correlation_partner }}
                </span>
              </span>
            </div>
            <div v-if="selectedAlert.segmentation_count" class="meta-row">
              <span class="meta-k">
                <Tooltip title="SAM2 实例分割 — 从异常热力图峰值反推每个物体的精确边界。主图上的红框就是分割结果。">
                  <span>分割结果</span>
                </Tooltip>
              </span>
              <span class="meta-v meta-v--segmentation">
                <Tag color="magenta" style="margin: 0">
                  {{ selectedAlert.segmentation_count }} 个对象
                </Tag>
                <span
                  v-if="selectedAlert.segmentation_total_area_px != null"
                  class="segmentation-area"
                >
                  总面积 {{ formatArea(selectedAlert.segmentation_total_area_px) }}
                </span>
              </span>
            </div>
            <div v-if="selectedAlert.speed_ms || selectedAlert.speed_px_per_sec || selectedAlert.trajectory_model" class="meta-row">
              <span class="meta-k">物理数据</span>
              <span class="meta-v">
                <Tag v-if="selectedAlert.speed_ms" color="cyan" style="margin: 0">{{ selectedAlert.speed_ms.toFixed(2) }} m/s</Tag>
                <Tag v-else-if="selectedAlert.speed_px_per_sec" color="cyan" style="margin: 0">{{ selectedAlert.speed_px_per_sec.toFixed(0) }} px/s</Tag>
                <Tag v-if="selectedAlert.trajectory_model" color="green" style="margin: 0; margin-left: 4px">{{ selectedAlert.trajectory_model }}</Tag>
              </span>
            </div>
            <div class="meta-row">
              <span class="meta-k">触发时间</span>
              <span class="meta-v">{{ formatTimestamp(selectedAlert.timestamp || selectedAlert.created_at) }}</span>
            </div>
            <div v-if="selectedAlert.assigned_to" class="meta-row">
              <span class="meta-k">指派</span>
              <span class="meta-v">{{ selectedAlert.assigned_to }}</span>
            </div>
            <div v-if="selectedAlert.notes" class="meta-row">
              <span class="meta-k">备注</span>
              <span class="meta-v meta-v--notes">{{ selectedAlert.notes }}</span>
            </div>
          </div>
        </div>

        <div class="meta-panel">
          <div class="meta-panel-hd">
            <span>操作</span>
            <b>{{ workflowLabel[selectedAlert.workflow_status] || '待处理' }}</b>
          </div>
          <div class="meta-panel-bd actions-panel-bd">
            <Button
              v-if="selectedAlert.workflow_status === 'new'"
              type="primary"
              block
              @click="handleAcknowledge"
            >
              <template #icon><CheckCircleOutlined /></template>
              确认真实
            </Button>
            <Button
              v-if="selectedAlert.workflow_status === 'new' || selectedAlert.workflow_status === 'acknowledged'"
              block
              @click="handleFalsePositive"
            >
              <template #icon><StopOutlined /></template>
              标记误报
            </Button>
            <div v-if="selectedAlert.workflow_status !== 'new' && selectedAlert.workflow_status !== 'acknowledged'" class="status-display">
              <Tag :color="workflowColor[selectedAlert.workflow_status]" style="font-size: 13px; padding: 4px 16px">
                {{ workflowLabel[selectedAlert.workflow_status] }}
              </Tag>
            </div>
            <Button
              danger
              block
              @click="handleDelete"
            >
              <template #icon><DeleteOutlined /></template>
              删除告警
            </Button>

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
  background: var(--argus-surface);
}

.detail-header {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 10px 24px;
  background: var(--argus-header-bg);
  border-bottom: 1px solid var(--argus-border);
  flex-shrink: 0;
}
.detail-crumb {
  font-size: 13px;
  color: var(--argus-text-muted);
}
.detail-crumb b {
  color: var(--argus-text);
}
.detail-crumb-sep {
  margin: 0 8px;
  color: var(--argus-text-muted);
  opacity: .5;
}
.detail-header-actions {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}

.sev-badge {
  display: inline-block;
  padding: 2px 10px;
  font-size: 12px;
  font-weight: 600;
  color: #fff;
  border-radius: 3px;
}
.sev-high { background: #b91c1c; }
.sev-medium { background: #d97706; }
.sev-low { background: #2563eb; }
.sev-info { background: #4b5563; }

.detail-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
}
.detail-section-spacing {
  margin-bottom: 20px;
}

.image-mode-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}
.score-display-wrapper {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}
.compare-section {
  margin-bottom: 12px;
}
.image-viewer {
  border-radius: 8px;
  overflow: hidden;
  background: #000;
  text-align: center;
  position: relative;
}
.alert-detail-img {
  max-width: 100%;
  max-height: 480px;
  object-fit: contain;
}
.image-action-bar {
  display: flex;
  gap: 6px;
  margin-top: 6px;
}

.no-image-placeholder {
  padding: 40px;
  text-align: center;
  background: var(--argus-card-bg-solid);
  border-radius: 8px;
  margin-bottom: 20px;
}

.meta-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 20px;
}
.meta-panel {
  border: 1px solid var(--argus-border);
  background: var(--argus-card-bg-solid);
  border-radius: 6px;
  overflow: hidden;
}
.meta-panel-hd {
  padding: 10px 14px;
  border-bottom: 1px solid var(--argus-border);
  font-size: 13px;
  font-weight: 500;
  color: var(--argus-text-muted);
  display: flex;
  justify-content: space-between;
}
.meta-panel-hd b {
  color: #f59e0b;
}
.meta-panel-bd {
  padding: 12px 14px;
}
.meta-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px solid var(--argus-border);
  font-size: 13px;
  line-height: 1.8;
}
.meta-row:last-child {
  border-bottom: none;
}
.meta-k {
  color: var(--argus-text-muted);
  font-size: 13px;
  min-width: 70px;
}
.meta-v {
  color: var(--argus-text);
  font-size: 13px;
}
.meta-v--notes {
  max-width: 180px;
  text-align: right;
}
.meta-v--classification,
.meta-v--segmentation {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.meta-v--corroboration {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.correlation-partner {
  font-size: 11px;
  color: var(--argus-text-muted);
  font-variant-numeric: tabular-nums;
}
.classification-conf,
.segmentation-area {
  font-size: 11px;
  color: var(--argus-text-muted);
  font-variant-numeric: tabular-nums;
}

.actions-panel-bd {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.status-display {
  text-align: center;
  padding: 12px 0;
}

</style>
