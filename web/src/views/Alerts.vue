<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'

defineOptions({ name: 'AlertsPage' })
import {
  Table, Tag, Button, Space, Typography, Select, Tooltip,
  message, Segmented, Steps, Modal, Popover, List, Spin,
} from 'ant-design-vue'
import {
  CloseOutlined,
  CheckCircleOutlined,
  StopOutlined,
  ExportOutlined,
  DeleteOutlined,
} from '@ant-design/icons-vue'
import { getAlerts, getCameras, acknowledgeAlert, markFalsePositive, deleteAlert, bulkDeleteAlerts, bulkAcknowledge, bulkFalsePositive, getAlertGroup } from '../api'
import { formatRelativeTime } from '../utils/time'
import { useWebSocket } from '../composables/useWebSocket'
import ReplayPlayer from '../components/ReplayPlayer.vue'
import AnnotationOverlay from '../components/AnnotationOverlay.vue'
import ImageCompareSlider from '../components/ImageCompareSlider.vue'

const route = useRoute()
const router = useRouter()

const alerts = ref<any[]>([])
const cameras = ref<any[]>([])
const totalAlerts = ref(0)
const loading = ref(true)
const filters = ref({ camera_id: '', severity: '' })

// Detail panel state
const selectedAlert = ref<any>(null)
const imageMode = ref<'composite' | 'snapshot' | 'heatmap' | 'compare'>('composite')
const annotationMode = ref(false)

// Event group expansion
const groupAlerts = ref<any[]>([])
const groupLoading = ref(false)
const groupPopoverVisible = ref<Record<string, boolean>>({})

async function loadGroupAlerts(eventGroupId: string) {
  groupLoading.value = true
  try {
    const data = await getAlertGroup(eventGroupId)
    groupAlerts.value = data.alerts || []
  } catch {
    groupAlerts.value = []
  } finally {
    groupLoading.value = false
  }
}

function toggleGroupPopover(eventGroupId: string) {
  const isOpen = groupPopoverVisible.value[eventGroupId]
  // Close all others
  groupPopoverVisible.value = {}
  if (!isOpen) {
    groupPopoverVisible.value[eventGroupId] = true
    loadGroupAlerts(eventGroupId)
  }
}

// Bulk selection
const selectedRowKeys = ref<(string | number)[]>([])

async function fetchData() {
  try {
    const params: Record<string, any> = { limit: 100 }
    if (filters.value.camera_id) params.camera_id = filters.value.camera_id
    if (filters.value.severity) params.severity = filters.value.severity
    const [a, c] = await Promise.all([getAlerts(params), getCameras()])
    alerts.value = a.alerts
    totalAlerts.value = a.total ?? a.alerts.length
    cameras.value = c.cameras || []
  } finally {
    loading.value = false
  }
}

let _filterTimer: ReturnType<typeof setTimeout> | null = null
function debouncedFetchData() {
  if (_filterTimer) clearTimeout(_filterTimer)
  _filterTimer = setTimeout(fetchData, 300)
}

useWebSocket({
  topics: ['alerts'],
  onMessage(topic, data) {
    if (topic === 'alerts') {
      if (data && typeof data === 'object' && !Array.isArray(data) && data.alert_id) {
        const idx = alerts.value.findIndex((a: any) => a.alert_id === data.alert_id)
        if (idx >= 0) {
          alerts.value[idx] = { ...alerts.value[idx], ...data }
          // Update detail if viewing this alert
          if (selectedAlert.value?.alert_id === data.alert_id) {
            selectedAlert.value = { ...selectedAlert.value, ...data }
          }
        } else {
          alerts.value.unshift(data)
        }
      } else if (Array.isArray(data)) {
        alerts.value = data
      }
    }
  },
  fallbackPoll: fetchData,
  fallbackInterval: 15000,
})

onMounted(async () => {
  await fetchData()
  // Support URL query: ?id=xxx to auto-open alert detail
  if (route.query.id) {
    const target = alerts.value.find(a => a.alert_id === route.query.id)
    if (target) showDetail(target)
  }
})

// Keyboard navigation
const selectedIndex = ref(-1)

function handleKeydown(e: KeyboardEvent) {
  if (!alerts.value.length) return
  if (e.key === 'ArrowDown') {
    e.preventDefault()
    selectedIndex.value = Math.min(selectedIndex.value + 1, alerts.value.length - 1)
    showDetail(alerts.value[selectedIndex.value])
  } else if (e.key === 'ArrowUp') {
    e.preventDefault()
    selectedIndex.value = Math.max(selectedIndex.value - 1, 0)
    showDetail(alerts.value[selectedIndex.value])
  } else if (e.key === 'Escape') {
    closeDetail()
  }
}

onMounted(() => document.addEventListener('keydown', handleKeydown))
onUnmounted(() => document.removeEventListener('keydown', handleKeydown))

function showDetail(record: any) {
  selectedAlert.value = record
  selectedIndex.value = alerts.value.findIndex(a => a.alert_id === record.alert_id)
  imageMode.value = 'composite'
}

function closeDetail() {
  selectedAlert.value = null
  selectedIndex.value = -1
  // Clean up URL query
  if (route.query.id) {
    router.replace({ query: {} })
  }
}

async function handleAcknowledge(id: string) {
  try {
    await acknowledgeAlert(id)
    message.success('已确认')
    fetchData()
    if (selectedAlert.value?.alert_id === id) {
      selectedAlert.value = { ...selectedAlert.value, workflow_status: 'acknowledged' }
    }
  } catch (e: any) {
    message.error(e.response?.data?.error || '确认失败')
  }
}

async function handleFalsePositive(id: string) {
  try {
    await markFalsePositive(id)
    message.success('已标记误报')
    fetchData()
    if (selectedAlert.value?.alert_id === id) {
      selectedAlert.value = { ...selectedAlert.value, workflow_status: 'false_positive' }
    }
  } catch (e: any) {
    message.error(e.response?.data?.error || '标记失败')
  }
}

function handleDelete(id: string) {
  Modal.confirm({
    title: '确认删除',
    content: '删除后无法恢复，确定要删除此条告警吗？',
    okText: '删除',
    okType: 'danger',
    cancelText: '取消',
    async onOk() {
      try {
        await deleteAlert(id)
        message.success('已删除')
        if (selectedAlert.value?.alert_id === id) {
          closeDetail()
        }
        fetchData()
      } catch (e: any) {
        message.error(e.response?.data?.error || '删除失败')
      }
    },
  })
}

// ── Bulk actions ──
function handleBulkDelete() {
  if (!selectedRowKeys.value.length) return
  Modal.confirm({
    title: '批量删除',
    content: `确定要删除选中的 ${selectedRowKeys.value.length} 条告警吗？删除后无法恢复。`,
    okText: '删除',
    okType: 'danger',
    cancelText: '取消',
    async onOk() {
      try {
        const res = await bulkDeleteAlerts(selectedRowKeys.value as string[])
        message.success(res.message || `已删除 ${res.count} 条`)
        selectedRowKeys.value = []
        if (selectedAlert.value && !alerts.value.some(a => a.alert_id === selectedAlert.value.alert_id)) {
          closeDetail()
        }
        fetchData()
      } catch (e: any) {
        message.error(e.response?.data?.error || '批量删除失败')
      }
    },
  })
}

async function handleBulkAcknowledge() {
  if (!selectedRowKeys.value.length) return
  try {
    const res = await bulkAcknowledge(selectedRowKeys.value as string[])
    message.success(res.message || `已确认 ${res.count} 条`)
    selectedRowKeys.value = []
    fetchData()
  } catch (e: any) {
    message.error(e.response?.data?.error || '批量确认失败')
  }
}

async function handleBulkFalsePositive() {
  if (!selectedRowKeys.value.length) return
  try {
    const res = await bulkFalsePositive(selectedRowKeys.value as string[])
    message.success(res.message || `已标记 ${res.count} 条为误报`)
    selectedRowKeys.value = []
    fetchData()
  } catch (e: any) {
    message.error(e.response?.data?.error || '批量标记失败')
  }
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
const workflowColor: Record<string, string> = {
  new: 'default', acknowledged: 'green', investigating: 'blue',
  resolved: 'cyan', closed: 'default', false_positive: 'orange', uncertain: 'gold',
}

// Header stats (computed to avoid re-filtering on every render)
const activeCount = computed(() => alerts.value.filter(a => a.workflow_status === 'new').length)
const resolvedCount = computed(() => alerts.value.filter(a => ['resolved', 'closed'].includes(a.workflow_status)).length)

// Workflow steps for timeline
const workflowSteps = ['new', 'acknowledged', 'investigating', 'resolved', 'closed']
const workflowStepIndex = computed(() => {
  if (!selectedAlert.value) return 0
  const status = selectedAlert.value.workflow_status
  if (status === 'false_positive') return -1
  return workflowSteps.indexOf(status)
})


function formatTimestamp(ts: string | number | undefined): string {
  if (!ts) return '--'
  const date = typeof ts === 'string' ? new Date(ts) : new Date(ts * 1000)
  return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

// Row class for severity highlighting
function rowClassName(record: any) {
  if (record.alert_id === selectedAlert.value?.alert_id) return 'alert-row-selected'
  if (record.severity === 'high' && record.workflow_status === 'new') return 'alert-row-high'
  if (record.severity === 'medium' && record.workflow_status === 'new') return 'alert-row-medium'
  return ''
}

// Anomaly score color
import { scoreColor } from '../utils/colors'

// Table columns — adapt based on whether detail is open
const columns = computed(() => {
  const isCompact = !!selectedAlert.value
  const base: any[] = [
    { title: '', key: 'thumbnail', width: 52 },
    {
      title: '严重度',
      key: 'severity',
      width: isCompact ? 60 : 72,
      filters: [
        { text: '高', value: 'high' },
        { text: '中', value: 'medium' },
        { text: '低', value: 'low' },
        { text: '提示', value: 'info' },
      ],
    },
    { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id', width: isCompact ? 80 : 100, ellipsis: true },
    { title: '分数', key: 'score', width: 64 },
    { title: '时间', key: 'time', width: 80 },
  ]
  if (!isCompact) {
    base.splice(3, 0, { title: '区域', dataIndex: 'zone_id', key: 'zone_id', width: 80, ellipsis: true })
    base.push(
      { title: '状态', key: 'status', width: 90 },
      { title: '操作', key: 'action', width: 160 },
    )
  } else {
    base.push({ title: '状态', key: 'status', width: 70 })
  }
  return base
})
</script>

<template>
  <main class="alerts-layout glass">
    <!-- Left: Alert List -->
    <div
      :style="{
        width: selectedAlert ? '520px' : '100%',
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        transition: 'width 0.25s ease',
        borderRight: selectedAlert ? '1px solid var(--argus-sidebar-border)' : 'none',
        overflow: 'hidden',
      }"
    >
      <!-- Header -->
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
            <button :class="['alerts-chip', { on: !filters.severity }]" @click="filters.severity = ''; fetchData()">全部</button>
            <button :class="['alerts-chip', { on: filters.severity === 'high' }]" @click="filters.severity = 'high'; fetchData()">高</button>
            <button :class="['alerts-chip', { on: filters.severity === 'medium' }]" @click="filters.severity = 'medium'; fetchData()">中</button>
            <button :class="['alerts-chip', { on: filters.severity === 'low' }]" @click="filters.severity = 'low'; fetchData()">低</button>
          </div>
          <div class="alerts-chip-group alerts-chip-group--end">
            <button class="alerts-chip alerts-chip--export" @click="handleExportCSV">CSV ↓</button>
            <button class="alerts-chip alerts-chip--export" @click="handleExportPDF">PDF ↓</button>
          </div>
        </div>
      </div>

      <!-- Table -->
      <div class="alert-list-body">
        <!-- Bulk action bar -->
        <div
          v-if="selectedRowKeys.length > 0"
          class="bulk-action-bar"
        >
          <Typography.Text style="font-size: 12px">
            已选 {{ selectedRowKeys.length }} 条
          </Typography.Text>
          <Button size="small" type="primary" @click="handleBulkAcknowledge">
            <template #icon><CheckCircleOutlined /></template>
            批量确认
          </Button>
          <Button size="small" @click="handleBulkFalsePositive">
            <template #icon><StopOutlined /></template>
            批量误报
          </Button>
          <Button size="small" danger @click="handleBulkDelete">
            <template #icon><DeleteOutlined /></template>
            批量删除
          </Button>
          <Button size="small" type="text" style="color: #9ca3af" @click="selectedRowKeys = []">
            取消选择
          </Button>
        </div>

        <Table
          :columns="columns"
          :data-source="alerts"
          :loading="loading"
          row-key="alert_id"
          size="small"
          :row-selection="{ selectedRowKeys, onChange: (keys: (string | number)[]) => { selectedRowKeys = keys } }"
          :pagination="selectedAlert ? { pageSize: 50, simple: true, size: 'small' } : { pageSize: 20, total: totalAlerts, showTotal: (t: number) => `共 ${t} 条` }"
          :custom-row="(record: any) => ({
            onClick: () => showDetail(record),
            class: rowClassName(record),
          })"
          :scroll="{ x: 800, y: selectedAlert ? 'calc(100vh - 160px)' : undefined }"
          style="cursor: pointer"
        >
          <template #bodyCell="{ column, record }">
            <!-- Thumbnail -->
            <template v-if="column.key === 'thumbnail'">
              <div class="alert-thumb">
                <img
                  v-if="record.snapshot_path"
                  :src="`/api/alerts/${record.alert_id}/image/snapshot`"
                  class="alert-thumb-img"
                  loading="lazy"
                />
                <div v-else class="alert-thumb-empty">--</div>
              </div>
            </template>
            <!-- Severity -->
            <template v-if="column.key === 'severity'">
              <Space :size="4">
                <Tag :color="severityColor[record.severity]" style="margin: 0">
                  {{ severityLabel[record.severity] || record.severity }}
                </Tag>
                <Popover
                  v-if="record.event_group_count > 1 && record.event_group_id"
                  :open="groupPopoverVisible[record.event_group_id]"
                  trigger="click"
                  placement="right"
                  :destroy-tooltip-on-hide="true"
                  @openChange="(v: boolean) => { if (!v) groupPopoverVisible[record.event_group_id] = false }"
                >
                  <template #content>
                    <div class="group-popover-content">
                      <Typography.Text strong style="margin-bottom: 8px; display: block">
                        事件组 ({{ groupAlerts.length }} 条)
                      </Typography.Text>
                      <Spin v-if="groupLoading" size="small" />
                      <List v-else :data-source="groupAlerts" size="small" :split="true">
                        <template #renderItem="{ item }">
                          <List.Item style="padding: 4px 0; cursor: pointer" @click="showDetail(item); groupPopoverVisible = {}">
                            <Space :size="8">
                              <Tag :color="severityColor[item.severity]" style="margin: 0; font-size: 10px">
                                {{ severityLabel[item.severity] }}
                              </Tag>
                              <Typography.Text style="font-size: 12px">
                                {{ item.anomaly_score?.toFixed(3) }}
                              </Typography.Text>
                              <Typography.Text type="secondary" style="font-size: 11px">
                                {{ formatRelativeTime(item.timestamp) }}
                              </Typography.Text>
                            </Space>
                          </List.Item>
                        </template>
                      </List>
                    </div>
                  </template>
                  <Tag
                    color="default"
                    style="margin: 0; font-size: 10px; cursor: pointer"
                    @click.stop="toggleGroupPopover(record.event_group_id)"
                  >
                    x{{ record.event_group_count }}
                  </Tag>
                </Popover>
              </Space>
            </template>
            <!-- Score -->
            <template v-if="column.key === 'score'">
              <span class="alert-score" :style="{ color: scoreColor(record.anomaly_score) }">
                {{ record.anomaly_score?.toFixed(2) }}
              </span>
            </template>
            <!-- Time -->
            <template v-if="column.key === 'time'">
              <Tooltip :title="formatTimestamp(record.timestamp || record.created_at)">
                <span class="alert-time">
                  {{ formatRelativeTime(record.timestamp || record.created_at) }}
                </span>
              </Tooltip>
            </template>
            <!-- Status -->
            <template v-if="column.key === 'status'">
              <Tag :color="workflowColor[record.workflow_status] || 'default'" style="margin: 0; font-size: 11px">
                {{ workflowLabel[record.workflow_status] || record.workflow_status || '待处理' }}
              </Tag>
            </template>
            <!-- Action (only when detail is closed) -->
            <template v-if="column.key === 'action'">
              <Space size="small" @click.stop>
                <Button
                  v-if="record.workflow_status === 'new'"
                  type="primary"
                  size="small"
                  @click="handleAcknowledge(record.alert_id)"
                >确认</Button>
                <Button
                  v-if="record.workflow_status === 'new' || record.workflow_status === 'acknowledged'"
                  size="small"
                  @click="handleFalsePositive(record.alert_id)"
                >误报</Button>
              </Space>
            </template>
          </template>
        </Table>
      </div>
    </div>

    <!-- Right: Detail Panel -->
    <div
      v-if="selectedAlert"
      class="detail-panel"
    >
      <!-- Detail Header -->
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
          <Tooltip title="导出证据包">
            <Button size="small" type="text" style="color: var(--argus-text-muted)">
              <template #icon><ExportOutlined /></template>
            </Button>
          </Tooltip>
          <Button size="small" type="text" style="color: var(--argus-text-muted)" @click="closeDetail">
            <template #icon><CloseOutlined /></template>
          </Button>
        </div>
      </div>

      <!-- Detail Content -->
      <div class="detail-content">
        <!-- Replay Player (when recording exists) -->
        <ReplayPlayer
          v-if="selectedAlert.has_recording"
          :alert-id="selectedAlert.alert_id"
          class="detail-section-spacing"
        />

        <!-- Static snapshot fallback (no recording) -->
        <div v-if="!selectedAlert.has_recording && selectedAlert.snapshot_path" class="detail-section-spacing">
          <!-- Image mode toggle -->
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

          <!-- Image display: standard view or comparison slider -->
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
              ref="alertImageRef"
              :src="`/api/alerts/${selectedAlert.alert_id}/image/${imageMode}`"
              class="alert-detail-img"
              :alt="selectedAlert.alert_id"
            />
            <!-- Canvas annotation overlay (toggled) -->
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

        <!-- No image at all -->
        <div
          v-if="!selectedAlert.has_recording && !selectedAlert.snapshot_path"
          class="no-image-placeholder"
        >
          <Typography.Text type="secondary" style="font-size: 14px">
            无快照或录像数据
          </Typography.Text>
        </div>

        <!-- Metadata & Actions Grid -->
        <div class="meta-grid">
          <!-- Left: Metadata -->
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
                <span class="meta-v"><Tag :color="severityColor[selectedAlert.severity]" style="margin: 0">{{ severityLabel[selectedAlert.severity] }}</Tag></span>
              </div>
              <div class="meta-row">
                <span class="meta-k">置信度</span>
                <span class="meta-v" :style="{ color: scoreColor(selectedAlert.anomaly_score), fontWeight: 600 }">{{ selectedAlert.anomaly_score?.toFixed(4) }}</span>
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

          <!-- Right: Actions -->
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
                @click="handleAcknowledge(selectedAlert.alert_id)"
              >
                <template #icon><CheckCircleOutlined /></template>
                确认真实
              </Button>
              <Button
                v-if="selectedAlert.workflow_status === 'new' || selectedAlert.workflow_status === 'acknowledged'"
                block
                @click="handleFalsePositive(selectedAlert.alert_id)"
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
                @click="handleDelete(selectedAlert.alert_id)"
              >
                <template #icon><DeleteOutlined /></template>
                删除告警
              </Button>

              <!-- Workflow timeline -->
              <div class="workflow-timeline">
                <div class="workflow-label">工作流进度</div>
                <Steps
                  v-if="selectedAlert.workflow_status !== 'false_positive'"
                  :current="workflowStepIndex"
                  size="small"
                  style="font-size: 11px"
                >
                  <Steps.Step title="待处理" />
                  <Steps.Step title="已确认" />
                  <Steps.Step title="调查中" />
                  <Steps.Step title="已解决" />
                  <Steps.Step title="已关闭" />
                </Steps>
                <div v-else class="false-positive-display">
                  <Tag color="orange" style="font-size: 12px; padding: 2px 12px">已标记为误报</Tag>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </main>
</template>

<style scoped>
/* ── Header ── */
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
  font-family: var(--argus-font-mono);
  font-size: 10px;
  color: var(--argus-text-muted);
  letter-spacing: .1em;
}
.alerts-stats b {
  color: var(--argus-text);
  margin-left: 2px;
}

/* ── Filter chips ── */
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
.alerts-chip {
  padding: 4px 12px;
  border: 1px solid var(--argus-border);
  background: transparent;
  color: var(--argus-text-muted);
  font-family: var(--argus-font-mono);
  font-size: 11px;
  letter-spacing: .08em;
  cursor: pointer;
  transition: all .15s;
  border-radius: 0;
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

/* ── Thumbnail ── */
.alert-thumb {
  width: 48px;
  height: 36px;
  border-radius: 3px;
  overflow: hidden;
  background: var(--argus-footer-bg);
  border: 1px solid var(--argus-border);
  transition: border-color .15s;
}
.alert-thumb:hover {
  border-color: #3b82f6;
}
.alert-thumb-empty {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--argus-text-muted);
  font-size: 10px;
}

/* ── Score & Time ── */
.alert-score {
  font-family: var(--argus-font-mono);
  font-size: 15px;
  font-weight: 700;
  letter-spacing: -.02em;
}
.alert-time {
  font-family: var(--argus-font-mono);
  font-size: 10px;
  color: var(--argus-text-muted);
  letter-spacing: .08em;
}

/* ── Severity badge (detail header) ── */
.sev-badge {
  display: inline-block;
  padding: 2px 8px;
  font-family: var(--argus-font-mono);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: .12em;
  color: #fff;
  text-transform: uppercase;
}
.sev-high { background: #b91c1c; }
.sev-medium { background: #d97706; }
.sev-low { background: #2563eb; }
.sev-info { background: #4b5563; }

/* ── Detail header ── */
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
  font-family: var(--argus-font-mono);
  font-size: 12px;
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

/* ── Metadata panels ── */
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
  font-family: var(--argus-font-mono);
  font-size: 10px;
  color: var(--argus-text-muted);
  letter-spacing: .15em;
  text-transform: uppercase;
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
  padding: 5px 0;
  border-bottom: 1px dashed var(--argus-border);
  font-size: 12px;
  line-height: 1.8;
}
.meta-row:last-child {
  border-bottom: none;
}
.meta-k {
  color: var(--argus-text-muted);
  font-family: var(--argus-font-mono);
  font-size: 10px;
  letter-spacing: .1em;
  min-width: 80px;
}
.meta-v {
  color: var(--argus-text);
  font-size: 12px;
}
.meta-v--notes {
  max-width: 180px;
  text-align: right;
}

/* ── Layout ── */
.alerts-layout {
  display: flex;
  flex: 1;
  min-height: 0;
  overflow: hidden;
  margin: 12px;
  border-radius: var(--r-lg, 12px);
  background: var(--argus-surface, rgba(255,255,255,0.85));
  backdrop-filter: blur(16px);
}
.alerts-chip-group--end {
  margin-left: auto;
}
.alert-list-body {
  flex: 1;
  overflow: auto;
  padding: 0 8px;
}

/* ── Bulk action bar ── */
.bulk-action-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--argus-card-bg-solid);
  border-radius: 6px;
  margin-bottom: 8px;
}

/* ── Thumbnail image ── */
.alert-thumb-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

/* ── Group popover ── */
.group-popover-content {
  max-width: 320px;
  max-height: 300px;
  overflow-y: auto;
}

/* ── Detail panel ── */
.detail-panel {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--argus-surface);
}
.detail-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
}
.detail-section-spacing {
  margin-bottom: 20px;
}

/* ── Image mode bar ── */
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

/* ── No image placeholder ── */
.no-image-placeholder {
  padding: 40px;
  text-align: center;
  background: var(--argus-card-bg-solid);
  border-radius: 8px;
  margin-bottom: 20px;
}

/* ── Actions panel ── */
.actions-panel-bd {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.status-display {
  text-align: center;
  padding: 12px 0;
}

/* ── Workflow timeline ── */
.workflow-timeline {
  border-top: 1px solid var(--argus-border);
  margin-top: 4px;
  padding-top: 10px;
}
.workflow-label {
  font-size: 10px;
  color: var(--argus-text-muted);
  letter-spacing: .1em;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.false-positive-display {
  text-align: center;
  padding: 8px 0;
}
</style>

<style>
/* ── Alert row highlighting (unscoped — targets Ant table internals) ── */
.alert-row-high td {
  background: rgba(185, 28, 28, 0.05) !important;
  border-left: 2px solid #b91c1c !important;
}
.alert-row-medium td {
  background: rgba(217, 119, 6, 0.04) !important;
  border-left: 2px solid #d97706 !important;
}
.alert-row-medium td:not(:first-child) {
  border-left: none !important;
}
.alert-row-high td:not(:first-child) {
  border-left: none !important;
}
.alert-row-selected td {
  background: rgba(59, 130, 246, 0.10) !important;
  border-left: 2px solid #3b82f6 !important;
}
.alert-row-selected td:not(:first-child) {
  border-left: none !important;
}

/* Compact table in detail mode */
.ant-table-small .ant-table-cell {
  padding: 6px 8px !important;
}

/* Table row hover transition */
.ant-table-tbody > tr {
  transition: background .15s ease;
}
</style>
