<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'

defineOptions({ name: 'AlertsPage' })
import {
  Table, Tag, Button, Space, Typography, Select, Tooltip,
  Divider, message, Segmented, Steps, Modal, Popover, List, Spin,
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
const detailData = ref<any>(null)
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
const selectedRowKeys = ref<string[]>([])

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
  detailData.value = record
}

function closeDetail() {
  selectedAlert.value = null
  detailData.value = null
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
        const res = await bulkDeleteAlerts(selectedRowKeys.value)
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
    const res = await bulkAcknowledge(selectedRowKeys.value)
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
    const res = await bulkFalsePositive(selectedRowKeys.value)
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
function scoreColor(score: number): string {
  if (score >= 0.95) return '#ef4444'
  if (score >= 0.85) return '#f97316'
  if (score >= 0.7) return '#f59e0b'
  return '#3b82f6'
}

// Table columns — adapt based on whether detail is open
const columns = computed(() => {
  const base = [
    { title: '', key: 'thumbnail', width: 56 },
    {
      title: '严重度',
      key: 'severity',
      width: 72,
      filters: [
        { text: '高', value: 'high' },
        { text: '中', value: 'medium' },
        { text: '低', value: 'low' },
        { text: '提示', value: 'info' },
      ],
    },
    { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id', width: 100, ellipsis: true },
    { title: '分数', key: 'score', width: 72 },
    { title: '时间', key: 'time', width: 90 },
  ]
  // When detail is closed, show more columns
  if (!selectedAlert.value) {
    base.splice(3, 0, { title: '区域', dataIndex: 'zone_id', key: 'zone_id', width: 80, ellipsis: true } as any)
    base.push(
      { title: '状态', key: 'status', width: 90 } as any,
      { title: '操作', key: 'action', width: 160 } as any,
    )
  } else {
    base.push({ title: '状态', key: 'status', width: 80 } as any)
  }
  return base
})
</script>

<template>
  <div style="display: flex; height: calc(100vh - 72px); margin: -24px; overflow: hidden">
    <!-- Left: Alert List -->
    <div
      :style="{
        width: selectedAlert ? '420px' : '100%',
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        transition: 'width 0.25s ease',
        borderRight: selectedAlert ? '1px solid #1f2937' : 'none',
        overflow: 'hidden',
      }"
    >
      <!-- Header -->
      <div style="display: flex; justify-content: space-between; align-items: center; padding: 16px 20px 12px; flex-shrink: 0">
        <Typography.Title :level="4" style="margin: 0">告警中心</Typography.Title>
        <Space size="small">
          <Select
            v-model:value="filters.camera_id"
            placeholder="全部摄像头"
            allow-clear
            size="small"
            style="width: 130px"
            @change="fetchData"
          >
            <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
              {{ cam.camera_id }}
            </Select.Option>
          </Select>
          <Select
            v-model:value="filters.severity"
            placeholder="严重度"
            allow-clear
            size="small"
            style="width: 90px"
            @change="fetchData"
          >
            <Select.Option value="high">高</Select.Option>
            <Select.Option value="medium">中</Select.Option>
            <Select.Option value="low">低</Select.Option>
            <Select.Option value="info">提示</Select.Option>
          </Select>
          <Tooltip title="导出 CSV">
            <Button size="small" @click="handleExportCSV">
              CSV
            </Button>
          </Tooltip>
          <Tooltip title="导出 PDF (打印报告)">
            <Button size="small" @click="handleExportPDF">
              PDF
            </Button>
          </Tooltip>
        </Space>
      </div>

      <!-- Table -->
      <div style="flex: 1; overflow: auto; padding: 0 8px">
        <!-- Bulk action bar -->
        <div
          v-if="selectedRowKeys.length > 0"
          style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: #1a1a2e; border-radius: 6px; margin-bottom: 8px"
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
          :row-selection="{ selectedRowKeys, onChange: (keys: string[]) => { selectedRowKeys = keys } }"
          :pagination="selectedAlert ? { pageSize: 50, simple: true, size: 'small' } : { pageSize: 20, total: totalAlerts, showTotal: (t: number) => `共 ${t} 条` }"
          :custom-row="(record: any) => ({
            onClick: () => showDetail(record),
            class: rowClassName(record),
          })"
          :scroll="{ y: selectedAlert ? 'calc(100vh - 160px)' : undefined }"
          style="cursor: pointer"
        >
          <template #bodyCell="{ column, record }">
            <!-- Thumbnail -->
            <template v-if="column.key === 'thumbnail'">
              <div style="width: 48px; height: 36px; border-radius: 3px; overflow: hidden; background: #0f0f1a">
                <img
                  v-if="record.snapshot_path"
                  :src="`/api/alerts/${record.alert_id}/image/snapshot`"
                  style="width: 100%; height: 100%; object-fit: cover"
                  loading="lazy"
                />
                <div v-else style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #4a5568; font-size: 10px">
                  --
                </div>
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
                    <div style="max-width: 320px; max-height: 300px; overflow-y: auto">
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
              <span :style="{ color: scoreColor(record.anomaly_score), fontWeight: 600, fontSize: '12px' }">
                {{ record.anomaly_score?.toFixed(2) }}
              </span>
            </template>
            <!-- Time -->
            <template v-if="column.key === 'time'">
              <Tooltip :title="formatTimestamp(record.timestamp || record.created_at)">
                <Typography.Text type="secondary" style="font-size: 11px">
                  {{ formatRelativeTime(record.timestamp || record.created_at) }}
                </Typography.Text>
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
      style="flex: 1; min-width: 0; display: flex; flex-direction: column; overflow: hidden; background: #0d0d1a"
    >
      <!-- Detail Header -->
      <div style="display: flex; align-items: center; gap: 12px; padding: 12px 20px; background: #141420; border-bottom: 1px solid #1f2937; flex-shrink: 0">
        <Tag
          :color="severityColor[selectedAlert.severity]"
          style="margin: 0; font-size: 13px; padding: 2px 10px"
        >
          {{ severityLabel[selectedAlert.severity] }}
        </Tag>
        <Typography.Text strong style="font-size: 14px">
          {{ selectedAlert.camera_id }}
        </Typography.Text>
        <Typography.Text type="secondary" style="font-size: 12px">
          {{ selectedAlert.zone_id }}
        </Typography.Text>
        <Typography.Text type="secondary" style="font-size: 12px">
          {{ formatTimestamp(selectedAlert.timestamp || selectedAlert.created_at) }}
        </Typography.Text>

        <div style="margin-left: auto; display: flex; align-items: center; gap: 8px">
          <Tag
            :color="workflowColor[selectedAlert.workflow_status] || 'default'"
            style="margin: 0"
          >
            {{ workflowLabel[selectedAlert.workflow_status] || '待处理' }}
          </Tag>
          <Tooltip title="导出证据包">
            <Button size="small" type="text" style="color: #9ca3af">
              <template #icon><ExportOutlined /></template>
            </Button>
          </Tooltip>
          <Button size="small" type="text" style="color: #9ca3af" @click="closeDetail">
            <template #icon><CloseOutlined /></template>
          </Button>
        </div>
      </div>

      <!-- Detail Content -->
      <div style="flex: 1; overflow-y: auto; padding: 16px 20px">
        <!-- Replay Player (when recording exists) -->
        <ReplayPlayer
          v-if="selectedAlert.has_recording"
          :alert-id="selectedAlert.alert_id"
          style="margin-bottom: 20px"
        />

        <!-- Static snapshot fallback (no recording) -->
        <div v-if="!selectedAlert.has_recording && selectedAlert.snapshot_path" style="margin-bottom: 20px">
          <!-- Image mode toggle -->
          <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px">
            <Segmented
              v-model:value="imageMode"
              :options="[
                { label: '叠加图', value: 'composite' },
                { label: '原图', value: 'snapshot' },
                { label: '热力图', value: 'heatmap' },
              ]"
              size="small"
            />
            <div style="margin-left: auto; display: flex; align-items: center; gap: 8px">
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
          <div v-if="imageMode === 'compare'" style="margin-bottom: 12px">
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
          <div v-else style="border-radius: 8px; overflow: hidden; background: #000; text-align: center; position: relative">
            <img
              ref="alertImageRef"
              :src="`/api/alerts/${selectedAlert.alert_id}/image/${imageMode}`"
              style="max-width: 100%; max-height: 480px; object-fit: contain"
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
          <div style="display: flex; gap: 6px; margin-top: 6px">
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
          style="padding: 40px; text-align: center; background: #1a1a2e; border-radius: 8px; margin-bottom: 20px"
        >
          <Typography.Text type="secondary" style="font-size: 14px">
            无快照或录像数据
          </Typography.Text>
        </div>

        <!-- Metadata & Actions Grid -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px">
          <!-- Left: Metadata -->
          <div style="background: #1a1a2e; border-radius: 8px; padding: 14px">
            <Typography.Text strong style="font-size: 12px; color: #9ca3af; display: block; margin-bottom: 10px">
              告警信息
            </Typography.Text>
            <div style="display: flex; flex-direction: column; gap: 8px">
              <div style="display: flex; justify-content: space-between; align-items: center">
                <Typography.Text type="secondary" style="font-size: 12px">告警 ID</Typography.Text>
                <Typography.Text style="font-size: 11px; font-family: monospace; color: #94a3b8" copyable>
                  {{ selectedAlert.alert_id?.slice(0, 16) }}...
                </Typography.Text>
              </div>
              <div style="display: flex; justify-content: space-between">
                <Typography.Text type="secondary" style="font-size: 12px">摄像头</Typography.Text>
                <Typography.Text style="font-size: 12px">{{ selectedAlert.camera_id }}</Typography.Text>
              </div>
              <div style="display: flex; justify-content: space-between">
                <Typography.Text type="secondary" style="font-size: 12px">区域</Typography.Text>
                <Typography.Text style="font-size: 12px">{{ selectedAlert.zone_id }}</Typography.Text>
              </div>
              <div style="display: flex; justify-content: space-between; align-items: center">
                <Typography.Text type="secondary" style="font-size: 12px">严重度</Typography.Text>
                <Tag :color="severityColor[selectedAlert.severity]" style="margin: 0">
                  {{ severityLabel[selectedAlert.severity] }}
                </Tag>
              </div>
              <div style="display: flex; justify-content: space-between; align-items: center">
                <Typography.Text type="secondary" style="font-size: 12px">异常分数</Typography.Text>
                <span :style="{ color: scoreColor(selectedAlert.anomaly_score), fontWeight: 600, fontSize: '13px' }">
                  {{ selectedAlert.anomaly_score?.toFixed(4) }}
                </span>
              </div>
              <div style="display: flex; justify-content: space-between">
                <Typography.Text type="secondary" style="font-size: 12px">触发时间</Typography.Text>
                <Typography.Text style="font-size: 12px">
                  {{ formatTimestamp(selectedAlert.timestamp || selectedAlert.created_at) }}
                </Typography.Text>
              </div>
              <div v-if="selectedAlert.assigned_to" style="display: flex; justify-content: space-between">
                <Typography.Text type="secondary" style="font-size: 12px">处理人</Typography.Text>
                <Typography.Text style="font-size: 12px">{{ selectedAlert.assigned_to }}</Typography.Text>
              </div>
              <div v-if="selectedAlert.notes" style="display: flex; justify-content: space-between">
                <Typography.Text type="secondary" style="font-size: 12px">备注</Typography.Text>
                <Typography.Text style="font-size: 12px; max-width: 180px; text-align: right">{{ selectedAlert.notes }}</Typography.Text>
              </div>
            </div>
          </div>

          <!-- Right: Actions -->
          <div style="background: #1a1a2e; border-radius: 8px; padding: 14px">
            <Typography.Text strong style="font-size: 12px; color: #9ca3af; display: block; margin-bottom: 10px">
              操作
            </Typography.Text>
            <div style="display: flex; flex-direction: column; gap: 8px">
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
              <div v-if="selectedAlert.workflow_status !== 'new' && selectedAlert.workflow_status !== 'acknowledged'" style="text-align: center; padding: 12px 0">
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
              <Divider style="margin: 8px 0; border-color: #2d2d4a" />
              <Typography.Text type="secondary" style="font-size: 11px; margin-bottom: 4px">工作流进度</Typography.Text>
              <Steps
                v-if="selectedAlert.workflow_status !== 'false_positive'"
                :current="workflowStepIndex"
                size="small"
                direction="vertical"
                style="font-size: 11px"
              >
                <Steps.Step title="待处理" />
                <Steps.Step title="已确认" />
                <Steps.Step title="调查中" />
                <Steps.Step title="已解决" />
                <Steps.Step title="已关闭" />
              </Steps>
              <div v-else style="text-align: center; padding: 8px 0">
                <Tag color="orange" style="font-size: 12px; padding: 2px 12px">已标记为误报</Tag>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style>
/* Alert row highlighting */
.alert-row-high td {
  background: rgba(239, 68, 68, 0.06) !important;
}
.alert-row-medium td {
  background: rgba(249, 115, 22, 0.04) !important;
}
.alert-row-selected td {
  background: rgba(59, 130, 246, 0.12) !important;
}

/* Compact table in detail mode */
.ant-table-small .ant-table-cell {
  padding: 6px 8px !important;
}
</style>
