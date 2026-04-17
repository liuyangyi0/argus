<script setup lang="ts">
import { ref, computed } from 'vue'
import { storeToRefs } from 'pinia'
import {
  Table, Tag, Button, Space, Typography, Tooltip,
  Popover, List, Spin, message, Modal,
} from 'ant-design-vue'
import {
  CheckCircleOutlined,
  StopOutlined,
  DeleteOutlined,
} from '@ant-design/icons-vue'
import { getAlertGroup } from '../../api'
import { useAlertStore } from '../../stores/useAlertStore'
import { formatRelativeTime, formatTimestamp } from '../../utils/time'
import { scoreColor } from '../../utils/colors'

defineOptions({ name: 'AlertsTable' })

const emit = defineEmits<{
  (e: 'select', record: any): void
}>()

const store = useAlertStore()
const { alerts, loading, totalAlerts, selectedAlert } = storeToRefs(store)

// Bulk selection lives here because the bulk action bar is part of the table region.
const selectedRowKeys = ref<(string | number)[]>([])

// Event group popover state
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
  groupPopoverVisible.value = {}
  if (!isOpen) {
    groupPopoverVisible.value[eventGroupId] = true
    loadGroupAlerts(eventGroupId)
  }
}

function handleRowClick(record: any) {
  emit('select', record)
}

async function handleAcknowledge(id: string) {
  try {
    await store.ackAlert(id)
    message.success('已确认')
  } catch (e: any) {
    message.error(e.response?.data?.error || '确认失败')
  }
}

async function handleFalsePositive(id: string) {
  try {
    await store.fpAlert(id)
    message.success('已标记误报')
  } catch (e: any) {
    message.error(e.response?.data?.error || '标记失败')
  }
}

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
        const keys = selectedRowKeys.value as string[]
        const res = await store.bulkDel(keys)
        message.success(res.message || `已删除 ${res.count} 条`)
        selectedRowKeys.value = []
      } catch (e: any) {
        message.error(e.response?.data?.error || '批量删除失败')
      }
    },
  })
}

async function handleBulkAcknowledge() {
  if (!selectedRowKeys.value.length) return
  try {
    const res = await store.bulkAck(selectedRowKeys.value as string[])
    message.success(res.message || `已确认 ${res.count} 条`)
    selectedRowKeys.value = []
  } catch (e: any) {
    message.error(e.response?.data?.error || '批量确认失败')
  }
}

async function handleBulkFalsePositive() {
  if (!selectedRowKeys.value.length) return
  try {
    const res = await store.bulkFp(selectedRowKeys.value as string[])
    message.success(res.message || `已标记 ${res.count} 条为误报`)
    selectedRowKeys.value = []
  } catch (e: any) {
    message.error(e.response?.data?.error || '批量标记失败')
  }
}

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

function rowClassName(record: any) {
  if (record.alert_id === selectedAlert.value?.alert_id) return 'alert-row-selected'
  if (record.severity === 'high' && record.workflow_status === 'new') return 'alert-row-high'
  if (record.severity === 'medium' && record.workflow_status === 'new') return 'alert-row-medium'
  return ''
}

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
    { title: '分类', key: 'category', width: isCompact ? 64 : 76 },
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
  <div class="alert-list-body">
    <!-- Bulk action bar -->
    <div v-if="selectedRowKeys.length > 0" class="bulk-action-bar">
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
        onClick: () => handleRowClick(record),
        class: rowClassName(record),
      })"
      :scroll="{ x: 800, y: selectedAlert ? 'calc(100vh - 160px)' : undefined }"
      style="cursor: pointer"
    >
      <template #bodyCell="{ column, record }">
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
                      <List.Item style="padding: 4px 0; cursor: pointer" @click="handleRowClick(item); groupPopoverVisible = {}">
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

        <template v-if="column.key === 'category'">
          <Tag :color="categoryMeta[record.category]?.color || 'default'" style="margin: 0; font-size: 11px">
            {{ categoryMeta[record.category]?.label || record.category || '未分类' }}
          </Tag>
        </template>

        <template v-if="column.key === 'score'">
          <span class="alert-score" :style="{ color: scoreColor(record.anomaly_score) }">
            {{ record.anomaly_score?.toFixed(2) }}
          </span>
        </template>

        <template v-if="column.key === 'time'">
          <Tooltip :title="formatRelativeTime(record.timestamp || record.created_at)">
            <span class="alert-time">
              {{ formatTimestamp(record.timestamp || record.created_at) }}
            </span>
          </Tooltip>
        </template>

        <template v-if="column.key === 'status'">
          <Tag :color="workflowColor[record.workflow_status] || 'default'" style="margin: 0; font-size: 11px">
            {{ workflowLabel[record.workflow_status] || record.workflow_status || '待处理' }}
          </Tag>
        </template>

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
</template>

<style scoped>
.alert-list-body {
  flex: 1;
  overflow: auto;
  padding: 0 8px;
}

.bulk-action-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: var(--argus-card-bg-solid);
  border-radius: 6px;
  margin-bottom: 8px;
}

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
.alert-thumb-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
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

.alert-score {
  font-size: 14px;
  font-weight: 600;
}
.alert-time {
  font-size: 13px;
  color: var(--argus-text-muted);
}

.group-popover-content {
  max-width: 320px;
  max-height: 300px;
  overflow-y: auto;
}
</style>

<style>
/* Alert row highlighting — unscoped so it reaches Ant table internals */
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

.ant-table-small .ant-table-cell {
  padding: 6px 8px !important;
}

.ant-table-tbody > tr {
  transition: background .15s ease;
}
</style>
