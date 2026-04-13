<template>
  <div class="inbox-list glass-panel">
    <!-- Inbox Header / Bulk Actions -->
    <div class="inbox-header">
      <div class="select-all-wrap">
        <input 
          type="checkbox" 
          class="custom-checkbox"
          :checked="selectedRowKeys.length > 0 && selectedRowKeys.length === alerts.length"
          :indeterminate="selectedRowKeys.length > 0 && selectedRowKeys.length < alerts.length"
          @change="toggleSelectAll" 
        />
        <span v-if="selectedRowKeys.length > 0" class="selection-count">
          已选 {{ selectedRowKeys.length }} 项
        </span>
        <span v-else class="selection-count" style="opacity: 0.5;">
          全选 (共 {{ alerts.length }} 项)
        </span>
      </div>

      <div class="bulk-actions" v-if="selectedRowKeys.length > 0">
        <button class="bulk-btn ack" @click="handleBulkAcknowledge" title="批量确认"><CheckCircleOutlined /></button>
        <button class="bulk-btn fp" @click="handleBulkFalsePositive" title="标记误报"><StopOutlined /></button>
        <button class="bulk-btn del" @click="handleBulkDelete" title="批量删除"><DeleteOutlined /></button>
      </div>
    </div>
    
    <div class="alert-grid" v-if="alerts.length > 0">
      <div 
        v-for="item in alerts" 
        :key="item.alert_id"
        class="inbox-item"
        :class="{ 
          'active': selectedAlert?.alert_id === item.alert_id,
          'item-new': item.workflow_status === 'new' && (item.severity === 'high' || item.severity === 'medium')
        }"
        @click="showDetail(item)"
      >
         <div class="item-checkbox" @click.stop>
           <input 
             type="checkbox" 
             class="custom-checkbox"
             :value="item.alert_id"
             v-model="selectedRowKeys"
           />
         </div>

         <div class="item-snapshot">
           <img v-if="item.snapshot_url || item.snapshot_path" :src="item.snapshot_url || `/api/alerts/${item.alert_id}/image/snapshot`" />
           <div v-else class="no-img">暂无</div>
           <!-- Group indicator -->
           <Popover 
             v-if="item.event_group_id && item.event_group_count > 1" 
             title="同组关联告警" 
             trigger="click" 
             placement="right"
             v-model:visible="groupPopoverVisible[item.event_group_id]"
             @visibleChange="(v) => v && loadGroupAlerts(item.event_group_id)"
           >
             <div class="group-badge" @click.stop>+{{ item.event_group_count - 1 }}</div>
             <template #content>
               <Spin v-if="groupLoading" size="small" />
               <List size="small" :dataSource="groupAlerts" v-else class="popover-list">
                 <template #renderItem="{ item: gItem }">
                   <List.Item class="popover-list-item" @click="showDetail(gItem); groupPopoverVisible[item.event_group_id] = false">
                     <span class="g-time">{{ formatTimestamp(gItem.timestamp) }}</span>
                     <Tag :color="scoreColor(gItem.score)" style="margin:0; font-size: 10px;">{{ gItem.score?.toFixed(2) }}</Tag>
                   </List.Item>
                 </template>
               </List>
             </template>
           </Popover>
         </div>

         <div class="item-meta">
           <div class="meta-header">
             <div class="meta-header-left">
               <span :class="['severity-dot', `sev-${item.severity}`]"></span>
               <span class="severity-text">{{ severityLabel[item.severity] || item.severity }}</span>
             </div>
             <span class="time">{{ formatTimestamp(item.timestamp || item.created_at) }}</span>
           </div>
           
           <div class="title">{{ item.camera_name || item.camera_id }}</div>
           <div class="subtitle">{{ item.task_name || item.zone_id || 'Global Zone' }}</div>
           
           <div class="meta-footer">
             <div class="score-badge" :style="{ color: scoreColor(item.anomaly_score || item.score), borderColor: `${scoreColor(item.anomaly_score || item.score)}44`, background: `${scoreColor(item.anomaly_score || item.score)}11` }">
               SCORE {{ (item.anomaly_score || item.score)?.toFixed(2) }}
             </div>
             <span class="status-badge" :class="item.workflow_status">{{ workflowLabel[item.workflow_status] || '待处理' }}</span>
           </div>
         </div>
      </div>
    </div>
    
    <div v-else class="empty-inbox">
      没有查询到告警记录
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { storeToRefs } from 'pinia'
import { message, Popover, Spin, List, Tag } from 'ant-design-vue'
import { CheckCircleOutlined, StopOutlined, DeleteOutlined } from '@ant-design/icons-vue'
import { useAlertStore } from '../../stores/useAlertStore'
import { getAlertGroup } from '../../api'
import { scoreColor } from '../../utils/colors'

const alertStore = useAlertStore()
const { alerts, selectedAlert } = storeToRefs(alertStore) 

const selectedRowKeys = ref<string[]>([])

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

function showDetail(record: any) {
  alertStore.selectedAlert = record
}

function toggleSelectAll(e: Event) {
  const checked = (e.target as HTMLInputElement).checked
  if (checked) {
    selectedRowKeys.value = alerts.value.map((a: any) => a.alert_id)
  } else {
    selectedRowKeys.value = []
  }
}

const groupAlerts = ref<any[]>([])
const groupLoading = ref(false)
const groupPopoverVisible = ref<Record<string, boolean>>({})

async function loadGroupAlerts(eventGroupId: string) {
  if (!eventGroupId) return
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

async function handleBulkAcknowledge() {
  if (!selectedRowKeys.value.length) return
  try {
    const res = await alertStore.bulkAck(selectedRowKeys.value)     
    message.success(res.message || `已确认 ${res.count} 条`)
    selectedRowKeys.value = []
  } catch (e: any) {
    message.error(e.response?.data?.error || '批量确认失败')
  }
}

async function handleBulkFalsePositive() {
  if (!selectedRowKeys.value.length) return
  try {
    const res = await alertStore.bulkFp(selectedRowKeys.value)      
    message.success(res.message || `已标记 ${res.count} 条为误报`)
    selectedRowKeys.value = []
  } catch (e: any) {
    message.error(e.response?.data?.error || '批量标记失败')
  }
}

async function handleBulkDelete() {
  if (!selectedRowKeys.value.length) return
  if (!confirm(`确定要删除选中的 ${selectedRowKeys.value.length} 条告警吗？无法恢复。`)) return
  try {
    const res = await alertStore.bulkDel(selectedRowKeys.value)     
    message.success(res.message || `已删除 ${res.count} 条`)
    selectedRowKeys.value = []
  } catch (e: any) {
    message.error(e.response?.data?.error || '批量删除失败')
  }
}
</script>

<style scoped>
.inbox-list {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: var(--bg);
}
.inbox-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: var(--glass-strong);
  border-bottom: 0.5px solid var(--line-2);
  flex-shrink: 0;
}
.select-all-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
}
.selection-count {
  font-size: 13px;
  color: var(--ink-3);
  font-weight: 500;
}
.custom-checkbox {
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 4px;
  border: 1px solid var(--line-3);
  background: #fff;
  cursor: pointer;
  position: relative;
  transition: all 0.2s;
}
.custom-checkbox:checked {
  background: var(--accent);
  border-color: var(--accent);
}
.custom-checkbox:checked::after {
  content: '';
  position: absolute;
  left: 5px;
  top: 2px;
  width: 4px;
  height: 8px;
  border: solid white;
  border-width: 0 2px 2px 0;
  transform: rotate(45deg);
}
.custom-checkbox:indeterminate {
  background: var(--accent);
  border-color: var(--accent);
}
.custom-checkbox:indeterminate::after {
  content: '';
  position: absolute;
  left: 3px;
  top: 7px;
  width: 8px;
  height: 2px;
  background: white;
}

.bulk-actions {
  display: flex;
  gap: 8px;
}
.bulk-btn {
  background: rgba(10, 10, 15, 0.04);
  border: 0.5px solid var(--line-2);
  color: var(--ink-4);
  width: 32px;
  height: 32px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
}
.bulk-btn:hover { background: rgba(10, 10, 15, 0.08); color: var(--ink-2); }
.bulk-btn.ack:hover { background: rgba(21, 163, 74, 0.1); color: var(--green); }
.bulk-btn.del:hover { background: rgba(229, 72, 77, 0.1); color: var(--red); }
.bulk-btn.fp:hover { background: rgba(217, 119, 6, 0.1); color: var(--amber); }

.empty-inbox {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--ink-5);
  font-size: 14px;
  letter-spacing: 0.1em;
}

.alert-grid {
  flex: 1;
  padding: 12px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.inbox-item {
  display: flex;
  gap: 14px;
  padding: 14px;
  background: var(--glass);
  border-radius: var(--r-sm);
  border: 0.5px solid var(--line);
  cursor: pointer;
  transition: all 0.2s ease;
  position: relative;
  align-items: center;
}
.inbox-item:hover {
  background: var(--glass-strong);
  border-color: var(--line-2);
  box-shadow: var(--sh-1);
}
.inbox-item.active {
  background: #fff;
  border-color: var(--line-3);
  box-shadow: var(--sh-2);
}
.item-new {
  border-left: 3px solid var(--red);
}
.item-new:not(.active) {
  background: linear-gradient(90deg, rgba(229, 72, 77, 0.04) 0%, var(--glass) 100%);
}

.item-checkbox {
  display: flex;
  align-items: center;
  padding-right: 4px;
}

.item-snapshot {
  position: relative;
  width: 96px;
  height: 72px;
  border-radius: var(--r-xs);
  overflow: hidden;
  background: rgba(10, 10, 15, 0.06);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.item-snapshot img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.95;
}
.inbox-item:hover .item-snapshot img { opacity: 1; }
.no-img {
  font-size: 12px;
  color: var(--ink-5);
}

.group-badge {
  position: absolute;
  top: 4px;
  right: 4px;
  background: rgba(10, 10, 15, 0.7);
  backdrop-filter: blur(4px);
  color: #fff;
  font-size: 11px;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  cursor: pointer;
  pointer-events: auto;
}
.group-badge:hover { background: rgba(10, 10, 15, 0.85); }

.item-meta {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 6px;
}

.meta-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.meta-header-left {
  display: flex;
  align-items: center;
  gap: 6px;
}
.severity-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}
.severity-dot.sev-high { background: var(--red); box-shadow: 0 0 6px rgba(229, 72, 77, 0.4); }
.severity-dot.sev-medium { background: var(--amber); }
.severity-dot.sev-low { background: var(--blue); }
.severity-dot.sev-info { background: var(--ink-5); }
.severity-text {
  font-size: 11px;
  color: var(--ink-4);
  font-weight: 500;
}
.time {
  font-size: 11px;
  color: var(--ink-5);
  font-family: 'JetBrains Mono', monospace;
}

.title {
  font-size: 14px;
  color: var(--ink-2);
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.subtitle {
  font-size: 12px;
  color: var(--ink-4);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.meta-footer {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 2px;
}
.score-badge {
  font-size: 10px;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  padding: 2px 6px;
  border-radius: 4px;
  border: 1px solid;
}
.status-badge {
  font-size: 11px;
  color: var(--ink-4);
  padding: 2px 6px;
  background: rgba(10, 10, 15, 0.04);
  border-radius: 4px;
}
.status-badge.new { color: var(--blue); background: rgba(37, 99, 235, 0.08); }
.status-badge.acknowledged { color: var(--green); background: rgba(21, 163, 74, 0.08); }
.status-badge.false_positive { color: var(--amber); background: rgba(217, 119, 6, 0.08); }

.popover-list { width: 240px; }
.popover-list-item {
  display: flex;
  justify-content: space-between;
  padding: 6px 4px;
  cursor: pointer;
}
.popover-list-item:hover { background: rgba(10, 10, 15, 0.04); }
.g-time { font-size: 12px; font-family: monospace; color: var(--ink-2); }
</style>
