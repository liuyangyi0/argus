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
  background: rgba(0, 0, 0, 0.2);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  flex-shrink: 0;
}
.select-all-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
}
.selection-count {
  font-size: 13px;
  color: #fff;
  font-weight: 500;
}
.custom-checkbox {
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 4px;
  border: 1px solid rgba(255,255,255,0.3);
  background: rgba(0, 0, 0, 0.2);
  cursor: pointer;
  position: relative;
  transition: all 0.2s;
}
.custom-checkbox:checked {
  background: #1890ff;
  border-color: #1890ff;
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
  background: #1890ff;
  border-color: #1890ff;
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
  background: rgba(255,255,255,0.1);
  border: none;
  color: rgba(255,255,255,0.8);
  width: 32px;
  height: 32px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
}
.bulk-btn:hover { background: rgba(255,255,255,0.2); color: #fff; }
.bulk-btn.ack:hover { background: rgba(16, 185, 129, 0.3); color: #10b981; }
.bulk-btn.del:hover { background: rgba(239, 68, 68, 0.3); color: #ef4444; }
.bulk-btn.fp:hover { background: rgba(245, 158, 11, 0.3); color: #fbbf24; }

.empty-inbox {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(255,255,255,0.3);
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
  background: rgba(255, 255, 255, 0.02);
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.05);
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  align-items: center;
}
.inbox-item:hover {
  background: rgba(255, 255, 255, 0.06);
  border-color: rgba(255, 255, 255, 0.15);
}
.inbox-item.active {
  background: rgba(24, 144, 255, 0.1);
  border-color: rgba(24, 144, 255, 0.4);
  box-shadow: 0 0 0 1px rgba(24, 144, 255, 0.2);
}
.item-new {
  border-left: 3px solid rgba(239, 68, 68, 0.8);
}
.item-new:not(.active) {
  background: linear-gradient(90deg, rgba(239, 68, 68, 0.05) 0%, rgba(255,255,255,0.02) 100%);
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
  border-radius: 6px;
  overflow: hidden;
  background: rgba(0,0,0,0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.item-snapshot img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.9;
}
.inbox-item:hover .item-snapshot img { opacity: 1; }
.no-img {
  font-size: 12px;
  color: rgba(255,255,255,0.3);
}

.group-badge {
  position: absolute;
  top: 4px;
  right: 4px;
  background: rgba(0,0,0,0.65);
  backdrop-filter: blur(4px);
  color: #fff;
  font-size: 11px;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.2);
  cursor: pointer;
  pointer-events: auto;
}
.group-badge:hover { background: rgba(0,0,0,0.8); border-color: rgba(255,255,255,0.5); }

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
.severity-dot.sev-high { background: #ef4444; box-shadow: 0 0 8px rgba(239,68,68,0.6); }
.severity-dot.sev-medium { background: #f59e0b; }
.severity-dot.sev-low { background: #3b82f6; }
.severity-dot.sev-info { background: #6b7280; }
.severity-text {
  font-size: 11px;
  color: rgba(255,255,255,0.6);
  font-weight: 500;
}
.time {
  font-size: 11px;
  color: rgba(255,255,255,0.4);
  font-family: 'JetBrains Mono', monospace;
}

.title {
  font-size: 14px;
  color: rgba(255,255,255,0.9);
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.subtitle {
  font-size: 12px;
  color: rgba(255,255,255,0.5);
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
  color: rgba(255,255,255,0.5);
  padding: 2px 6px;
  background: rgba(255,255,255,0.05);
  border-radius: 4px;
}
.status-badge.new { color: #fff; background: rgba(59, 130, 246, 0.4); }
.status-badge.acknowledged { color: #10b981; background: rgba(16, 185, 129, 0.1); }
.status-badge.false_positive { color: #f59e0b; background: rgba(245, 158, 11, 0.1); }

.popover-list { width: 240px; }
.popover-list-item { 
  display: flex; 
  justify-content: space-between; 
  padding: 6px 4px; 
  cursor: pointer; 
}
.popover-list-item:hover { background: rgba(0,0,0,0.05); }
.g-time { font-size: 12px; font-family: monospace; color: var(--ink-2); }
</style>
