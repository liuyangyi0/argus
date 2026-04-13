<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Card, Descriptions, Table, Badge, Popconfirm, Button, message, Skeleton } from 'ant-design-vue'
import { getHealth, getStorageInfo, cleanupAlerts } from '../../api'
import { useWebSocket } from '../../composables/useWebSocket'

const health = ref<any>(null)
const storageInfo = ref<any>(null)
const cleanupLoading = ref(false)

async function loadStorageInfo() {
  try { storageInfo.value = await getStorageInfo() } catch { /* silent */ }
}

async function handleCleanup() {
  cleanupLoading.value = true
  try {
    const res = await cleanupAlerts()
    message.success(`已清理 ${res.deleted} 条旧告警`)
    loadStorageInfo()
  } catch { message.error('清理失败') }
  finally { cleanupLoading.value = false }
}

async function fetchHealth() {
  try {
    const res = await getHealth()
    health.value = res
  } catch (e) {
    message.error('操作失败')
  }
}

useWebSocket({
  topics: ['health'],
  onMessage(topic, data) {
    if (topic === 'health') health.value = data
  },
  fallbackPoll: fetchHealth,
  fallbackInterval: 15000,
})

onMounted(() => { fetchHealth(); loadStorageInfo() })
</script>

<template>
  <div>
    <Card v-if="!health">
      <Skeleton active :paragraph="{ rows: 4 }" />
    </Card>
    <Card v-if="health">
      <Descriptions :column="2" bordered size="small" title="运行状态">
        <Descriptions.Item label="系统状态">
          <Badge :status="health.status === 'healthy' ? 'success' : 'warning'" />
          {{ health.status?.toUpperCase() }}
        </Descriptions.Item>
        <Descriptions.Item label="运行时间">{{ Math.floor(health.uptime_seconds / 60) }} 分钟</Descriptions.Item>
        <Descriptions.Item label="累计告警">{{ health.total_alerts }}</Descriptions.Item>
        <Descriptions.Item label="Python 版本">{{ health.python_version }}</Descriptions.Item>
        <Descriptions.Item label="操作系统">{{ health.platform }}</Descriptions.Item>
        <Descriptions.Item label="摄像头数">{{ health.cameras?.length || 0 }}</Descriptions.Item>
      </Descriptions>
    </Card>
    
    <Card title="摄像头健康" style="margin-top: 16px" v-if="health?.cameras?.length">
      <Table
        :data-source="health.cameras"
        :pagination="false"
        row-key="camera_id"
        size="small"
        :columns="[
          { title: '摄像头', dataIndex: 'camera_id', key: 'id' },
          { title: '状态', key: 'status' },
          { title: '帧数', dataIndex: 'frames_captured', key: 'frames' },
          { title: '延迟', key: 'latency' },
        ]"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'status'">
            <Badge :status="record.connected ? 'success' : 'default'" />
            {{ record.connected ? '在线' : '离线' }}
          </template>
          <template v-if="column.key === 'latency'">
            {{ record.avg_latency_ms?.toFixed(1) }}ms
          </template>
        </template>
      </Table>
    </Card>

    <Card title="存储与维护" style="margin-top: 16px" v-if="storageInfo">
      <Descriptions :column="2" bordered size="small">
        <Descriptions.Item label="告警记录总数">{{ storageInfo.alert_count }}</Descriptions.Item>
        <Descriptions.Item label="告警保留天数">{{ storageInfo.retention_days }} 天</Descriptions.Item>
        <Descriptions.Item v-if="storageInfo.disk" label="磁盘已用">
          {{ storageInfo.disk.used_gb }} / {{ storageInfo.disk.total_gb }} GB ({{ storageInfo.disk.percent_used }}%)
        </Descriptions.Item>
        <Descriptions.Item v-if="storageInfo.disk" label="磁盘可用">
          {{ storageInfo.disk.free_gb }} GB
        </Descriptions.Item>
      </Descriptions>
      <div style="margin-top: 12px">
        <Popconfirm
          :title="`确定清理超过 ${storageInfo.retention_days} 天的旧告警数据？`"
          ok-text="确定"
          cancel-text="取消"
          @confirm="handleCleanup"
        >
          <Button type="primary" danger :loading="cleanupLoading">清理旧告警</Button>
        </Popconfirm>
      </div>
    </Card>
  </div>
</template>
