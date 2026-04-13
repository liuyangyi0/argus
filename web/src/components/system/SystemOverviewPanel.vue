<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Card, Descriptions, Table, Badge, Tag, Popconfirm, Button, message, Skeleton, Tooltip } from 'ant-design-vue'
import { WarningOutlined } from '@ant-design/icons-vue'
import { getHealth, getStorageInfo, cleanupAlerts, getDriftStatus, getCameraHealth } from '../../api'
import { useWebSocket } from '../../composables/useWebSocket'

const health = ref<any>(null)
const storageInfo = ref<any>(null)
const cleanupLoading = ref(false)
const driftData = ref<any[]>([])
const healthCheckData = ref<any[]>([])

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

async function loadDrift() {
  try {
    const res = await getDriftStatus()
    driftData.value = res?.cameras ?? []
  } catch { /* silent */ }
}

async function loadCameraHealth() {
  try {
    const res = await getCameraHealth()
    healthCheckData.value = res?.cameras ?? []
  } catch { /* silent */ }
}

onMounted(() => { Promise.all([fetchHealth(), loadStorageInfo(), loadDrift(), loadCameraHealth()]) })
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

    <!-- Drift Monitor -->
    <Card title="漂移监控" style="margin-top: 16px" v-if="driftData.length">
      <Table :data-source="driftData" :pagination="false" row-key="camera_id" size="small"
        :columns="[
          { title: '摄像头', dataIndex: 'camera_id', key: 'id' },
          { title: '状态', key: 'drift_status', width: 100 },
          { title: 'KS 统计量', dataIndex: 'ks_statistic', key: 'ks', width: 120 },
          { title: 'p 值', dataIndex: 'p_value', key: 'p', width: 120 },
          { title: '当前均值', dataIndex: 'current_mean', key: 'curr', width: 100 },
          { title: '参考均值', dataIndex: 'reference_mean', key: 'ref', width: 100 },
          { title: '样本数', dataIndex: 'samples_collected', key: 'samples', width: 80 },
        ]">
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'drift_status'">
            <Tag :color="record.is_drifted ? 'red' : 'green'">{{ record.is_drifted ? '已漂移' : '正常' }}</Tag>
          </template>
        </template>
      </Table>
    </Card>

    <!-- Camera Health (5-check) -->
    <Card title="摄像头健康检查" style="margin-top: 16px" v-if="healthCheckData.length">
      <Table :data-source="healthCheckData" :pagination="false" row-key="camera_id" size="small"
        :columns="[
          { title: '摄像头', dataIndex: 'camera_id', key: 'id' },
          { title: '冻结', key: 'frozen', width: 70 },
          { title: '清晰度', dataIndex: 'sharpness_score', key: 'sharpness', width: 80 },
          { title: '位移(px)', dataIndex: 'displacement_px', key: 'disp', width: 90 },
          { title: '闪光', key: 'flash', width: 70 },
          { title: '增益漂移', key: 'gain', width: 100 },
          { title: '告警', key: 'warnings' },
        ]">
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'frozen'">
            <Tag :color="record.is_frozen ? 'red' : 'green'" size="small">{{ record.is_frozen ? '是' : '否' }}</Tag>
          </template>
          <template v-if="column.key === 'flash'">
            <Tag :color="record.is_flash ? 'orange' : 'green'" size="small">{{ record.is_flash ? '是' : '否' }}</Tag>
          </template>
          <template v-if="column.key === 'gain'">
            <span :style="{ color: Math.abs(record.gain_drift_pct) > 20 ? 'var(--red)' : 'inherit' }">
              {{ record.gain_drift_pct }}%
            </span>
          </template>
          <template v-if="column.key === 'warnings'">
            <Tooltip v-for="w in record.warnings" :key="w" :title="w">
              <Tag color="orange" size="small"><WarningOutlined /> {{ w }}</Tag>
            </Tooltip>
            <span v-if="!record.warnings?.length" style="color: var(--ink-4)">—</span>
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
