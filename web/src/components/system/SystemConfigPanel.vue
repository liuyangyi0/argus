<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  Card, Space, Button, Typography, Switch, Select, Input, InputNumber,
  Slider, Table, Tag, message, Popconfirm, Tooltip,
} from 'ant-design-vue'
import {
  ReloadOutlined, SaveOutlined, DeleteOutlined, UndoOutlined,
  SendOutlined, LockOutlined, PoweroffOutlined,
} from '@ant-design/icons-vue'
import {
  reloadConfig, saveConfig, createBackup,
  getAudioAlerts, updateAudioAlerts,
  updateDetectionParams, updateNotifications, testWebhook,
  restartCamera, clearLock,
  listBackups, restoreBackup, deleteBackup,
} from '../../api'
import { getCameras } from '../../api/cameras'
import { useAuthStore } from '../../stores/useAuthStore'

// All write operations on this panel are admin-only. Backend RBAC remains
// authoritative; client-side gating just avoids the 403 toast on click.
const auth = useAuthStore()

// ── Config reload / save ──
const configLoading = ref(false)
const saveLoading = ref(false)

async function handleReloadConfig() {
  configLoading.value = true
  try { await reloadConfig(); message.success('配置已重新加载') }
  catch { message.error('重载失败') }
  finally { configLoading.value = false }
}
async function handleSaveConfig() {
  saveLoading.value = true
  try { await saveConfig(); message.success('配置已保存至 YAML') }
  catch { message.error('保存失败') }
  finally { saveLoading.value = false }
}

// ── Audio alerts ──
const audioConfig = ref<any>({ low: {}, medium: {}, high: {} })
const audioLoading = ref(false)

async function loadAudioConfig() {
  audioLoading.value = true
  try { audioConfig.value = await getAudioAlerts() }
  catch { /* silent */ }
  finally { audioLoading.value = false }
}
async function saveAudioConfig() {
  try { await updateAudioAlerts(audioConfig.value); message.success('音频配置已保存') }
  catch { message.error('保存失败') }
}

// ── Detection params ──
const detectionParams = ref({
  anomaly_threshold: 0.5,
  sev_info: 0.3, sev_low: 0.5, sev_medium: 0.7, sev_high: 0.85,
})
const detectionLoading = ref(false)

// 痛点 9: per-section apply toast
import { useConfigApplyToast } from '../../composables/useConfigApplyToast'
const { notifyAll: notifyConfigApply } = useConfigApplyToast()

async function handleSaveDetection() {
  detectionLoading.value = true
  try {
    const res = await updateDetectionParams(detectionParams.value)
    notifyConfigApply(res || {})
    if (!res?.anomaly_threshold?.changed
        && !res?.severity?.changed
        && !res?.temporal?.changed
        && !res?.suppression?.changed) {
      // No knob actually changed — keep the old generic toast
      message.success(`检测参数已更新 (${res?.pipelines_updated ?? 0} 条流水线)`)
    }
  } catch { message.error('更新失败') }
  finally { detectionLoading.value = false }
}

// ── Webhook ──
const webhookConfig = ref({ webhook_enabled: false, webhook_url: '', webhook_timeout: 10 })
const webhookLoading = ref(false)
const webhookTestLoading = ref(false)

async function handleSaveWebhook() {
  webhookLoading.value = true
  try { await updateNotifications(webhookConfig.value); message.success('Webhook 配置已保存') }
  catch { message.error('保存失败') }
  finally { webhookLoading.value = false }
}
async function handleTestWebhook() {
  webhookTestLoading.value = true
  try { await testWebhook(); message.success('测试消息已发送') }
  catch { message.error('发送失败') }
  finally { webhookTestLoading.value = false }
}

// ── Camera control ──
const cameras = ref<any[]>([])
const cameraActionLoading = ref<Record<string, boolean>>({})

async function loadCameras() {
  try {
    const res = await getCameras()
    cameras.value = Array.isArray(res) ? res : res?.cameras ?? []
  } catch { /* silent */ }
}
async function handleRestartCamera(cameraId: string) {
  cameraActionLoading.value[cameraId] = true
  try { await restartCamera(cameraId); message.success(`${cameraId} 已重启`) }
  catch { message.error('重启失败') }
  finally { cameraActionLoading.value[cameraId] = false }
}
async function handleClearLock(cameraId: string) {
  try { await clearLock(cameraId); message.success(`${cameraId} 锁定已清除`) }
  catch { message.error('清除失败') }
}

const cameraColumns = [
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '状态', dataIndex: 'connected', key: 'connected', width: 80 },
  { title: '操作', key: 'actions', width: 200 },
]

// ── Backup management ──
const backups = ref<any[]>([])
const backupsLoading = ref(false)
const backupCreateLoading = ref(false)

async function loadBackups() {
  backupsLoading.value = true
  try {
    const res = await listBackups()
    backups.value = res?.backups ?? []
  } catch { backups.value = [] }
  finally { backupsLoading.value = false }
}
async function handleCreateBackup() {
  backupCreateLoading.value = true
  try { await createBackup(); message.success('备份已创建'); loadBackups() }
  catch { message.error('备份失败') }
  finally { backupCreateLoading.value = false }
}
async function handleRestore(name: string) {
  try { await restoreBackup(name); message.success(`已从 ${name} 恢复，建议重启服务`) }
  catch { message.error('恢复失败') }
}
async function handleDeleteBackup(name: string) {
  try { await deleteBackup(name); message.success('备份已删除'); loadBackups() }
  catch { message.error('删除失败') }
}

const backupColumns = [
  { title: '创建时间', dataIndex: 'created', key: 'created' },
  { title: '大小', dataIndex: 'size_mb', key: 'size_mb', width: 100 },
  { title: '内容', key: 'content', width: 160 },
  { title: '操作', key: 'actions', width: 180 },
]

// ── Init ──
onMounted(() => { Promise.all([loadAudioConfig(), loadCameras(), loadBackups()]) })
</script>

<template>
  <div>
    <!-- Config reload / save -->
    <Card title="系统配置" style="margin-bottom: 16px">
      <Space v-if="auth.hasRole(['admin'])">
        <Button type="primary" :loading="configLoading" @click="handleReloadConfig">
          <template #icon><ReloadOutlined /></template>重新加载配置
        </Button>
        <Button :loading="saveLoading" @click="handleSaveConfig">
          <template #icon><SaveOutlined /></template>保存当前配置
        </Button>
      </Space>
      <p style="color: var(--ink-4); margin-top: 12px; font-size: 13px">
        重新加载: 从 configs/default.yaml 读取。保存: 将运行时配置写回 YAML。
      </p>
    </Card>

    <!-- Detection params -->
    <Card title="检测参数" style="margin-bottom: 16px">
      <div style="max-width: 500px">
        <div style="margin-bottom: 16px">
          <Typography.Text strong>异常阈值</Typography.Text>
          <Slider v-model:value="detectionParams.anomaly_threshold" :min="0.1" :max="1" :step="0.05" />
        </div>
        <Typography.Text strong style="margin-bottom: 8px; display: block">严重度阈值</Typography.Text>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px">
          <div><Typography.Text type="secondary">提示 (info)</Typography.Text>
            <InputNumber v-model:value="detectionParams.sev_info" :min="0" :max="1" :step="0.05" style="width: 100%" /></div>
          <div><Typography.Text type="secondary">低 (low)</Typography.Text>
            <InputNumber v-model:value="detectionParams.sev_low" :min="0" :max="1" :step="0.05" style="width: 100%" /></div>
          <div><Typography.Text type="secondary">中 (medium)</Typography.Text>
            <InputNumber v-model:value="detectionParams.sev_medium" :min="0" :max="1" :step="0.05" style="width: 100%" /></div>
          <div><Typography.Text type="secondary">高 (high)</Typography.Text>
            <InputNumber v-model:value="detectionParams.sev_high" :min="0" :max="1" :step="0.05" style="width: 100%" /></div>
        </div>
        <Button v-if="auth.hasRole(['admin'])" type="primary" :loading="detectionLoading" @click="handleSaveDetection">更新检测参数</Button>
      </div>
    </Card>

    <!-- Webhook notifications -->
    <Card title="Webhook 通知" style="margin-bottom: 16px">
      <div style="max-width: 500px">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px">
          <Typography.Text strong>启用 Webhook</Typography.Text>
          <Switch v-model:checked="webhookConfig.webhook_enabled" checked-children="开" un-checked-children="关" size="small" />
        </div>
        <template v-if="webhookConfig.webhook_enabled">
          <div style="margin-bottom: 8px">
            <Typography.Text type="secondary">URL</Typography.Text>
            <Input v-model:value="webhookConfig.webhook_url" placeholder="https://example.com/webhook" />
          </div>
          <div style="margin-bottom: 12px">
            <Typography.Text type="secondary">超时 (秒)</Typography.Text>
            <InputNumber v-model:value="webhookConfig.webhook_timeout" :min="1" :max="60" style="width: 100%" />
          </div>
        </template>
        <Space v-if="auth.hasRole(['admin'])">
          <Button type="primary" :loading="webhookLoading" @click="handleSaveWebhook">
            <template #icon><SaveOutlined /></template>保存
          </Button>
          <Button :loading="webhookTestLoading" :disabled="!webhookConfig.webhook_enabled" @click="handleTestWebhook">
            <template #icon><SendOutlined /></template>发送测试
          </Button>
        </Space>
      </div>
    </Card>

    <!-- Camera control -->
    <Card title="摄像头控制" style="margin-bottom: 16px">
      <Table :dataSource="cameras" :columns="cameraColumns" :pagination="false" size="small" rowKey="camera_id">
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'connected'">
            <Tag :color="record.connected ? 'green' : 'red'">{{ record.connected ? '在线' : '离线' }}</Tag>
          </template>
          <template v-if="column.key === 'actions'">
            <Space size="small">
              <Popconfirm title="确定重启该摄像头流水线？" @confirm="handleRestartCamera(record.camera_id)">
                <Button size="small" :loading="cameraActionLoading[record.camera_id]">
                  <template #icon><PoweroffOutlined /></template>重启
                </Button>
              </Popconfirm>
              <Tooltip title="清除异常区域锁定">
                <Button size="small" @click="handleClearLock(record.camera_id)">
                  <template #icon><LockOutlined /></template>解锁
                </Button>
              </Tooltip>
            </Space>
          </template>
        </template>
      </Table>
    </Card>

    <!-- Backup management -->
    <Card title="数据备份" style="margin-bottom: 16px">
      <div style="margin-bottom: 12px">
        <Button v-if="auth.hasRole(['admin'])" type="primary" :loading="backupCreateLoading" @click="handleCreateBackup">立即备份</Button>
        <Button style="margin-left: 8px" @click="loadBackups">
          <template #icon><ReloadOutlined /></template>刷新
        </Button>
      </div>
      <Table :dataSource="backups" :columns="backupColumns" :loading="backupsLoading" :pagination="false" size="small" rowKey="name">
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'size_mb'">{{ record.size_mb }} MB</template>
          <template v-if="column.key === 'content'">
            <Space size="small">
              <Tag v-if="record.has_db" color="blue">数据库</Tag>
              <Tag v-if="record.has_configs" color="green">配置</Tag>
              <Tag v-if="record.has_models" color="orange">模型</Tag>
            </Space>
          </template>
          <template v-if="column.key === 'actions'">
            <Space v-if="auth.hasRole(['admin'])" size="small">
              <Popconfirm title="确定从此备份恢复？当前数据库将被覆盖。" @confirm="handleRestore(record.name)">
                <Button size="small" type="primary" ghost><template #icon><UndoOutlined /></template>恢复</Button>
              </Popconfirm>
              <Popconfirm title="确定删除此备份？不可撤销。" @confirm="handleDeleteBackup(record.name)">
                <Button size="small" danger><template #icon><DeleteOutlined /></template></Button>
              </Popconfirm>
            </Space>
          </template>
        </template>
      </Table>
    </Card>

    <!-- Audio alerts -->
    <Card title="告警音频配置" style="max-width: 600px">
      <div v-for="(sev, key) in { low: '低级 (LOW)', medium: '中级 (MEDIUM)', high: '高级 (HIGH)' }" :key="key" style="margin-bottom: 16px; padding: 12px; border: 0.5px solid var(--line-2); border-radius: 6px">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px">
          <Typography.Text strong>{{ sev }}</Typography.Text>
          <Switch v-if="audioConfig[key]" v-model:checked="audioConfig[key].enabled" checked-children="开" un-checked-children="关" size="small" />
        </div>
        <Space v-if="audioConfig[key]?.enabled">
          <Select v-model:value="audioConfig[key].sound" style="width: 200px" placeholder="声音">
            <Select.Option value="beep_single">单声提示</Select.Option>
            <Select.Option value="beep_double">双声提示</Select.Option>
            <Select.Option value="beep_double_voice">双声 + 语音播报</Select.Option>
          </Select>
          <Input v-if="key === 'high'" v-model:value="audioConfig[key].voice_template" placeholder="语音模板, 如: {camera} 高级别告警" style="width: 250px" />
        </Space>
      </div>
      <Button v-if="auth.hasRole(['admin'])" type="primary" @click="saveAudioConfig">保存音频配置</Button>
    </Card>
  </div>
</template>
