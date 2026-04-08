<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import {
  Card, Tabs, Descriptions, Typography, Table, Button, Space, Tag,
  Form, Input, Select, message, Popconfirm, Badge, Switch,
} from 'ant-design-vue'
import api, {
  getHealth,
  getUsers as apiGetUsers,
  createUser as apiCreateUser,
  deleteUser as apiDeleteUser,
  toggleUserActive,
  getAuditLogs,
  getDegradationHistory,
  getAudioAlerts,
  updateAudioAlerts,
} from '../api'
import { useWebSocket } from '../composables/useWebSocket'

const activeTab = ref('overview')
const health = ref<any>(null)

// ── Users ──
const users = ref<any[]>([])
const usersLoading = ref(false)
const newUser = ref({ username: '', password: '', role: 'operator', display_name: '' })

// ── Audit ──
const auditEntries = ref<any[]>([])
const auditLoading = ref(false)
const auditTotal = ref(0)
const auditPagination = reactive({ current: 1, pageSize: 20 })
const auditFilters = reactive({ user: '', action: '' })
const auditUserOptions = ref<string[]>([])

// ── Config ──
const configLoading = ref(false)

// ── Degradation History ──
const degradationEvents = ref<any[]>([])
const degradationLoading = ref(false)
const degradationDays = ref(7)

// ── Audio Alerts ──
const audioConfig = ref<any>({ low: {}, medium: {}, high: {} })
const audioLoading = ref(false)

async function fetchHealth() {
  try {
    const res = await getHealth()
    health.value = res
  } catch (e) {
    console.error(e)
  }
}

const { } = useWebSocket({
  topics: ['health'],
  onMessage(topic, data) {
    if (topic === 'health') health.value = data
  },
  fallbackPoll: fetchHealth,
  fallbackInterval: 15000,
})

onMounted(fetchHealth)

// ── Users functions ──
async function loadUsers() {
  usersLoading.value = true
  try {
    const res = await apiGetUsers()
    users.value = res.users
  } catch (e) {
    console.error(e)
    message.error('加载用户列表失败')
  } finally {
    usersLoading.value = false
  }
}

async function createUser() {
  try {
    await apiCreateUser({
      username: newUser.value.username,
      password: newUser.value.password,
      role: newUser.value.role,
      display_name: newUser.value.display_name,
    })
    message.success('用户已创建')
    newUser.value = { username: '', password: '', role: 'operator', display_name: '' }
    loadUsers()
  } catch (e: any) {
    message.error(e.response?.data?.error || '创建失败')
  }
}

async function deleteUser(username: string) {
  try {
    await apiDeleteUser(username)
    message.success('已删除')
    loadUsers()
  } catch (e: any) {
    message.error('删除失败')
  }
}

async function handleToggleActive(username: string) {
  try {
    await toggleUserActive(username)
    message.success('状态已更新')
    loadUsers()
  } catch (e: any) {
    message.error('切换状态失败')
  }
}

// ── Audit functions ──
async function loadAudit() {
  auditLoading.value = true
  try {
    const res = await getAuditLogs({
      page: auditPagination.current,
      page_size: auditPagination.pageSize,
      user: auditFilters.user || undefined,
      action: auditFilters.action || undefined,
    })
    auditEntries.value = res.entries
    auditTotal.value = res.total
    const uniqueUsers = new Set<string>(res.entries.map((e: any) => e.user).filter(Boolean))
    auditUserOptions.value = Array.from(uniqueUsers).sort()
  } catch (e) {
    console.error(e)
    message.error('加载审计日志失败')
  } finally {
    auditLoading.value = false
  }
}

function onAuditTableChange(pagination: any) {
  auditPagination.current = pagination.current
  auditPagination.pageSize = pagination.pageSize
  loadAudit()
}

function onAuditFilterChange() {
  auditPagination.current = 1
  loadAudit()
}

// ── Config functions ──
async function reloadConfig() {
  configLoading.value = true
  try {
    await api.post('/config/reload')
    message.success('配置已重新加载')
  } catch (e: any) {
    message.error('重载失败')
  } finally {
    configLoading.value = false
  }
}

async function createBackup() {
  try {
    await api.post('/backup/create')
    message.success('备份已创建')
  } catch (e: any) {
    message.error(e.response?.data?.error || '备份失败')
  }
}

async function loadDegradation() {
  degradationLoading.value = true
  try {
    const res = await getDegradationHistory(degradationDays.value)
    degradationEvents.value = res || []
  } catch { /* silent */ }
  finally { degradationLoading.value = false }
}

async function loadAudioConfig() {
  audioLoading.value = true
  try {
    const res = await getAudioAlerts()
    audioConfig.value = res
  } catch { /* silent */ }
  finally { audioLoading.value = false }
}

async function saveAudioConfig() {
  try {
    await updateAudioAlerts(audioConfig.value)
    message.success('音频配置已保存')
  } catch { message.error('保存失败') }
}

function onTabChange(key: string | number) {
  activeTab.value = String(key)
  if (key === 'users') loadUsers()
  if (key === 'audit') loadAudit()
  if (key === 'degradation') loadDegradation()
  if (key === 'audio') loadAudioConfig()
}

// ── Column definitions ──
const userColumns = [
  { title: '用户名', dataIndex: 'username', key: 'username' },
  { title: '显示名', dataIndex: 'display_name', key: 'display_name' },
  { title: '角色', dataIndex: 'role', key: 'role' },
  { title: '状态', key: 'active', width: 100 },
  { title: '最后登录', dataIndex: 'last_login', key: 'last_login' },
  { title: '操作', key: 'action', width: 120 },
]

const auditColumns = [
  { title: '时间', dataIndex: 'timestamp', key: 'timestamp', width: 180 },
  { title: '用户', dataIndex: 'user', key: 'user', width: 120 },
  { title: '操作', dataIndex: 'action', key: 'action', width: 150 },
  { title: '详情', dataIndex: 'details', key: 'details', ellipsis: true },
  { title: 'IP 地址', dataIndex: 'ip_address', key: 'ip_address', width: 140 },
]

const roleLabels: Record<string, string> = { admin: '管理员', engineer: '工程师', operator: '操作员', viewer: '观察者' }
const roleColors: Record<string, string> = { admin: 'red', engineer: 'purple', operator: 'blue', viewer: 'default' }

const degradationColumns = [
  { title: '时间', key: 'started_at', width: 180 },
  { title: '级别', key: 'level', width: 80 },
  { title: '类别', dataIndex: 'category', key: 'category', width: 140 },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id', width: 100 },
  { title: '标题', dataIndex: 'title', key: 'title', ellipsis: true },
  { title: '持续', key: 'duration', width: 100 },
  { title: '状态', key: 'status', width: 80 },
]

const degradationLevelColors: Record<string, string> = {
  info: 'blue', warning: 'gold', moderate: 'orange', severe: 'red',
}
const degradationLevelLabels: Record<string, string> = {
  info: '提示', warning: '警告', moderate: '中度', severe: '严重',
}
</script>

<template>
  <div>
    <Typography.Title :level="3" style="margin-bottom: 24px">系统管理</Typography.Title>
    <Tabs :activeKey="activeTab" @change="onTabChange">
      <!-- Overview -->
      <Tabs.TabPane key="overview" tab="系统概览">
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
      </Tabs.TabPane>

      <!-- Config -->
      <Tabs.TabPane key="config" tab="配置管理">
        <Card title="系统配置">
          <Space>
            <Button type="primary" :loading="configLoading" @click="reloadConfig">重新加载配置</Button>
          </Space>
          <p style="color: #666; margin-top: 12px; font-size: 13px">
            重新从 configs/default.yaml 加载配置。修改配置文件后点击此按钮生效。
          </p>
        </Card>
      </Tabs.TabPane>

      <!-- Backup -->
      <Tabs.TabPane key="backup" tab="数据备份">
        <Card title="数据备份">
          <p style="color: #888; margin-bottom: 16px">
            备份内容：SQLite 数据库、配置文件。
          </p>
          <Button type="primary" @click="createBackup">立即备份</Button>
        </Card>
      </Tabs.TabPane>

      <!-- Audit -->
      <Tabs.TabPane key="audit" tab="审计日志">
        <Card title="审计日志">
          <Space style="margin-bottom: 16px">
            <Select
              v-model:value="auditFilters.user"
              placeholder="按用户筛选"
              allow-clear
              style="width: 160px"
              @change="onAuditFilterChange"
            >
              <Select.Option v-for="u in auditUserOptions" :key="u" :value="u">{{ u }}</Select.Option>
            </Select>
            <Input
              v-model:value="auditFilters.action"
              placeholder="按操作筛选"
              allow-clear
              style="width: 200px"
              @press-enter="onAuditFilterChange"
            />
            <Button @click="onAuditFilterChange">筛选</Button>
          </Space>
          <Table
            :columns="auditColumns"
            :data-source="auditEntries"
            :loading="auditLoading"
            :pagination="{
              current: auditPagination.current,
              pageSize: auditPagination.pageSize,
              total: auditTotal,
              showSizeChanger: true,
              showTotal: (t: number) => `共 ${t} 条`,
            }"
            row-key="id"
            size="small"
            @change="onAuditTableChange"
          />
        </Card>
      </Tabs.TabPane>

      <!-- Degradation History -->
      <Tabs.TabPane key="degradation" tab="降级事件">
        <Card title="降级事件历史">
          <Space style="margin-bottom: 16px">
            <Select v-model:value="degradationDays" style="width: 120px" @change="loadDegradation">
              <Select.Option :value="1">最近 1 天</Select.Option>
              <Select.Option :value="7">最近 7 天</Select.Option>
              <Select.Option :value="30">最近 30 天</Select.Option>
              <Select.Option :value="90">最近 90 天</Select.Option>
            </Select>
          </Space>
          <Table
            :columns="degradationColumns"
            :data-source="degradationEvents"
            :loading="degradationLoading"
            :pagination="{ pageSize: 20 }"
            row-key="event_id"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'started_at'">
                {{ new Date(record.started_at * 1000).toLocaleString('zh-CN') }}
              </template>
              <template v-if="column.key === 'level'">
                <Tag :color="degradationLevelColors[record.level]">
                  {{ degradationLevelLabels[record.level] || record.level }}
                </Tag>
              </template>
              <template v-if="column.key === 'duration'">
                {{
                  record.resolved_at
                    ? Math.round((record.resolved_at - record.started_at) / 60) + ' 分钟'
                    : '进行中'
                }}
              </template>
              <template v-if="column.key === 'status'">
                <Tag v-if="record.resolved_at" color="green">已恢复</Tag>
                <Tag v-else color="red">活跃</Tag>
              </template>
            </template>
          </Table>
        </Card>
      </Tabs.TabPane>

      <!-- Audio Config -->
      <Tabs.TabPane key="audio" tab="音频告警">
        <Card title="告警音频配置" style="max-width: 600px">
          <div v-for="(sev, key) in { low: '低级 (LOW)', medium: '中级 (MEDIUM)', high: '高级 (HIGH)' }" :key="key" style="margin-bottom: 16px; padding: 12px; border: 1px solid #2d2d4a; border-radius: 6px">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px">
              <Typography.Text strong>{{ sev }}</Typography.Text>
              <Switch
                v-model:checked="audioConfig[key].enabled"
                checked-children="开"
                un-checked-children="关"
                size="small"
              />
            </div>
            <Space v-if="audioConfig[key]?.enabled">
              <Select v-model:value="audioConfig[key].sound" style="width: 200px" placeholder="声音">
                <Select.Option value="beep_single">单声提示</Select.Option>
                <Select.Option value="beep_double">双声提示</Select.Option>
                <Select.Option value="beep_double_voice">双声 + 语音播报</Select.Option>
              </Select>
              <Input
                v-if="key === 'high'"
                v-model:value="audioConfig[key].voice_template"
                placeholder="语音模板, 如: {camera} 高级别告警"
                style="width: 250px"
              />
            </Space>
          </div>
          <Button type="primary" @click="saveAudioConfig">保存音频配置</Button>
        </Card>
      </Tabs.TabPane>

      <!-- Users -->
      <Tabs.TabPane key="users" tab="用户管理">
        <Card title="添加用户" style="margin-bottom: 16px">
          <Form layout="inline" @finish="createUser">
            <Form.Item>
              <Input v-model:value="newUser.username" placeholder="用户名" />
            </Form.Item>
            <Form.Item>
              <Input.Password v-model:value="newUser.password" placeholder="密码" />
            </Form.Item>
            <Form.Item>
              <Select v-model:value="newUser.role" style="width: 120px">
                <Select.Option value="admin">管理员</Select.Option>
                <Select.Option value="engineer">工程师</Select.Option>
                <Select.Option value="operator">操作员</Select.Option>
                <Select.Option value="viewer">观察者</Select.Option>
              </Select>
            </Form.Item>
            <Form.Item>
              <Input v-model:value="newUser.display_name" placeholder="显示名（可选）" />
            </Form.Item>
            <Form.Item>
              <Button type="primary" html-type="submit">添加</Button>
            </Form.Item>
          </Form>
        </Card>
        <Card title="用户列表">
          <Table
            :columns="userColumns"
            :data-source="users"
            :loading="usersLoading"
            :pagination="false"
            row-key="username"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'role'">
                <Tag :color="roleColors[record.role]">{{ roleLabels[record.role] || record.role }}</Tag>
              </template>
              <template v-if="column.key === 'active'">
                <Switch
                  :checked="record.active"
                  checked-children="启用"
                  un-checked-children="禁用"
                  size="small"
                  @change="handleToggleActive(record.username)"
                />
              </template>
              <template v-if="column.key === 'last_login'">
                {{ record.last_login || '-' }}
              </template>
              <template v-if="column.key === 'action'">
                <Popconfirm title="确定删除此用户？" @confirm="deleteUser(record.username)">
                  <Button size="small" danger>删除</Button>
                </Popconfirm>
              </template>
            </template>
          </Table>
        </Card>
      </Tabs.TabPane>
    </Tabs>
  </div>
</template>
