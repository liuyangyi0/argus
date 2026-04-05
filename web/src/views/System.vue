<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  Card, Tabs, Descriptions, Typography, Table, Button, Space, Tag,
  Form, Input, Select, message, Popconfirm, Badge,
} from 'ant-design-vue'
import api, { getHealth } from '../api'

const activeTab = ref('overview')
const health = ref<any>(null)

// ── Users ──
const users = ref<any[]>([])
const usersLoading = ref(false)
const newUser = ref({ username: '', password: '', role: 'operator', display_name: '' })

// ── Config ──
const configLoading = ref(false)

onMounted(async () => {
  try {
    const res = await getHealth()
    health.value = res.data
  } catch (e) {
    console.error(e)
  }
})

async function loadUsers() {
  usersLoading.value = true
  try {
    // Parse users from HTML endpoint (until JSON API exists)
    const res = await api.get('/users', { responseType: 'text', headers: { 'HX-Request': 'true' } })
    const html = res.data as string
    const rows: any[] = []
    // Simple extraction
    const trRegex = /<td[^>]*>([^<]+)<\/td>\s*<td[^>]*>([^<]*)<\/td>\s*<td[^>]*>([^<]*)<\/td>/g
    let match
    while ((match = trRegex.exec(html)) !== null) {
      const username = match[1].trim()
      if (username && username !== '用户名' && !username.includes('td')) {
        rows.push({ username, role: match[2].trim(), display_name: match[3].trim() })
      }
    }
    users.value = rows
  } catch (e) {
    console.error(e)
  } finally {
    usersLoading.value = false
  }
}

async function createUser() {
  try {
    const form = new FormData()
    form.append('username', newUser.value.username)
    form.append('password', newUser.value.password)
    form.append('role', newUser.value.role)
    form.append('display_name', newUser.value.display_name)
    await api.post('/users', form)
    message.success('用户已创建')
    newUser.value = { username: '', password: '', role: 'operator', display_name: '' }
    loadUsers()
  } catch (e: any) {
    message.error(e.response?.data?.error || '创建失败')
  }
}

async function deleteUser(username: string) {
  try {
    await api.delete(`/users/${username}`)
    message.success('已删除')
    loadUsers()
  } catch (e: any) {
    message.error('删除失败')
  }
}

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

function onTabChange(key: string | number) {
  activeTab.value = String(key)
  if (key === 'users') loadUsers()
}

const userColumns = [
  { title: '用户名', dataIndex: 'username', key: 'username' },
  { title: '角色', dataIndex: 'role', key: 'role' },
  { title: '显示名', dataIndex: 'display_name', key: 'display_name' },
  { title: '操作', key: 'action', width: 120 },
]

const roleLabels: Record<string, string> = { admin: '管理员', operator: '操作员', viewer: '观察者' }
const roleColors: Record<string, string> = { admin: 'red', operator: 'blue', viewer: 'default' }
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
        <Card>
          <p style="color: #666">审计日志查看需要后端 JSON API 支持，当前可通过旧版界面 (localhost:8080/audit) 查看。</p>
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
