<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Card, Table, Button, Form, Input, Select, message, Popconfirm, Tag, Switch } from 'ant-design-vue'
import { getUsers as apiGetUsers, createUser as apiCreateUser, deleteUser as apiDeleteUser, toggleUserActive } from '../../api'

const users = ref<any[]>([])
const usersLoading = ref(false)
const createLoading = ref(false)
const busyRow = ref<string | null>(null)
const newUser = ref({ username: '', password: '', role: 'operator', display_name: '' })

async function loadUsers() {
  usersLoading.value = true
  try {
    const res = await apiGetUsers()
    users.value = res.users
  } catch (e) {
    message.error('加载用户列表失败')
  } finally {
    usersLoading.value = false
  }
}

async function createUser() {
  createLoading.value = true
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
  } finally {
    createLoading.value = false
  }
}

async function deleteUser(username: string) {
  busyRow.value = username
  try {
    await apiDeleteUser(username)
    message.success('已删除')
    loadUsers()
  } catch (e: any) {
    message.error('删除失败')
  } finally {
    busyRow.value = null
  }
}

async function handleToggleActive(username: string) {
  busyRow.value = username
  try {
    await toggleUserActive(username)
    message.success('状态已更新')
    loadUsers()
  } catch (e: any) {
    message.error('切换状态失败')
  } finally {
    busyRow.value = null
  }
}

onMounted(() => {
  loadUsers()
})

const userColumns = [
  { title: '用户名', dataIndex: 'username', key: 'username' },
  { title: '显示名', dataIndex: 'display_name', key: 'display_name' },
  { title: '角色', dataIndex: 'role', key: 'role' },
  { title: '状态', key: 'active', width: 100 },
  { title: '最后登录', dataIndex: 'last_login', key: 'last_login' },
  { title: '操作', key: 'action', width: 120 },
]

const roleLabels: Record<string, string> = { admin: '管理员', engineer: '工程师', operator: '操作员', viewer: '观察者' }
const roleColors: Record<string, string> = { admin: 'red', engineer: 'purple', operator: 'blue', viewer: 'default' }
</script>

<template>
  <div>
    <Card title="添加用户" style="margin-bottom: 16px">
      <Form layout="inline" @finish="createUser">
        <Form.Item :rules="[{ required: true, message: '请输入用户名' }]" name="username">
          <Input v-model:value="newUser.username" placeholder="用户名" />
        </Form.Item>
        <Form.Item :rules="[{ required: true, min: 6, message: '密码至少6位' }]" name="password">
          <Input.Password v-model:value="newUser.password" placeholder="密码 (至少6位)" />
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
          <Button type="primary" html-type="submit" :loading="createLoading">添加</Button>
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
              <Button size="small" danger :loading="busyRow === record.username">删除</Button>
            </Popconfirm>
          </template>
        </template>
      </Table>
    </Card>
  </div>
</template>
