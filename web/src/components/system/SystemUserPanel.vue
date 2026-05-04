<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { Button, Card, Form, Input, Modal, Popconfirm, Select, Switch, Table, Tag, message } from 'ant-design-vue'

import type { CreateUserPayload, UpdateUserPayload, UserItem } from '../../api'
import { createUser, deleteUser, getUsers, updateUser } from '../../api'
import { useAuthStore } from '../../stores/useAuthStore'
import { extractErrorMessage } from '../../utils/error'

// User-management writes are admin-only. Even though the parent route is
// already admin-gated, hide the buttons here too — the panel could be reused
// in a non-admin context, and double gating prevents 403 toasts on click.
const auth = useAuthStore()

type ModalMode = 'create' | 'edit'

interface UserFormState {
  username: string
  password: string
  display_name: string
  role: UserItem['role']
  active: boolean
}

const users = ref<UserItem[]>([])
const usersLoading = ref(false)
const submitLoading = ref(false)
const busyRow = ref<string | null>(null)

const filters = reactive({
  username: '',
  display_name: '',
  role: undefined as UserItem['role'] | undefined,
})

const modalOpen = ref(false)
const modalMode = ref<ModalMode>('create')
const editingUsername = ref<string | null>(null)
const userForm = ref<UserFormState>(createEmptyForm())

const roleOptions = [
  { label: '管理员', value: 'admin', color: 'red' },
  { label: '操作员', value: 'operator', color: 'blue' },
  { label: '观察者', value: 'viewer', color: 'default' },
] satisfies Array<{ label: string; value: UserItem['role']; color: string }>

const roleLabelMap = Object.fromEntries(
  roleOptions.map((item) => [item.value, item.label]),
) as Record<UserItem['role'], string>

const roleColorMap = Object.fromEntries(
  roleOptions.map((item) => [item.value, item.color]),
) as Record<UserItem['role'], string>

const userColumns = [
  { title: '用户名', dataIndex: 'username', key: 'username', width: 180 },
  { title: '显示名', dataIndex: 'display_name', key: 'display_name', width: 180 },
  { title: '角色', dataIndex: 'role', key: 'role', width: 120 },
  { title: '状态', key: 'active', width: 100 },
  { title: '最后登录', dataIndex: 'last_login', key: 'last_login', width: 220 },
  { title: '操作', key: 'action', width: 160, fixed: 'right' as const },
]

function createEmptyForm(): UserFormState {
  return {
    username: '',
    password: '',
    display_name: '',
    role: 'operator',
    active: true,
  }
}

function buildQueryParams() {
  return {
    username: filters.username.trim() || undefined,
    display_name: filters.display_name.trim() || undefined,
    role: filters.role || undefined,
  }
}

function buildCreatePayload(): CreateUserPayload | null {
  const payload: CreateUserPayload = {
    username: userForm.value.username.trim(),
    password: userForm.value.password,
    display_name: userForm.value.display_name.trim(),
    role: userForm.value.role,
    active: userForm.value.active,
  }

  if (!payload.username) {
    message.warning('请输入用户名')
    return null
  }
  if (!payload.password) {
    message.warning('请输入密码')
    return null
  }
  if (payload.password.length < 6) {
    message.warning('密码至少 6 位')
    return null
  }

  return payload
}

function buildUpdatePayload(): UpdateUserPayload | null {
  const payload: UpdateUserPayload = {
    password: userForm.value.password || undefined,
    display_name: userForm.value.display_name.trim(),
    role: userForm.value.role,
    active: userForm.value.active,
  }

  if (payload.password && payload.password.length < 6) {
    message.warning('密码至少 6 位')
    return null
  }

  return payload
}

async function loadUsers() {
  usersLoading.value = true
  try {
    const res = await getUsers(buildQueryParams())
    users.value = res.users
  } catch (e) {
    message.error(extractErrorMessage(e, '加载用户列表失败'))
  } finally {
    usersLoading.value = false
  }
}

function openCreateModal() {
  modalMode.value = 'create'
  editingUsername.value = null
  userForm.value = createEmptyForm()
  modalOpen.value = true
}

function openEditModal(user: UserItem) {
  modalMode.value = 'edit'
  editingUsername.value = user.username
  userForm.value = {
    username: user.username,
    password: '',
    display_name: user.display_name || '',
    role: user.role,
    active: user.active,
  }
  modalOpen.value = true
}

function handleEdit(record: Record<string, any>) {
  openEditModal(record as UserItem)
}

function getRoleLabel(role: unknown) {
  const nextRole = role as UserItem['role']
  return roleLabelMap[nextRole] || String(role || '')
}

function getRoleColor(role: unknown) {
  const nextRole = role as UserItem['role']
  return roleColorMap[nextRole] || 'default'
}

async function handleSubmit() {
  submitLoading.value = true
  try {
    if (modalMode.value === 'create') {
      const payload = buildCreatePayload()
      if (!payload) return
      await createUser(payload)
      message.success('用户已创建')
    } else if (editingUsername.value) {
      const payload = buildUpdatePayload()
      if (!payload) return
      await updateUser(editingUsername.value, payload)
      message.success('用户已更新')
    }

    modalOpen.value = false
    userForm.value = createEmptyForm()
    await loadUsers()
  } catch (e) {
    message.error(extractErrorMessage(e, modalMode.value === 'create' ? '创建失败' : '更新失败'))
  } finally {
    submitLoading.value = false
  }
}

async function handleDelete(user: UserItem) {
  busyRow.value = user.username
  try {
    await deleteUser(user.username)
    message.success('用户已删除')
    await loadUsers()
  } catch (e) {
    message.error(extractErrorMessage(e, '删除失败'))
  } finally {
    busyRow.value = null
  }
}

function handleDeleteRecord(record: Record<string, any>) {
  return handleDelete(record as UserItem)
}

function handleSearch() {
  loadUsers()
}

function handleReset() {
  filters.username = ''
  filters.display_name = ''
  filters.role = undefined
  loadUsers()
}

onMounted(() => {
  loadUsers()
})
</script>

<template>
  <div>
    <Card title="查询条件" style="margin-bottom: 16px">
      <Form layout="inline">
        <Form.Item>
          <Input v-model:value="filters.username" placeholder="用户名" allow-clear @press-enter="handleSearch" />
        </Form.Item>
        <Form.Item>
          <Input v-model:value="filters.display_name" placeholder="显示名" allow-clear @press-enter="handleSearch" />
        </Form.Item>
        <Form.Item>
          <Select
            v-model:value="filters.role"
            placeholder="角色"
            allow-clear
            style="width: 140px"
          >
            <Select.Option
              v-for="option in roleOptions"
              :key="option.value"
              :value="option.value"
            >
              {{ option.label }}
            </Select.Option>
          </Select>
        </Form.Item>
        <Form.Item>
          <Button type="primary" @click="handleSearch">查询</Button>
        </Form.Item>
        <Form.Item>
          <Button @click="handleReset">重置</Button>
        </Form.Item>
      </Form>
    </Card>

    <Card title="用户列表">
      <template #extra>
        <Button v-if="auth.hasRole(['admin'])" type="primary" @click="openCreateModal">新增用户</Button>
      </template>

      <Table
        :columns="userColumns"
        :data-source="users"
        :loading="usersLoading"
        :pagination="false"
        :scroll="{ x: 980 }"
        row-key="username"
        size="small"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'role'">
            <Tag :color="getRoleColor(record.role)">
              {{ getRoleLabel(record.role) }}
            </Tag>
          </template>
          <template v-else-if="column.key === 'active'">
            <Tag :color="record.active ? 'green' : 'default'">
              {{ record.active ? '启用' : '禁用' }}
            </Tag>
          </template>
          <template v-else-if="column.key === 'last_login'">
            {{ record.last_login || '-' }}
          </template>
          <template v-else-if="column.key === 'action'">
            <Button v-if="auth.hasRole(['admin'])" size="small" style="margin-right: 8px" @click="handleEdit(record)">编辑</Button>
            <Popconfirm v-if="auth.hasRole(['admin'])" title="确定删除该用户吗？" @confirm="handleDeleteRecord(record)">
              <Button size="small" danger :loading="busyRow === record.username">删除</Button>
            </Popconfirm>
          </template>
        </template>
      </Table>
    </Card>

    <Modal
      v-model:open="modalOpen"
      :title="modalMode === 'create' ? '新增用户' : '编辑用户'"
      :confirm-loading="submitLoading"
      ok-text="保存"
      cancel-text="取消"
      width="640px"
      @ok="handleSubmit"
    >
      <Form layout="vertical" style="margin-top: 16px">
        <Form.Item label="用户名" required>
          <Input
            v-model:value="userForm.username"
            placeholder="请输入用户名"
            :disabled="modalMode === 'edit'"
          />
        </Form.Item>
        <Form.Item :label="modalMode === 'create' ? '密码' : '密码（留空则不修改）'" :required="modalMode === 'create'">
          <Input.Password
            v-model:value="userForm.password"
            :placeholder="modalMode === 'create' ? '请输入密码，至少 6 位' : '如需重置密码，请输入新密码'"
          />
        </Form.Item>
        <Form.Item label="显示名">
          <Input v-model:value="userForm.display_name" placeholder="请输入显示名" />
        </Form.Item>
        <Form.Item label="角色" required>
          <Select v-model:value="userForm.role">
            <Select.Option
              v-for="option in roleOptions"
              :key="option.value"
              :value="option.value"
            >
              {{ option.label }}
            </Select.Option>
          </Select>
        </Form.Item>
        <Form.Item label="状态">
          <Switch
            v-model:checked="userForm.active"
            checked-children="启用"
            un-checked-children="禁用"
          />
        </Form.Item>
      </Form>
    </Modal>
  </div>
</template>
