<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { Button, Card, Form, Input, Modal, Popconfirm, Select, Table, Tag, message } from 'ant-design-vue'

import type { RegionItem, RegionPayload } from '../../api'
import { createRegion, deleteRegion, getRegions, updateRegion } from '../../api'
import { extractErrorMessage } from '../../utils/error'

type ModalMode = 'create' | 'edit'

interface RegionFormState {
  name: string
  owner: string
  email: string
  phone: string
  notification_methods: string[]
}

const regions = ref<RegionItem[]>([])
const regionsLoading = ref(false)
const submitLoading = ref(false)
const busyRow = ref<number | null>(null)

const filters = reactive({
  name: '',
  owner: '',
  phone: '',
  email: '',
})

const modalOpen = ref(false)
const modalMode = ref<ModalMode>('create')
const editingRegionId = ref<number | null>(null)
const regionForm = ref<RegionFormState>(createEmptyForm())

const notificationMethodOptions = [
  { label: '邮箱', value: 'email', color: 'blue' },
  { label: '电话', value: 'phone', color: 'green' },
  { label: '短信', value: 'sms', color: 'orange' },
  { label: 'Webhook', value: 'webhook', color: 'purple' },
]

const notificationMethodLabelMap = Object.fromEntries(
  notificationMethodOptions.map((item) => [item.value, item.label]),
) as Record<string, string>

const notificationMethodColorMap = Object.fromEntries(
  notificationMethodOptions.map((item) => [item.value, item.color]),
) as Record<string, string>

const regionColumns = [
  { title: '区域名称', dataIndex: 'name', key: 'name', width: 180 },
  { title: '负责人', dataIndex: 'owner', key: 'owner', width: 140 },
  { title: '邮箱', dataIndex: 'email', key: 'email', width: 220 },
  { title: '电话', dataIndex: 'phone', key: 'phone', width: 160 },
  { title: '告警通知方式', key: 'notification_methods', width: 220 },
  { title: '操作', key: 'action', width: 160, fixed: 'right' as const },
]

function createEmptyForm(): RegionFormState {
  return {
    name: '',
    owner: '',
    email: '',
    phone: '',
    notification_methods: [],
  }
}

function buildQueryParams() {
  return {
    name: filters.name.trim() || undefined,
    owner: filters.owner.trim() || undefined,
    phone: filters.phone.trim() || undefined,
    email: filters.email.trim() || undefined,
  }
}

function buildPayload(): RegionPayload | null {
  const payload: RegionPayload = {
    name: regionForm.value.name.trim(),
    owner: regionForm.value.owner.trim(),
    email: regionForm.value.email.trim() || undefined,
    phone: regionForm.value.phone.trim() || undefined,
    notification_methods: regionForm.value.notification_methods,
  }

  if (!payload.name) {
    message.warning('请输入区域名称')
    return null
  }
  if (!payload.owner) {
    message.warning('请输入负责人')
    return null
  }
  if (!payload.notification_methods.length) {
    message.warning('请至少选择一种告警通知方式')
    return null
  }
  if (payload.notification_methods.includes('email') && !payload.email) {
    message.warning('通知方式包含邮箱时，请填写邮箱')
    return null
  }
  if ((payload.notification_methods.includes('phone') || payload.notification_methods.includes('sms')) && !payload.phone) {
    message.warning('通知方式包含电话或短信时，请填写电话')
    return null
  }

  return payload
}

async function loadRegions() {
  regionsLoading.value = true
  try {
    const res = await getRegions(buildQueryParams())
    regions.value = res.regions
  } catch (e) {
    message.error(extractErrorMessage(e, '加载区域列表失败'))
  } finally {
    regionsLoading.value = false
  }
}

function openCreateModal() {
  modalMode.value = 'create'
  editingRegionId.value = null
  regionForm.value = createEmptyForm()
  modalOpen.value = true
}

function openEditModal(region: RegionItem) {
  modalMode.value = 'edit'
  editingRegionId.value = region.id
  regionForm.value = {
    name: region.name,
    owner: region.owner,
    email: region.email || '',
    phone: region.phone || '',
    notification_methods: [...(region.notification_methods || [])],
  }
  modalOpen.value = true
}

function handleEdit(record: Record<string, any>) {
  openEditModal(record as RegionItem)
}

async function handleSubmit() {
  const payload = buildPayload()
  if (!payload) return

  submitLoading.value = true
  try {
    if (modalMode.value === 'create') {
      await createRegion(payload)
      message.success('区域已创建')
    } else if (editingRegionId.value != null) {
      await updateRegion(editingRegionId.value, payload)
      message.success('区域已更新')
    }
    modalOpen.value = false
    regionForm.value = createEmptyForm()
    await loadRegions()
  } catch (e) {
    message.error(extractErrorMessage(e, modalMode.value === 'create' ? '创建失败' : '更新失败'))
  } finally {
    submitLoading.value = false
  }
}

async function handleDelete(region: RegionItem) {
  busyRow.value = region.id
  try {
    await deleteRegion(region.id)
    message.success('区域已删除')
    await loadRegions()
  } catch (e) {
    message.error(extractErrorMessage(e, '删除失败'))
  } finally {
    busyRow.value = null
  }
}

function handleDeleteRecord(record: Record<string, any>) {
  return handleDelete(record as RegionItem)
}

function handleSearch() {
  loadRegions()
}

function handleReset() {
  filters.name = ''
  filters.owner = ''
  filters.phone = ''
  filters.email = ''
  loadRegions()
}

onMounted(() => {
  loadRegions()
})
</script>

<template>
  <div>
    <Card title="查询条件" style="margin-bottom: 16px">
      <Form layout="inline">
        <Form.Item>
          <Input v-model:value="filters.name" placeholder="区域名称" allow-clear @press-enter="handleSearch" />
        </Form.Item>
        <Form.Item>
          <Input v-model:value="filters.owner" placeholder="负责人" allow-clear @press-enter="handleSearch" />
        </Form.Item>
        <Form.Item>
          <Input v-model:value="filters.phone" placeholder="电话" allow-clear @press-enter="handleSearch" />
        </Form.Item>
        <Form.Item>
          <Input v-model:value="filters.email" placeholder="邮箱" allow-clear @press-enter="handleSearch" />
        </Form.Item>
        <Form.Item>
          <Button type="primary" @click="handleSearch">查询</Button>
        </Form.Item>
        <Form.Item>
          <Button @click="handleReset">重置</Button>
        </Form.Item>
      </Form>
    </Card>

    <Card title="区域列表">
      <template #extra>
        <Button type="primary" @click="openCreateModal">新增区域</Button>
      </template>

      <Table
        :columns="regionColumns"
        :data-source="regions"
        :loading="regionsLoading"
        :pagination="false"
        :scroll="{ x: 1080 }"
        row-key="id"
        size="small"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'email'">
            {{ record.email || '-' }}
          </template>
          <template v-else-if="column.key === 'phone'">
            {{ record.phone || '-' }}
          </template>
          <template v-else-if="column.key === 'notification_methods'">
            <template v-if="record.notification_methods?.length">
              <Tag
                v-for="method in record.notification_methods"
                :key="method"
                :color="notificationMethodColorMap[method] || 'default'"
              >
                {{ notificationMethodLabelMap[method] || method }}
              </Tag>
            </template>
            <template v-else>-</template>
          </template>
          <template v-else-if="column.key === 'action'">
            <Button size="small" style="margin-right: 8px" @click="handleEdit(record)">编辑</Button>
            <Popconfirm title="确定删除该区域吗？" @confirm="handleDeleteRecord(record)">
              <Button size="small" danger :loading="busyRow === record.id">删除</Button>
            </Popconfirm>
          </template>
        </template>
      </Table>
    </Card>

    <Modal
      v-model:open="modalOpen"
      :title="modalMode === 'create' ? '新增区域' : '编辑区域'"
      :confirm-loading="submitLoading"
      ok-text="保存"
      cancel-text="取消"
      width="640px"
      @ok="handleSubmit"
    >
      <Form layout="vertical" style="margin-top: 16px">
        <Form.Item label="区域名称" required>
          <Input v-model:value="regionForm.name" placeholder="请输入区域名称" />
        </Form.Item>
        <Form.Item label="负责人" required>
          <Input v-model:value="regionForm.owner" placeholder="请输入负责人" />
        </Form.Item>
        <Form.Item label="邮箱">
          <Input v-model:value="regionForm.email" placeholder="请输入邮箱" />
        </Form.Item>
        <Form.Item label="电话">
          <Input v-model:value="regionForm.phone" placeholder="请输入电话" />
        </Form.Item>
        <Form.Item label="告警通知方式" required>
          <Select
            v-model:value="regionForm.notification_methods"
            mode="multiple"
            placeholder="请选择告警通知方式"
          >
            <Select.Option
              v-for="option in notificationMethodOptions"
              :key="option.value"
              :value="option.value"
            >
              {{ option.label }}
            </Select.Option>
          </Select>
        </Form.Item>
      </Form>
    </Modal>
  </div>
</template>
