<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { Button, Card, Form, Input, Modal, Popconfirm, Space, Table, Typography, message } from 'ant-design-vue'

import type { RegionItem, RegionPayload } from '../../api'
import { createRegion, deleteRegion, getRegions, updateRegion } from '../../api'
import { extractErrorMessage } from '../../utils/error'

type ModalMode = 'create' | 'edit'

interface RegionFormState {
  name: string
  owner: string
  phone: string
}

const regions = ref<RegionItem[]>([])
const regionsLoading = ref(false)
const submitLoading = ref(false)
const busyRow = ref<number | null>(null)

const filters = reactive({
  name: '',
  owner: '',
  phone: '',
})

const modalOpen = ref(false)
const modalMode = ref<ModalMode>('create')
const editingRegionId = ref<number | null>(null)
const regionForm = ref<RegionFormState>({ name: '', owner: '', phone: '' })

const columns = [
  { title: '名称', dataIndex: 'name', key: 'name' },
  { title: '负责人', dataIndex: 'owner', key: 'owner', width: 160 },
  { title: '电话', dataIndex: 'phone', key: 'phone', width: 180 },
  { title: '更新时间', dataIndex: 'updated_at', key: 'updated_at', width: 180 },
  { title: '操作', key: 'actions', width: 180 },
]

async function loadRegions() {
  regionsLoading.value = true
  try {
    const res = await getRegions({
      name: filters.name.trim() || undefined,
      owner: filters.owner.trim() || undefined,
      phone: filters.phone.trim() || undefined,
    })
    regions.value = res.regions || []
  } catch (e) {
    message.error(extractErrorMessage(e, '加载区域失败'))
  } finally {
    regionsLoading.value = false
  }
}

onMounted(loadRegions)

function resetForm() {
  regionForm.value = { name: '', owner: '', phone: '' }
}

function openCreate() {
  modalMode.value = 'create'
  editingRegionId.value = null
  resetForm()
  modalOpen.value = true
}

function openEdit(record: RegionItem) {
  modalMode.value = 'edit'
  editingRegionId.value = record.id
  regionForm.value = {
    name: record.name,
    owner: record.owner,
    phone: record.phone || '',
  }
  modalOpen.value = true
}

async function handleSubmit() {
  if (!regionForm.value.name.trim()) {
    message.warning('区域名称不能为空')
    return
  }
  if (!regionForm.value.owner.trim()) {
    message.warning('负责人不能为空')
    return
  }
  const payload: RegionPayload = {
    name: regionForm.value.name.trim(),
    owner: regionForm.value.owner.trim(),
    phone: regionForm.value.phone.trim() || undefined,
  }
  submitLoading.value = true
  try {
    if (modalMode.value === 'create') {
      await createRegion(payload)
      message.success('已新增区域')
    } else if (editingRegionId.value != null) {
      await updateRegion(editingRegionId.value, payload)
      message.success('已更新区域')
    }
    modalOpen.value = false
    resetForm()
    await loadRegions()
  } catch (e) {
    message.error(extractErrorMessage(e, '提交失败'))
  } finally {
    submitLoading.value = false
  }
}

async function handleDelete(record: RegionItem) {
  busyRow.value = record.id
  try {
    await deleteRegion(record.id)
    message.success('已删除区域')
    await loadRegions()
  } catch (e) {
    message.error(extractErrorMessage(e, '删除失败'))
  } finally {
    busyRow.value = null
  }
}

function resetFilters() {
  filters.name = ''
  filters.owner = ''
  filters.phone = ''
  loadRegions()
}
</script>

<template>
  <main style="display: flex; flex-direction: column; gap: 16px">
    <Card :bordered="false">
      <Space wrap>
        <Input v-model:value="filters.name" placeholder="按名称筛选" allow-clear style="width: 180px" />
        <Input v-model:value="filters.owner" placeholder="按负责人筛选" allow-clear style="width: 180px" />
        <Input v-model:value="filters.phone" placeholder="按电话筛选" allow-clear style="width: 180px" />
        <Button type="primary" @click="loadRegions">查询</Button>
        <Button @click="resetFilters">重置</Button>
        <Button type="primary" @click="openCreate">新增区域</Button>
      </Space>
    </Card>

    <Card :bordered="false">
      <Typography.Paragraph style="color: var(--ink-3); font-size: 12px; margin: 0 0 12px">
        区域作为联系人/责任人卡片使用。邮件、短信、模板系统已于 2026-05 一并移除；
        Webhook 通知配置在「系统配置」面板里。
      </Typography.Paragraph>
      <Table
        :columns="columns"
        :data-source="regions"
        :loading="regionsLoading"
        :pagination="{ pageSize: 20 }"
        row-key="id"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'updated_at'">
            {{ record.updated_at?.replace('T', ' ').slice(0, 19) || '-' }}
          </template>
          <template v-else-if="column.key === 'actions'">
            <Space size="small">
              <Button size="small" @click="openEdit(record as RegionItem)">编辑</Button>
              <Popconfirm
                title="确定删除此区域吗？"
                ok-text="删除"
                ok-type="danger"
                cancel-text="取消"
                @confirm="handleDelete(record as RegionItem)"
              >
                <Button size="small" danger :loading="busyRow === record.id">删除</Button>
              </Popconfirm>
            </Space>
          </template>
        </template>
      </Table>
    </Card>

    <Modal
      v-model:open="modalOpen"
      :title="modalMode === 'create' ? '新增区域' : '编辑区域'"
      :ok-text="modalMode === 'create' ? '创建' : '保存'"
      cancel-text="取消"
      :confirm-loading="submitLoading"
      @ok="handleSubmit"
    >
      <Form layout="vertical">
        <Form.Item label="名称" required>
          <Input v-model:value="regionForm.name" placeholder="例如：核反应堆主厂房" />
        </Form.Item>
        <Form.Item label="负责人" required>
          <Input v-model:value="regionForm.owner" placeholder="姓名" />
        </Form.Item>
        <Form.Item label="电话">
          <Input v-model:value="regionForm.phone" placeholder="可选" />
        </Form.Item>
      </Form>
    </Modal>
  </main>
</template>
