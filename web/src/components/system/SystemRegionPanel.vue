<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { Button, Card, Form, Input, Modal, Popconfirm, Select, Table, Tag, message } from 'ant-design-vue'

import type {
  NotificationTemplateItem,
  NotificationTemplateMethod,
  RegionItem,
  RegionPayload,
} from '../../api'
import {
  createRegion,
  deleteRegion,
  getNotificationTemplates,
  getRegions,
  updateRegion,
} from '../../api'
import { extractErrorMessage } from '../../utils/error'

type ModalMode = 'create' | 'edit'
type RegionNotificationMethod = 'email' | 'phone' | 'sms' | 'webhook'

interface RegionFormState {
  name: string
  owner: string
  email: string
  phone: string
  notification_methods: RegionNotificationMethod[]
  notification_template_ids: number[]
}

const TEMPLATE_METHODS: NotificationTemplateMethod[] = ['email', 'sms', 'webhook']

const regions = ref<RegionItem[]>([])
const notificationTemplates = ref<NotificationTemplateItem[]>([])
const regionsLoading = ref(false)
const templatesLoading = ref(false)
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

const notificationMethodOptions: { label: string; value: RegionNotificationMethod; color: string }[] = [
  { label: '邮箱', value: 'email', color: 'blue' },
  { label: '电话', value: 'phone', color: 'green' },
  { label: '短信', value: 'sms', color: 'orange' },
  { label: 'Webhook', value: 'webhook', color: 'purple' },
]

const notificationMethodLabelMap = Object.fromEntries(
  notificationMethodOptions.map((item) => [item.value, item.label]),
) as Record<RegionNotificationMethod, string>

const notificationMethodColorMap = Object.fromEntries(
  notificationMethodOptions.map((item) => [item.value, item.color]),
) as Record<RegionNotificationMethod, string>

function getNotificationMethodLabel(method: string) {
  return notificationMethodLabelMap[method as RegionNotificationMethod] || method
}

function getNotificationMethodColor(method: string) {
  return notificationMethodColorMap[method as RegionNotificationMethod] || 'default'
}

const templateNameMap = computed(() => new Map(notificationTemplates.value.map((item) => [item.id, item])))

const availableNotificationTemplateOptions = computed(() => {
  const selectedMethods = regionForm.value.notification_methods.filter(
    (method): method is NotificationTemplateMethod => TEMPLATE_METHODS.includes(method as NotificationTemplateMethod),
  )
  if (!regionForm.value.notification_methods.length) {
    return notificationTemplates.value
  }
  if (!selectedMethods.length) {
    return []
  }
  return notificationTemplates.value.filter((item) => selectedMethods.includes(item.method))
})

const notificationTemplateFieldExtra = computed(() => {
  if (!regionForm.value.notification_methods.length) {
    return '可选择多个通知内容配置，保存后将与当前区域关联。'
  }
  const selectedMethods = regionForm.value.notification_methods.filter(
    (method): method is NotificationTemplateMethod => TEMPLATE_METHODS.includes(method as NotificationTemplateMethod),
  )
  if (!selectedMethods.length) {
    return '当前只选择了电话方式，没有可关联的通知内容配置。'
  }
  return '只显示与已选通知方式匹配的通知内容配置，可多选。'
})

const regionColumns = [
  { title: '区域名称', dataIndex: 'name', key: 'name', width: 180 },
  { title: '负责人', dataIndex: 'owner', key: 'owner', width: 140 },
  { title: '邮箱', dataIndex: 'email', key: 'email', width: 220 },
  { title: '电话', dataIndex: 'phone', key: 'phone', width: 160 },
  { title: '告警通知方式', key: 'notification_methods', width: 220 },
  { title: '通知内容配置', key: 'notification_templates', width: 320 },
  { title: '操作', key: 'action', width: 160, fixed: 'right' as const },
]

function createEmptyForm(): RegionFormState {
  return {
    name: '',
    owner: '',
    email: '',
    phone: '',
    notification_methods: [],
    notification_template_ids: [],
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

function filterTemplateOptionsByMethods(methods: RegionNotificationMethod[]) {
  const selectedMethods = methods.filter(
    (method): method is NotificationTemplateMethod => TEMPLATE_METHODS.includes(method as NotificationTemplateMethod),
  )
  if (!methods.length) {
    return notificationTemplates.value
  }
  if (!selectedMethods.length) {
    return []
  }
  return notificationTemplates.value.filter((item) => selectedMethods.includes(item.method))
}

function syncTemplateSelectionWithMethods(methods: RegionNotificationMethod[]) {
  const allowedIds = new Set(filterTemplateOptionsByMethods(methods).map((item) => item.id))
  regionForm.value.notification_template_ids = regionForm.value.notification_template_ids.filter((id) => allowedIds.has(id))
}

function buildPayload(): RegionPayload | null {
  const payload: RegionPayload = {
    name: regionForm.value.name.trim(),
    owner: regionForm.value.owner.trim(),
    email: regionForm.value.email.trim() || undefined,
    phone: regionForm.value.phone.trim() || undefined,
    notification_methods: [...regionForm.value.notification_methods],
    notification_template_ids: [...regionForm.value.notification_template_ids],
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

  const availableIds = new Set(availableNotificationTemplateOptions.value.map((item) => item.id))
  if (payload.notification_template_ids.some((id) => !availableIds.has(id))) {
    message.warning('所选通知内容配置与当前通知方式不匹配，请重新选择')
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

async function loadNotificationTemplates() {
  templatesLoading.value = true
  try {
    const res = await getNotificationTemplates()
    notificationTemplates.value = res.templates
    syncTemplateSelectionWithMethods(regionForm.value.notification_methods)
  } catch (e) {
    message.error(extractErrorMessage(e, '加载通知内容配置失败'))
  } finally {
    templatesLoading.value = false
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
    notification_methods: [...(region.notification_methods || [])] as RegionNotificationMethod[],
    notification_template_ids: [...(region.notification_template_ids || [])],
  }
  syncTemplateSelectionWithMethods(regionForm.value.notification_methods)
  modalOpen.value = true
}

function handleEdit(record: Record<string, any>) {
  openEditModal(record as RegionItem)
}

function handleNotificationMethodsChange(value: unknown) {
  const methods = (Array.isArray(value) ? value : []) as RegionNotificationMethod[]
  regionForm.value.notification_methods = methods
  syncTemplateSelectionWithMethods(methods)
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
  void Promise.all([loadRegions(), loadNotificationTemplates()])
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
        :scroll="{ x: 1320 }"
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
                :color="getNotificationMethodColor(method)"
              >
                {{ getNotificationMethodLabel(method) }}
              </Tag>
            </template>
            <template v-else>-</template>
          </template>
          <template v-else-if="column.key === 'notification_templates'">
            <template v-if="record.notification_templates?.length">
              <Tag
                v-for="template in record.notification_templates"
                :key="template.id"
                class="template-tag"
                :color="getNotificationMethodColor(template.method)"
              >
                {{ template.name }}{{ template.enabled ? '' : '（停用）' }}
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
      width="700px"
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
            @change="handleNotificationMethodsChange"
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
        <Form.Item label="通知内容配置" :extra="notificationTemplateFieldExtra">
          <Select
            v-model:value="regionForm.notification_template_ids"
            mode="multiple"
            show-search
            :loading="templatesLoading"
            :disabled="!availableNotificationTemplateOptions.length && !!regionForm.notification_methods.length"
            option-filter-prop="label"
            placeholder="请选择通知内容配置"
          >
            <Select.Option
              v-for="template in availableNotificationTemplateOptions"
              :key="template.id"
              :value="template.id"
              :label="`${template.name} ${getNotificationMethodLabel(template.method)} ${template.enabled ? '' : '停用'}`"
            >
              {{ template.name }}
              <span class="template-option-meta">
                {{ getNotificationMethodLabel(template.method) }}{{ template.enabled ? '' : ' / 停用' }}
              </span>
            </Select.Option>
          </Select>
        </Form.Item>
        <Form.Item
          v-if="regionForm.notification_template_ids.length"
          label="已选通知内容配置"
        >
          <div class="selected-template-list">
            <Tag
              v-for="templateId in regionForm.notification_template_ids"
              :key="templateId"
              class="template-tag"
              :color="getNotificationMethodColor(templateNameMap.get(templateId)?.method || '')"
            >
              {{ templateNameMap.get(templateId)?.name || `模板 #${templateId}` }}
            </Tag>
          </div>
        </Form.Item>
      </Form>
    </Modal>
  </div>
</template>

<style scoped>
.template-tag {
  margin-bottom: 6px;
}

.template-option-meta {
  margin-left: 8px;
  color: var(--ink-5);
  font-size: 12px;
}

.selected-template-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
</style>
