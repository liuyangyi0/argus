<!-- 现在模板只负责“属于哪种通知方式 + 内容配置 + 是否启用”。后面区域管理里选择具体模板时，再按选择的模板发送。 -->
<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  Modal,
  Popconfirm,
  Select,
  Space,
  Switch,
  Table,
  Tag,
  Typography,
  message,
} from 'ant-design-vue'
import { PlusOutlined, ReloadOutlined } from '@ant-design/icons-vue'

import {
  createNotificationTemplate,
  deleteNotificationTemplate,
  getNotificationTemplates,
  updateNotificationTemplate,
  type NotificationTemplateItem,
  type NotificationTemplateMethod,
  type NotificationTemplatePayload,
} from '../../api'
import { extractErrorMessage } from '../../utils/error'

defineOptions({ name: 'SystemNotificationTemplatePanel' })

type ModalMode = 'create' | 'edit'
type TemplateField = 'subject' | 'content'

interface TemplateFormState {
  name: string
  method: NotificationTemplateMethod
  subject: string
  content: string
  enabled: boolean
}

interface PlaceholderItem {
  token: string
  label: string
  sample: string
}

const methodOptions: { label: string; value: NotificationTemplateMethod; color: string }[] = [
  { label: '邮箱', value: 'email', color: 'blue' },
  { label: '短信', value: 'sms', color: 'orange' },
  { label: 'Webhook', value: 'webhook', color: 'purple' },
]

const placeholders: PlaceholderItem[] = [
  { token: 'alert_id', label: '告警编号', sample: 'ALERT-20260419-001' },
  { token: 'camera_id', label: '摄像头 ID', sample: 'CAM-01' },
  { token: 'camera_name', label: '摄像头名称', sample: '一号机大厅入口' },
  { token: 'region_name', label: '区域名称', sample: '反应堆厂房 A 区' },
  { token: 'severity', label: '严重度', sample: 'high' },
  { token: 'category', label: '告警类型', sample: 'foreign_object' },
  { token: 'anomaly_score', label: '异常分数', sample: '0.92' },
  { token: 'timestamp', label: '发生时间', sample: '2026-04-19 14:32:08' },
  { token: 'zone_id', label: '检测区域', sample: 'critical-zone-1' },
  { token: 'owner', label: '负责人', sample: '张工' },
  { token: 'phone', label: '联系电话', sample: '13800000000' },
  { token: 'email', label: '联系邮箱', sample: 'ops@example.com' },
  { token: 'snapshot_url', label: '告警快照', sample: '/alerts/ALERT-20260419-001/snapshot.jpg' },
]

const placeholderSampleMap = Object.fromEntries(
  placeholders.map((item) => [item.token, item.sample]),
) as Record<string, string>
const methodLabelMap = Object.fromEntries(methodOptions.map((item) => [item.value, item.label])) as Record<string, string>
const methodColorMap = Object.fromEntries(methodOptions.map((item) => [item.value, item.color])) as Record<string, string>

const templates = ref<NotificationTemplateItem[]>([])
const loading = ref(false)
const submitting = ref(false)
const deletingId = ref<number | null>(null)
const error = ref<string | null>(null)
const methodFilter = ref<NotificationTemplateMethod | undefined>(undefined)

const modalOpen = ref(false)
const modalMode = ref<ModalMode>('create')
const editingId = ref<number | null>(null)
const activeField = ref<TemplateField>('content')
const formState = reactive<TemplateFormState>(createEmptyForm())

const templateColumns = [
  { title: '模板名称', dataIndex: 'name', key: 'name', width: 220 },
  { title: '通知方式', dataIndex: 'method', key: 'method', width: 110 },
  { title: '标题/摘要', key: 'summary', width: 260 },
  { title: '状态', key: 'state', width: 130 },
  { title: '更新时间', dataIndex: 'updated_at', key: 'updated_at', width: 190 },
  { title: '操作', key: 'action', width: 170, fixed: 'right' as const },
]

const contentRows = computed(() => (formState.method === 'webhook' ? 10 : 7))
const previewSubject = computed(() => renderTemplate(formState.subject))
const previewContent = computed(() => renderTemplate(formState.content))

function createEmptyForm(): TemplateFormState {
  return {
    name: '',
    method: 'email',
    subject: '',
    content: '',
    enabled: true,
  }
}

function resetForm(next: TemplateFormState = createEmptyForm()) {
  formState.name = next.name
  formState.method = next.method
  formState.subject = next.subject
  formState.content = next.content
  formState.enabled = next.enabled
  activeField.value = 'content'
}

function buildDefaultContent(method: NotificationTemplateMethod) {
  if (method === 'sms') {
    return '{camera_name} 在 {timestamp} 触发 {severity} 告警，区域 {region_name}，请及时处理。'
  }
  if (method === 'webhook') {
    return [
      '{',
      '  "alert_id": "{alert_id}",',
      '  "camera_id": "{camera_id}",',
      '  "camera_name": "{camera_name}",',
      '  "region_name": "{region_name}",',
      '  "severity": "{severity}",',
      '  "category": "{category}",',
      '  "anomaly_score": "{anomaly_score}",',
      '  "timestamp": "{timestamp}"',
      '}',
    ].join('\n')
  }
  return [
    '告警编号: {alert_id}',
    '摄像头: {camera_name}',
    '区域: {region_name}',
    '严重度: {severity}',
    '告警类型: {category}',
    '异常分数: {anomaly_score}',
    '发生时间: {timestamp}',
    '请尽快登录 Argus 查看现场画面。',
  ].join('\n')
}

function buildPayload(): NotificationTemplatePayload | null {
  const payload: NotificationTemplatePayload = {
    name: formState.name.trim(),
    method: formState.method,
    subject: formState.subject.trim(),
    content: formState.content.trim(),
    enabled: formState.enabled,
  }

  if (!payload.name) {
    message.warning('请输入模板名称')
    return null
  }
  if (!payload.content) {
    message.warning('请输入模板内容')
    return null
  }
  return payload
}

function renderTemplate(value: string) {
  return value.replace(/\{([a-zA-Z0-9_]+)\}/g, (match, token: string) => (
    placeholderSampleMap[token] ?? match
  ))
}

function formatTime(value?: string | null) {
  if (!value) return '-'
  return value.replace('T', ' ').slice(0, 19)
}

function formatPlaceholder(token: string) {
  return `{${token}}`
}

function insertPlaceholder(token: string) {
  const field = activeField.value
  const spacer = formState[field] && !formState[field].endsWith(' ') ? ' ' : ''
  formState[field] = `${formState[field]}${spacer}{${token}}`
}

async function loadTemplates() {
  loading.value = true
  error.value = null
  try {
    const res = await getNotificationTemplates(methodFilter.value)
    templates.value = res.templates
  } catch (e) {
    error.value = extractErrorMessage(e, '加载模板列表失败')
  } finally {
    loading.value = false
  }
}

function openCreateModal() {
  modalMode.value = 'create'
  editingId.value = null
  resetForm({
    ...createEmptyForm(),
    method: methodFilter.value || 'email',
    subject: methodFilter.value === 'email' || !methodFilter.value ? '[{severity}] {camera_name} 告警' : '',
    content: buildDefaultContent(methodFilter.value || 'email'),
  })
  modalOpen.value = true
}

function openEditModal(record: NotificationTemplateItem) {
  modalMode.value = 'edit'
  editingId.value = record.id
  resetForm({
    name: record.name,
    method: record.method,
    subject: record.subject || '',
    content: record.content || '',
    enabled: record.enabled,
  })
  modalOpen.value = true
}

function handleMethodChange(value: unknown) {
  const method = value as NotificationTemplateMethod
  if (!method) return
  if (modalMode.value === 'create' && !formState.name) {
    formState.name = `${methodLabelMap[method]}告警模板`
  }
  if (modalMode.value === 'create') {
    formState.subject = method === 'email' ? '[{severity}] {camera_name} 告警' : ''
    formState.content = buildDefaultContent(method)
  }
  activeField.value = 'content'
}

async function handleSubmit() {
  const payload = buildPayload()
  if (!payload) return

  submitting.value = true
  try {
    if (modalMode.value === 'create') {
      await createNotificationTemplate(payload)
      message.success('模板已创建')
    } else if (editingId.value != null) {
      await updateNotificationTemplate(editingId.value, payload)
      message.success('模板已更新')
    }
    modalOpen.value = false
    await loadTemplates()
  } catch (e) {
    message.error(extractErrorMessage(e, modalMode.value === 'create' ? '创建模板失败' : '更新模板失败'))
  } finally {
    submitting.value = false
  }
}

async function handleDelete(record: NotificationTemplateItem) {
  deletingId.value = record.id
  try {
    await deleteNotificationTemplate(record.id)
    message.success('模板已删除')
    await loadTemplates()
  } catch (e) {
    message.error(extractErrorMessage(e, '删除模板失败'))
  } finally {
    deletingId.value = null
  }
}

onMounted(() => {
  loadTemplates()
})
</script>

<template>
  <div>
    <Card title="通知内容配置">
      <template #extra>
        <Space>
          <Select
            v-model:value="methodFilter"
            allow-clear
            placeholder="通知方式"
            style="width: 140px"
            @change="loadTemplates"
          >
            <Select.Option v-for="method in methodOptions" :key="method.value" :value="method.value">
              {{ method.label }}
            </Select.Option>
          </Select>
          <Button :loading="loading" @click="loadTemplates">
            <template #icon><ReloadOutlined /></template>刷新
          </Button>
          <Button type="primary" @click="openCreateModal">
            <template #icon><PlusOutlined /></template>新增模板
          </Button>
        </Space>
      </template>

      <Alert
        v-if="error"
        type="error"
        :message="error"
        show-icon
        style="margin-bottom: 16px"
      />

      <Table
        :columns="templateColumns"
        :data-source="templates"
        :loading="loading"
        :pagination="{ pageSize: 10, showSizeChanger: false }"
        :scroll="{ x: 1080 }"
        row-key="id"
        size="small"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'method'">
            <Tag :color="methodColorMap[record.method] || 'default'">
              {{ methodLabelMap[record.method] || record.method }}
            </Tag>
          </template>
          <template v-else-if="column.key === 'summary'">
            <div class="summary-cell">
              <Typography.Text v-if="record.subject" strong>{{ record.subject }}</Typography.Text>
              <Typography.Text type="secondary">{{ record.content }}</Typography.Text>
            </div>
          </template>
          <template v-else-if="column.key === 'state'">
            <Space size="small">
              <Tag :color="record.enabled ? 'green' : 'default'">{{ record.enabled ? '启用' : '停用' }}</Tag>
            </Space>
          </template>
          <template v-else-if="column.key === 'updated_at'">
            {{ formatTime(record.updated_at) }}
          </template>
          <template v-else-if="column.key === 'action'">
            <Space size="small">
              <Button size="small" @click="openEditModal(record as NotificationTemplateItem)">
                编辑
              </Button>
              <Popconfirm title="确定删除该模板吗？" @confirm="handleDelete(record as NotificationTemplateItem)">
                <Button size="small" danger :loading="deletingId === record.id">
                 删除
                </Button>
              </Popconfirm>
            </Space>
          </template>
        </template>
      </Table>
    </Card>

    <Modal
      v-model:open="modalOpen"
      :title="modalMode === 'create' ? '新增通知模板' : '编辑通知模板'"
      :confirm-loading="submitting"
      ok-text="保存"
      cancel-text="取消"
      width="920px"
      @ok="handleSubmit"
    >
      <div class="modal-grid">
        <Form layout="vertical">
          <Form.Item label="通知方式" required>
            <Select v-model:value="formState.method" @change="handleMethodChange">
              <Select.Option v-for="method in methodOptions" :key="method.value" :value="method.value">
                {{ method.label }}
              </Select.Option>
            </Select>
          </Form.Item>
          <Form.Item label="模板名称" required>
            <Input v-model:value="formState.name" placeholder="请输入模板名称" />
          </Form.Item>
          <Form.Item v-if="formState.method === 'email'" label="邮件标题">
            <Input
              v-model:value="formState.subject"
              placeholder="如: [{severity}] {camera_name} 告警"
              @focus="activeField = 'subject'"
            />
          </Form.Item>
          <Form.Item label="模板内容" required>
            <Input.TextArea
              v-model:value="formState.content"
              :rows="contentRows"
              placeholder="请输入模板内容，可使用占位符"
              @focus="activeField = 'content'"
            />
            <div class="content-counter">{{ formState.content.length }} / 4000</div>
          </Form.Item>
          <div class="switch-row">
            <span>启用模板</span>
            <Switch v-model:checked="formState.enabled" />
          </div>
        </Form>

        <section class="side-panel">
          <Typography.Text strong>占位符</Typography.Text>
          <div class="placeholder-tags">
            <Tag
              v-for="item in placeholders"
              :key="item.token"
              class="placeholder-tag"
              @click="insertPlaceholder(item.token)"
            >
              {{ item.label }} {{ formatPlaceholder(item.token) }}
            </Tag>
          </div>

          <Typography.Text strong style="display: block; margin-top: 16px">发送预览</Typography.Text>
          <div v-if="formState.method === 'email'" class="preview-block">
            <Typography.Text type="secondary">标题</Typography.Text>
            <pre>{{ previewSubject || '-' }}</pre>
          </div>
          <div class="preview-block">
            <Typography.Text type="secondary">内容</Typography.Text>
            <pre>{{ previewContent || '-' }}</pre>
          </div>
        </section>
      </div>
    </Modal>
  </div>
</template>

<style scoped>
.summary-cell {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.summary-cell :deep(.ant-typography) {
  max-width: 240px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.modal-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 18px;
  margin-top: 12px;
}

.switch-row {
  min-height: 32px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  border-top: 1px solid var(--line-2);
}

.content-counter {
  margin-top: 4px;
  color: var(--ink-5);
  font-size: 12px;
  text-align: right;
}

.side-panel {
  min-width: 0;
  border: 1px solid var(--line-2);
  border-radius: 6px;
  padding: 14px;
  background: rgba(255, 255, 255, 0.66);
}

.placeholder-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.placeholder-tag {
  margin-inline-end: 0;
  cursor: pointer;
  user-select: none;
}

.preview-block {
  margin-top: 10px;
}

.preview-block pre {
  margin: 6px 0 0;
  min-height: 42px;
  max-height: 220px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  padding: 10px;
  border-radius: 6px;
  border: 1px solid var(--line-2);
  background: rgba(15, 23, 42, 0.04);
  color: var(--ink);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  line-height: 1.7;
}

@media (max-width: 900px) {
  .modal-grid {
    grid-template-columns: 1fr;
  }
}
</style>
