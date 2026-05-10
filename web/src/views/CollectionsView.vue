<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import {
  Button, Card, Popconfirm, Space, Table, Tag, Typography, message,
} from 'ant-design-vue'
import { ReloadOutlined } from '@ant-design/icons-vue'
import {
  getBaselineCollections,
  activateCollection,
  retireCollection,
  deleteCollection,
} from '../api/baselines'
import type { BaselineCollection } from '../types/api'
import { extractErrorMessage } from '../utils/error'
import ErrorDetailModal from '../components/common/ErrorDetailModal.vue'

defineOptions({ name: 'CollectionsView' })

const router = useRouter()
const collections = ref<BaselineCollection[]>([])
const loading = ref(false)

async function load() {
  loading.value = true
  try {
    const res = await getBaselineCollections()
    collections.value = res?.collections || []
  } catch (e) {
    message.error(extractErrorMessage(e, '加载失败'))
  } finally {
    loading.value = false
  }
}

onMounted(load)

const errorModalOpen = ref(false)
const errorModalText = ref<string | null>(null)
const errorModalTitle = ref('')

function showError(c: BaselineCollection) {
  errorModalTitle.value = `${c.camera_id}/${c.zone_id}/${c.version} 失败原因`
  errorModalText.value = c.error || '（未记录失败原因）'
  errorModalOpen.value = true
}

async function handleActivate(c: BaselineCollection) {
  try {
    await activateCollection(c.camera_id, c.zone_id, c.version)
    message.success(`已激活 ${c.version}`)
    await load()
  } catch (e) {
    message.error(extractErrorMessage(e, '激活失败'))
  }
}

async function handleRetire(c: BaselineCollection) {
  try {
    await retireCollection(c.camera_id, c.zone_id, c.version)
    message.success(`已 retire ${c.version}`)
    await load()
  } catch (e) {
    message.error(extractErrorMessage(e, '操作失败'))
  }
}

async function handleDelete(c: BaselineCollection) {
  try {
    await deleteCollection(c.camera_id, c.zone_id, c.version, { confirm: c.is_current })
    message.success(`已删除 ${c.version}`)
    await load()
  } catch (e) {
    message.error(extractErrorMessage(e, '删除失败'))
  }
}

function trainWithThis(c: BaselineCollection) {
  // 痛点 5: deep-link to TrainingCreate with this version pre-selected.
  const payload = {
    items: [{
      camera_id: c.camera_id,
      zone_id: c.zone_id,
      version: c.version,
      session_label: c.session_label || undefined,
    }],
    total_frames: c.image_count,
  }
  const encoded = btoa(JSON.stringify(payload))
  router.push({ path: '/models/training', query: { preselect: encoded } })
}

const STATUS_TAG: Record<string, { color: string; label: string }> = {
  ok: { color: 'green', label: '正常' },
  partial: { color: 'orange', label: '部分' },
  failed: { color: 'red', label: '失败' },
}

const columns = [
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id', width: 120 },
  { title: '区域', dataIndex: 'zone_id', key: 'zone_id', width: 100 },
  { title: '版本', dataIndex: 'version', key: 'version' },
  { title: 'Session', dataIndex: 'session_label', key: 'session_label', width: 120 },
  { title: '状态', key: 'status', width: 110 },
  { title: '帧数', dataIndex: 'image_count', key: 'image_count', width: 90 },
  { title: '接受率', key: 'acceptance_rate', width: 100 },
  { title: '采集时间', dataIndex: 'captured_at', key: 'captured_at', width: 170 },
  { title: '生命周期', dataIndex: 'state', key: 'state', width: 110 },
  { title: '操作', key: 'actions', width: 320, fixed: 'right' as const },
]

const rows = computed(() => collections.value)
</script>

<template>
  <main class="glass" style="padding: 24px; border-radius: var(--r-lg); min-width: 0; display: flex; flex-direction: column; flex: 1;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px">
      <Typography.Title :level="3" style="margin: 0">采集集合管理</Typography.Title>
      <Button @click="load">
        <template #icon><ReloadOutlined /></template>
        刷新
      </Button>
    </div>

    <Card :bordered="false">
      <Table
        :columns="columns"
        :data-source="rows"
        :loading="loading"
        :pagination="{ pageSize: 20 }"
        row-key="version"
        size="small"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'status'">
            <Tag :color="(STATUS_TAG[record.status] || {}).color || 'default'">
              {{ (STATUS_TAG[record.status] || {}).label || record.status }}
            </Tag>
            <Tag v-if="record.is_current" color="blue">当前</Tag>
          </template>
          <template v-else-if="column.key === 'acceptance_rate'">
            <span v-if="record.acceptance_rate != null">
              {{ (record.acceptance_rate * 100).toFixed(1) }}%
            </span>
            <span v-else style="color: var(--ink-3)">—</span>
          </template>
          <template v-else-if="column.key === 'captured_at'">
            {{ record.captured_at?.replace('T', ' ').slice(0, 19) || '-' }}
          </template>
          <template v-else-if="column.key === 'actions'">
            <Space size="small">
              <Button
                v-if="record.status === 'failed'"
                size="small"
                type="link"
                danger
                @click="showError(record as BaselineCollection)"
              >查看错误</Button>
              <Button
                v-if="record.status !== 'failed' && !record.is_current"
                size="small"
                type="primary"
                @click="handleActivate(record as BaselineCollection)"
              >激活</Button>
              <Button
                v-if="record.status !== 'failed'"
                size="small"
                @click="trainWithThis(record as BaselineCollection)"
              >用这批训练</Button>
              <Button
                v-if="record.status !== 'failed' && record.state !== 'retired'"
                size="small"
                @click="handleRetire(record as BaselineCollection)"
              >Retire</Button>
              <Popconfirm
                :title="record.is_current
                  ? '此版本是当前激活基线，删除后该摄像头将没有可用基线，确定？'
                  : '确认删除此版本？'"
                ok-text="删除"
                cancel-text="取消"
                ok-type="danger"
                @confirm="handleDelete(record as BaselineCollection)"
              >
                <Button size="small" danger>删除</Button>
              </Popconfirm>
            </Space>
          </template>
        </template>
      </Table>
    </Card>

    <ErrorDetailModal
      v-model:open="errorModalOpen"
      :title="errorModalTitle"
      :error="errorModalText"
    />
  </main>
</template>
