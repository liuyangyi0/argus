<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import {
  Card, Table, Button, Space, Select, Tag, Image, Statistic, Row, Col,
  message, Modal, Empty, Badge, Progress, Tooltip,
} from 'ant-design-vue'
import {
  CheckOutlined, CloseOutlined, ForwardOutlined, ReloadOutlined,
  ExperimentOutlined,
} from '@ant-design/icons-vue'
import {
  getLabelingQueue, labelEntry, skipEntry, getLabelingStats,
  triggerRetrain, getLabelingImage,
} from '../../api/labeling'

const props = defineProps<{ cameras: Array<{ camera_id: string; name?: string }> }>()

const loading = ref(false)
const statsLoading = ref(false)
const entries = ref<any[]>([])
const stats = ref<any>({ total: 0, pending: 0, labeled: 0, skipped: 0, by_label: {} })
const selectedCamera = ref<string | undefined>(undefined)
const retrainLoading = ref(false)

// Current entry being viewed
const previewEntry = ref<any>(null)
const previewVisible = ref(false)

const columns = [
  {
    title: '',
    dataIndex: 'frame_path',
    key: 'thumbnail',
    width: 80,
    customRender: ({ record: _record }: any) => null, // handled in template
  },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id', width: 120 },
  {
    title: '异常分数',
    dataIndex: 'anomaly_score',
    key: 'anomaly_score',
    width: 100,
    customRender: ({ text: _text }: any) => null, // handled in template
  },
  {
    title: '不确定性',
    dataIndex: 'entropy',
    key: 'entropy',
    width: 100,
    customRender: ({ text: _text }: any) => null, // handled in template
  },
  {
    title: '状态',
    dataIndex: 'status',
    key: 'status',
    width: 80,
  },
  {
    title: '操作',
    key: 'actions',
    width: 200,
    fixed: 'right' as const,
  },
]

async function loadQueue() {
  loading.value = true
  try {
    const data = await getLabelingQueue({
      camera_id: selectedCamera.value,
      limit: 50,
    })
    entries.value = data.entries || []
  } catch (e: any) {
    message.error('加载标注队列失败')
  } finally {
    loading.value = false
  }
}

async function loadStats() {
  statsLoading.value = true
  try {
    stats.value = await getLabelingStats(selectedCamera.value)
  } catch {
    // stats are optional
  } finally {
    statsLoading.value = false
  }
}

async function handleLabel(entryId: number, label: 'normal' | 'anomaly') {
  try {
    await labelEntry(entryId, { label, labeled_by: 'operator' })
    message.success(label === 'normal' ? '已标记为正常' : '已标记为异常')
    // Remove from list
    entries.value = entries.value.filter((e) => e.id !== entryId)
    if (previewEntry.value?.id === entryId) previewVisible.value = false
    loadStats()
  } catch (e: any) {
    message.error('标注失败: ' + (e.response?.data?.msg || e.message))
  }
}

async function handleSkip(entryId: number) {
  try {
    await skipEntry(entryId)
    entries.value = entries.value.filter((e) => e.id !== entryId)
    if (previewEntry.value?.id === entryId) previewVisible.value = false
    loadStats()
  } catch {
    message.error('跳过失败')
  }
}

function showPreview(record: any) {
  previewEntry.value = record
  previewVisible.value = true
}

async function handleTriggerRetrain() {
  retrainLoading.value = true
  try {
    const result = await triggerRetrain({
      camera_id: selectedCamera.value,
      triggered_by: 'operator',
    })
    message.success(`训练任务已创建: ${result.job_id}，需要工程师确认`)
    loadStats()
  } catch (e: any) {
    message.error(e.response?.data?.msg || '触发训练失败')
  } finally {
    retrainLoading.value = false
  }
}

const canRetrain = computed(() => {
  const labeled = stats.value.labeled || 0
  return labeled >= 5
})

const retrainTooltip = computed(() => {
  const labeled = stats.value.labeled || 0
  if (labeled < 5) return `需要至少 5 条标注 (当前 ${labeled} 条)`
  return `使用 ${labeled} 条标注数据触发增量训练`
})

watch(selectedCamera, () => {
  loadQueue()
  loadStats()
})

onMounted(() => {
  loadQueue()
  loadStats()
})
</script>

<template>
  <div>
    <!-- Stats -->
    <Row :gutter="16" style="margin-bottom: 16px">
      <Col :span="4">
        <Card size="small">
          <Statistic title="待标注" :value="stats.pending || 0">
            <template #suffix>
              <Badge status="processing" />
            </template>
          </Statistic>
        </Card>
      </Col>
      <Col :span="4">
        <Card size="small">
          <Statistic title="已标注" :value="stats.labeled || 0" :value-style="{ color: '#52c41a' }" />
        </Card>
      </Col>
      <Col :span="4">
        <Card size="small">
          <Statistic title="标记正常" :value="stats.by_label?.normal || 0" :value-style="{ color: '#1890ff' }" />
        </Card>
      </Col>
      <Col :span="4">
        <Card size="small">
          <Statistic title="标记异常" :value="stats.by_label?.anomaly || 0" :value-style="{ color: '#ff4d4f' }" />
        </Card>
      </Col>
      <Col :span="4">
        <Card size="small">
          <Statistic title="已跳过" :value="stats.skipped || 0" :value-style="{ color: '#999' }" />
        </Card>
      </Col>
      <Col :span="4">
        <Card size="small">
          <Statistic title="总计" :value="stats.total || 0" />
        </Card>
      </Col>
    </Row>

    <!-- Toolbar -->
    <Space style="margin-bottom: 16px">
      <Select
        v-model:value="selectedCamera"
        placeholder="全部摄像头"
        allow-clear
        style="width: 200px"
      >
        <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
          {{ cam.name || cam.camera_id }}
        </Select.Option>
      </Select>

      <Button @click="loadQueue" :loading="loading">
        <template #icon><ReloadOutlined /></template>
        刷新
      </Button>

      <Tooltip :title="retrainTooltip">
        <Button
          type="primary"
          :disabled="!canRetrain"
          :loading="retrainLoading"
          @click="handleTriggerRetrain"
        >
          <template #icon><ExperimentOutlined /></template>
          触发增量训练
        </Button>
      </Tooltip>
    </Space>

    <!-- Queue table -->
    <Table
      :columns="columns"
      :data-source="entries"
      :loading="loading"
      :pagination="false"
      row-key="id"
      size="small"
      :scroll="{ x: 700 }"
    >
      <template #bodyCell="{ column, record }">
        <!-- Thumbnail -->
        <template v-if="column.key === 'thumbnail'">
          <Image
            :src="getLabelingImage(record.id)"
            :width="60"
            :height="45"
            :preview="false"
            style="cursor: pointer; border-radius: 4px; object-fit: cover"
            @click="showPreview(record)"
            fallback="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNDUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHJlY3Qgd2lkdGg9IjYwIiBoZWlnaHQ9IjQ1IiBmaWxsPSIjZjBmMGYwIi8+PHRleHQgeD0iMzAiIHk9IjI1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOTk5IiBmb250LXNpemU9IjEwIj5OL0E8L3RleHQ+PC9zdmc+"
          />
        </template>

        <!-- Score -->
        <template v-if="column.key === 'anomaly_score'">
          <Tag :color="record.anomaly_score > 0.6 ? 'red' : record.anomaly_score > 0.4 ? 'orange' : 'blue'">
            {{ record.anomaly_score?.toFixed(3) }}
          </Tag>
        </template>

        <!-- Entropy -->
        <template v-if="column.key === 'entropy'">
          <Progress
            :percent="Math.round((record.entropy || 0) * 100)"
            :size="'small'"
            :stroke-color="record.entropy > 0.9 ? '#ff4d4f' : record.entropy > 0.7 ? '#faad14' : '#52c41a'"
            :show-info="true"
            :format="() => record.entropy?.toFixed(2)"
            style="width: 80px"
          />
        </template>

        <!-- Status -->
        <template v-if="column.key === 'status'">
          <Tag :color="record.status === 'pending' ? 'blue' : record.status === 'labeled' ? 'green' : 'default'">
            {{ record.status === 'pending' ? '待标注' : record.status === 'labeled' ? '已标注' : '已跳过' }}
          </Tag>
        </template>

        <!-- Actions -->
        <template v-if="column.key === 'actions'">
          <Space>
            <Tooltip title="标记为正常 (加入基线)">
              <Button size="small" type="primary" @click="handleLabel(record.id, 'normal')">
                <template #icon><CheckOutlined /></template>
                正常
              </Button>
            </Tooltip>
            <Tooltip title="标记为异常">
              <Button size="small" danger @click="handleLabel(record.id, 'anomaly')">
                <template #icon><CloseOutlined /></template>
                异常
              </Button>
            </Tooltip>
            <Tooltip title="跳过此帧">
              <Button size="small" @click="handleSkip(record.id)">
                <template #icon><ForwardOutlined /></template>
              </Button>
            </Tooltip>
          </Space>
        </template>
      </template>

      <!-- Empty state -->
      <template #emptyText>
        <Empty description="暂无待标注帧 — 系统运行中自动采集不确定帧" />
      </template>
    </Table>

    <!-- Preview modal -->
    <Modal
      v-model:open="previewVisible"
      :title="`帧预览 — ${previewEntry?.camera_id || ''} #${previewEntry?.frame_number || ''}`"
      :footer="null"
      width="720px"
      :destroy-on-close="true"
    >
      <div v-if="previewEntry" style="text-align: center">
        <Image
          :src="getLabelingImage(previewEntry.id)"
          style="max-width: 100%; border-radius: 8px"
        />
        <Row :gutter="16" style="margin-top: 16px">
          <Col :span="8">
            <Statistic title="异常分数" :value="previewEntry.anomaly_score" :precision="4" />
          </Col>
          <Col :span="8">
            <Statistic title="不确定性" :value="previewEntry.entropy" :precision="4" />
          </Col>
          <Col :span="8">
            <Statistic title="帧号" :value="previewEntry.frame_number" />
          </Col>
        </Row>
        <Space style="margin-top: 16px">
          <Button type="primary" size="large" @click="handleLabel(previewEntry.id, 'normal')">
            <template #icon><CheckOutlined /></template>
            正常 (加入基线)
          </Button>
          <Button danger size="large" @click="handleLabel(previewEntry.id, 'anomaly')">
            <template #icon><CloseOutlined /></template>
            异常
          </Button>
          <Button size="large" @click="handleSkip(previewEntry.id)">
            <template #icon><ForwardOutlined /></template>
            跳过
          </Button>
        </Space>
      </div>
    </Modal>
  </div>
</template>
