<script setup lang="ts">
import { ref, computed } from 'vue'
import {
  Card, Table, Button, Tag, Space, Modal, Form, Select, Input,
  Descriptions, Drawer, Steps, message, Tooltip,
} from 'ant-design-vue'
import {
  ReloadOutlined, SafetyCertificateOutlined,
  ExperimentOutlined, HistoryOutlined, LoadingOutlined,
} from '@ant-design/icons-vue'
import {
  promoteModel, retireModel,
  getStageHistory, getShadowReport,
} from '../../api'
import { STAGE_MAP, VALID_TRANSITIONS, STAGE_LABELS } from '../../composables/useModelState'

const props = defineProps<{
  cameras: any[]
  models: any[]
}>()

const emit = defineEmits<{
  changed: []
}>()

const promotingModel = ref<string | null>(null)
const promoteModalVisible = ref(false)
const promoteForm = ref({ version_id: '', target_stage: '', triggered_by: '', reason: '', canary_camera_id: '' })
const shadowReport = ref<any>(null)
const shadowReportVisible = ref(false)
const shadowReportLoading = ref(false)
const stageHistoryVisible = ref(false)
const stageHistory = ref<any[]>([])
const stageHistoryLoading = ref(false)

// ── Pipeline stats ──
const stageCounts = computed(() => {
  const counts: Record<string, number> = { candidate: 0, shadow: 0, canary: 0, production: 0, retired: 0 }
  for (const m of props.models) {
    if (counts[m.stage] !== undefined) counts[m.stage]++
  }
  return counts
})

const sortedModels = computed(() => {
  const order: Record<string, number> = { production: 0, canary: 1, shadow: 2, candidate: 3, retired: 4 }
  return [...props.models].sort((a: any, b: any) =>
    (order[a.stage] ?? 5) - (order[b.stage] ?? 5)
  )
})

// ── Actions ──
function handlePromote(record: any) {
  const transitions = VALID_TRANSITIONS[record.stage] || []
  if (transitions.length === 0) {
    message.warning('该模型当前阶段不支持推进')
    return
  }
  promoteForm.value = {
    version_id: record.model_version_id,
    target_stage: transitions[0],
    triggered_by: '',
    reason: '',
    canary_camera_id: '',
  }
  promoteModalVisible.value = true
}

async function submitPromote() {
  if (!promoteForm.value.triggered_by) {
    message.warning('请输入操作人')
    return
  }
  if (promoteForm.value.target_stage === 'canary' && !promoteForm.value.canary_camera_id) {
    message.warning('金丝雀阶段需要选择目标摄像头')
    return
  }
  promotingModel.value = promoteForm.value.version_id
  try {
    await promoteModel(promoteForm.value.version_id, {
      target_stage: promoteForm.value.target_stage,
      triggered_by: promoteForm.value.triggered_by,
      reason: promoteForm.value.reason || undefined,
      canary_camera_id: promoteForm.value.canary_camera_id || undefined,
    })
    message.success(`模型已推进到 ${STAGE_MAP[promoteForm.value.target_stage]?.text || promoteForm.value.target_stage}`)
    promoteModalVisible.value = false
    emit('changed')
  } catch (e: any) {
    message.error(e.response?.data?.error || '推进失败')
  } finally {
    promotingModel.value = null
  }
}

function handleRetire(record: any) {
  Modal.confirm({
    title: '确认退役',
    content: `确定要将模型 ${record.model_version_id} 标记为退役吗？模型文件将保留。`,
    okText: '确认退役',
    cancelText: '取消',
    okType: 'danger',
    async onOk() {
      try {
        await retireModel(record.model_version_id, { triggered_by: 'operator' })
        message.success('模型已退役')
        emit('changed')
      } catch (e: any) {
        message.error(e.response?.data?.error || '退役失败')
      }
    },
  })
}

async function handleViewShadowReport(record: any) {
  shadowReportLoading.value = true
  shadowReportVisible.value = true
  shadowReport.value = null
  try {
    const res = await getShadowReport(record.model_version_id, { days: 7 })
    shadowReport.value = res.data
  } catch (e: any) {
    message.error(e.response?.data?.error || '获取影子报告失败')
  } finally {
    shadowReportLoading.value = false
  }
}

async function handleViewStageHistory(record: any) {
  stageHistoryLoading.value = true
  stageHistoryVisible.value = true
  stageHistory.value = []
  try {
    const res = await getStageHistory(record.model_version_id)
    stageHistory.value = res.data.events || []
  } catch (e: any) {
    message.error(e.response?.data?.error || '获取历史失败')
  } finally {
    stageHistoryLoading.value = false
  }
}

const releaseColumns = [
  { title: '版本 ID', dataIndex: 'model_version_id', key: 'version_id', ellipsis: true },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '模型类型', dataIndex: 'model_type', key: 'model_type' },
  { title: '阶段', key: 'stage', width: 100 },
  { title: '组件', key: 'component_type', width: 80 },
  { title: '创建时间', dataIndex: 'created_at', key: 'created_at', width: 160 },
  { title: '操作', key: 'action', width: 300 },
]
</script>

<template>
  <!-- Pipeline Steps visualization -->
  <Card style="margin-bottom: 16px">
    <Steps :current="-1" size="small" style="padding: 8px 0">
      <Steps.Step title="候选" :description="`${stageCounts.candidate} 个模型`" status="process" />
      <Steps.Step title="影子" :description="`${stageCounts.shadow} 个模型`" :status="stageCounts.shadow > 0 ? 'process' : 'wait'" />
      <Steps.Step title="金丝雀" :description="`${stageCounts.canary} 个模型`" :status="stageCounts.canary > 0 ? 'process' : 'wait'" />
      <Steps.Step title="生产" :description="`${stageCounts.production} 个模型`" :status="stageCounts.production > 0 ? 'finish' : 'wait'" />
    </Steps>
  </Card>

  <!-- Release table -->
  <Card style="margin-bottom: 16px">
    <template #title>
      <div style="display: flex; justify-content: space-between; align-items: center">
        <Space>
          <SafetyCertificateOutlined />
          <span>发布流水线</span>
        </Space>
        <Button @click="$emit('changed')">
          <template #icon><ReloadOutlined /></template>
          刷新
        </Button>
      </div>
    </template>
    <p style="color: #8890a0; margin-bottom: 16px">
      四阶段发布流程：候选 → 影子 → 金丝雀 → 生产。每次推进需工程师显式操作。
    </p>
    <Table
      :columns="releaseColumns"
      :data-source="sortedModels"
      :pagination="{ pageSize: 15, showSizeChanger: false }"
      row-key="model_version_id"
      size="small"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'version_id'">
          <span style="font-family: monospace; font-size: 12px">{{ record.model_version_id }}</span>
        </template>
        <template v-if="column.key === 'stage'">
          <Tag :color="(STAGE_MAP[record.stage] || { color: 'default' }).color">
            {{ (STAGE_MAP[record.stage] || { text: record.stage }).text }}
          </Tag>
        </template>
        <template v-if="column.key === 'component_type'">
          <Tag v-if="record.component_type === 'head'" color="cyan">Head</Tag>
          <Tag v-else-if="record.component_type === 'backbone'" color="geekblue">Backbone</Tag>
          <Tag v-else>Full</Tag>
        </template>
        <template v-if="column.key === 'created_at'">
          {{ record.created_at ? record.created_at.replace('T', ' ').substring(0, 16) : '-' }}
        </template>
        <template v-if="column.key === 'action'">
          <Space>
            <Button
              size="small"
              type="primary"
              :disabled="record.stage === 'retired' || record.stage === 'production'"
              :loading="promotingModel === record.model_version_id"
              @click="handlePromote(record)"
            >
              推进
            </Button>
            <Button
              size="small"
              danger
              :disabled="record.stage === 'retired'"
              @click="handleRetire(record)"
            >
              退役
            </Button>
            <Tooltip title="查看影子推理报告" v-if="record.stage === 'shadow'">
              <Button size="small" @click="handleViewShadowReport(record)">
                <template #icon><ExperimentOutlined /></template>
              </Button>
            </Tooltip>
            <Tooltip title="阶段变更历史">
              <Button size="small" @click="handleViewStageHistory(record)">
                <template #icon><HistoryOutlined /></template>
              </Button>
            </Tooltip>
          </Space>
        </template>
      </template>
    </Table>
    <div v-if="models.length === 0" style="text-align: center; padding: 32px; color: #666">
      暂无注册的模型版本
    </div>
  </Card>

  <!-- Promote Modal -->
  <Modal
    v-model:open="promoteModalVisible"
    title="推进模型阶段"
    :confirmLoading="promotingModel !== null"
    @ok="submitPromote"
    okText="确认推进"
    cancelText="取消"
  >
    <Form layout="vertical" style="margin-top: 16px">
      <Form.Item label="目标阶段">
        <Select v-model:value="promoteForm.target_stage" style="width: 100%">
          <Select.Option
            v-for="stage in (VALID_TRANSITIONS[models.find(m => m.model_version_id === promoteForm.version_id)?.stage] || [])"
            :key="stage"
            :value="stage"
          >
            {{ STAGE_LABELS[stage] || stage }}
          </Select.Option>
        </Select>
      </Form.Item>
      <Form.Item label="操作人" required>
        <Input v-model:value="promoteForm.triggered_by" placeholder="输入操作人姓名" />
      </Form.Item>
      <Form.Item v-if="promoteForm.target_stage === 'canary'" label="金丝雀摄像头" required>
        <Select v-model:value="promoteForm.canary_camera_id" placeholder="选择目标摄像头" style="width: 100%">
          <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
            {{ cam.camera_id }}
          </Select.Option>
        </Select>
      </Form.Item>
      <Form.Item label="原因">
        <Input.TextArea v-model:value="promoteForm.reason" :rows="2" placeholder="推进原因（可选）" />
      </Form.Item>
    </Form>
  </Modal>

  <!-- Shadow Report Drawer -->
  <Drawer
    v-model:open="shadowReportVisible"
    title="影子推理报告"
    :width="480"
  >
    <div v-if="shadowReportLoading" style="text-align: center; padding: 40px">
      <LoadingOutlined spin style="font-size: 24px" />
    </div>
    <Descriptions v-else-if="shadowReport" :column="1" bordered size="small">
      <Descriptions.Item label="采样总数">{{ shadowReport.total_samples }}</Descriptions.Item>
      <Descriptions.Item label="影子告警率">{{ (shadowReport.shadow_alert_rate * 100).toFixed(1) }}%</Descriptions.Item>
      <Descriptions.Item label="生产告警率">{{ (shadowReport.production_alert_rate * 100).toFixed(1) }}%</Descriptions.Item>
      <Descriptions.Item label="误报差异">
        <span :style="{ color: shadowReport.false_positive_delta > 0 ? '#ff4d4f' : '#52c41a' }">
          {{ shadowReport.false_positive_delta > 0 ? '+' : '' }}{{ shadowReport.false_positive_delta }}
        </span>
      </Descriptions.Item>
      <Descriptions.Item label="平均分数偏差">{{ shadowReport.avg_score_divergence?.toFixed(4) || '-' }}</Descriptions.Item>
      <Descriptions.Item label="平均推理延迟">{{ shadowReport.avg_shadow_latency_ms?.toFixed(1) || '-' }} ms</Descriptions.Item>
    </Descriptions>
    <div v-else style="text-align: center; padding: 40px; color: #666">
      暂无影子推理数据
    </div>
  </Drawer>

  <!-- Stage History Drawer -->
  <Drawer
    v-model:open="stageHistoryVisible"
    title="阶段变更历史"
    :width="560"
  >
    <div v-if="stageHistoryLoading" style="text-align: center; padding: 40px">
      <LoadingOutlined spin style="font-size: 24px" />
    </div>
    <div v-else-if="stageHistory.length > 0">
      <div
        v-for="(event, idx) in stageHistory"
        :key="idx"
        style="padding: 12px; margin-bottom: 8px; background: #1e1e36; border-radius: 6px"
      >
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px">
          <Space>
            <Tag :color="(STAGE_MAP[event.from_stage] || { color: 'default' }).color">
              {{ (STAGE_MAP[event.from_stage] || { text: event.from_stage || '?' }).text }}
            </Tag>
            <span style="color: #8890a0">→</span>
            <Tag :color="(STAGE_MAP[event.to_stage] || { color: 'default' }).color">
              {{ (STAGE_MAP[event.to_stage] || { text: event.to_stage }).text }}
            </Tag>
          </Space>
          <span style="color: #8890a0; font-size: 12px">
            {{ event.timestamp ? event.timestamp.replace('T', ' ').substring(0, 19) : '' }}
          </span>
        </div>
        <div style="font-size: 12px; color: #8890a0">
          <span>操作人: {{ event.triggered_by }}</span>
          <span v-if="event.reason" style="margin-left: 12px">原因: {{ event.reason }}</span>
        </div>
      </div>
    </div>
    <div v-else style="text-align: center; padding: 40px; color: #666">
      暂无阶段变更记录
    </div>
  </Drawer>
</template>
