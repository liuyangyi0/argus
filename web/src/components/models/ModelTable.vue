<script setup lang="ts">
import { ref, computed } from 'vue'
import {
  Card, Table, Button, Tag, Space, Modal, Form, Select, Input,
  Descriptions, Drawer, Steps, Tooltip, Dropdown, Menu, message,
} from 'ant-design-vue'
import {
  ReloadOutlined, CheckOutlined, RollbackOutlined, DeleteOutlined,
  ExperimentOutlined, HistoryOutlined, LoadingOutlined,
  DownOutlined, ExportOutlined, AimOutlined,
} from '@ant-design/icons-vue'
import {
  activateModel, rollbackModel, deleteModel,
  promoteModel, retireModel,
  getStageHistory, getShadowReport,
  reexportModel, recalibrateModel,
} from '../../api'
import { STAGE_MAP, VALID_TRANSITIONS, STAGE_LABELS } from '../../composables/useModelState'

const props = defineProps<{
  models: any[]
  cameras: any[]
}>()

const emit = defineEmits<{
  changed: []
}>()

// ── State ──
const activatingModel = ref<string | null>(null)
const promotingModel = ref<string | null>(null)
const promoteModalVisible = ref(false)
const promoteForm = ref({ version_id: '', target_stage: '', triggered_by: '', reason: '', canary_camera_id: '' })
const shadowReport = ref<any>(null)
const shadowReportVisible = ref(false)
const shadowReportLoading = ref(false)
const stageHistoryVisible = ref(false)
const stageHistory = ref<any[]>([])
const stageHistoryLoading = ref(false)
const reexportModalVisible = ref(false)
const reexportForm = ref({ version_id: '', export_format: 'openvino', quantization: 'fp16' })
const reexportLoading = ref(false)
const recalibrateLoading = ref<string | null>(null)

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

// ── Actions: activate / rollback / delete ──

function handleActivate(record: any) {
  Modal.confirm({
    title: '确认激活',
    content: `确定要激活模型版本 ${record.model_version_id} 吗？这将停用该摄像头的其他模型。`,
    okText: '确认',
    cancelText: '取消',
    async onOk() {
      activatingModel.value = record.model_version_id
      try {
        await activateModel(record.model_version_id)
        message.success(`模型 ${record.model_version_id} 已激活`)
        emit('changed')
      } catch (e: any) {
        message.error(e.response?.data?.error || '激活失败')
      } finally {
        activatingModel.value = null
      }
    },
  })
}

function handleRollback(record: any) {
  Modal.confirm({
    title: '确认回滚',
    content: `确定要回滚摄像头 ${record.camera_id} 到上一个模型版本吗？`,
    okText: '确认回滚',
    cancelText: '取消',
    okType: 'danger',
    async onOk() {
      try {
        const res = await rollbackModel(record.model_version_id)
        message.success(`已回滚到 ${res.data.activated}`)
        emit('changed')
      } catch (e: any) {
        message.error(e.response?.data?.error || '回滚失败')
      }
    },
  })
}

function handleDeleteModel(record: any) {
  Modal.confirm({
    title: '删除模型',
    content: `确定删除模型 ${record.model_version_id}？模型文件将从磁盘永久删除。`,
    okText: '确认删除',
    okType: 'danger',
    cancelText: '取消',
    async onOk() {
      try {
        await deleteModel(record.model_version_id)
        message.success('模型已删除')
        emit('changed')
      } catch (e: any) {
        message.error(e.response?.data?.error || '删除失败')
      }
    },
  })
}

// ── Actions: promote / retire ──

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

// ── Drawers: shadow report / stage history ──

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

// ── Actions: reexport / recalibrate ──

function handleReexport(record: any) {
  reexportForm.value = {
    version_id: record.model_version_id,
    export_format: 'openvino',
    quantization: 'fp16',
  }
  reexportModalVisible.value = true
}

async function submitReexport() {
  reexportLoading.value = true
  try {
    await reexportModel(reexportForm.value.version_id, {
      export_format: reexportForm.value.export_format,
      quantization: reexportForm.value.quantization,
    })
    message.success('重新导出成功')
    reexportModalVisible.value = false
    emit('changed')
  } catch (e: any) {
    message.error(e.response?.data?.error || '重新导出失败')
  } finally {
    reexportLoading.value = false
  }
}

function handleRecalibrate(record: any) {
  Modal.confirm({
    title: '确认重新校准',
    content: `确定要重新校准模型 ${record.model_version_id} 的评分标准化吗？将使用基线图片重新计算校准参数。`,
    okText: '确认',
    cancelText: '取消',
    async onOk() {
      recalibrateLoading.value = record.model_version_id
      try {
        const res = await recalibrateModel(record.model_version_id)
        const n = res.data.n_samples
        message.success(`重新校准成功 (${n} 个样本)`)
        emit('changed')
      } catch (e: any) {
        message.error(e.response?.data?.error || '重新校准失败')
      } finally {
        recalibrateLoading.value = null
      }
    },
  })
}

// ── Dropdown menu dispatcher ──

function handleMenuClick(record: any, { key }: { key: string }) {
  switch (key) {
    case 'reexport': handleReexport(record); break
    case 'recalibrate': handleRecalibrate(record); break
    case 'retire': handleRetire(record); break
    case 'delete': handleDeleteModel(record); break
    case 'shadow-report': handleViewShadowReport(record); break
    case 'stage-history': handleViewStageHistory(record); break
  }
}

// ── Table columns ──
const columns = [
  { title: '版本 ID', dataIndex: 'model_version_id', key: 'version_id', ellipsis: true },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id', width: 90 },
  { title: '类型', dataIndex: 'model_type', key: 'model_type', width: 100 },
  { title: '阶段', key: 'stage', width: 90 },
  { title: '状态', key: 'is_active', width: 80 },
  { title: '创建时间', dataIndex: 'created_at', key: 'created_at', width: 150 },
  { title: '操作', key: 'action', width: 220 },
]
</script>

<template>
  <!-- Pipeline Steps visualization -->
  <Card size="small" style="margin-bottom: 16px">
    <Steps :current="-1" size="small" style="padding: 4px 0">
      <Steps.Step title="候选" :description="`${stageCounts.candidate} 个模型`" status="process" />
      <Steps.Step title="影子" :description="`${stageCounts.shadow} 个模型`" :status="stageCounts.shadow > 0 ? 'process' : 'wait'" />
      <Steps.Step title="金丝雀" :description="`${stageCounts.canary} 个模型`" :status="stageCounts.canary > 0 ? 'process' : 'wait'" />
      <Steps.Step title="生产" :description="`${stageCounts.production} 个模型`" :status="stageCounts.production > 0 ? 'finish' : 'wait'" />
    </Steps>
  </Card>

  <!-- Unified model table -->
  <Card style="margin-bottom: 16px">
    <template #title>
      <div style="display: flex; justify-content: space-between; align-items: center">
        <span>模型版本</span>
        <Button size="small" @click="$emit('changed')">
          <template #icon><ReloadOutlined /></template>
          刷新
        </Button>
      </div>
    </template>
    <Table
      :columns="columns"
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
        <template v-if="column.key === 'is_active'">
          <Tag v-if="record.is_active" color="green">已激活</Tag>
          <Tag v-else color="default">未激活</Tag>
        </template>
        <template v-if="column.key === 'created_at'">
          {{ record.created_at ? record.created_at.replace('T', ' ').substring(0, 16) : '-' }}
        </template>
        <template v-if="column.key === 'action'">
          <Space :size="4">
            <!-- Activate -->
            <Tooltip title="激活此版本">
              <Button
                v-if="!record.is_active"
                size="small"
                type="primary"
                :loading="activatingModel === record.model_version_id"
                @click="handleActivate(record)"
              >
                <template #icon><CheckOutlined /></template>
              </Button>
            </Tooltip>
            <!-- Promote -->
            <Tooltip title="推进阶段">
              <Button
                v-if="record.stage !== 'production' && record.stage !== 'retired'"
                size="small"
                type="primary"
                ghost
                :loading="promotingModel === record.model_version_id"
                @click="handlePromote(record)"
              >
                推进
              </Button>
            </Tooltip>
            <!-- Rollback -->
            <Tooltip title="回滚到上一版本">
              <Button
                v-if="record.is_active"
                size="small"
                danger
                @click="handleRollback(record)"
              >
                <template #icon><RollbackOutlined /></template>
              </Button>
            </Tooltip>
            <!-- More dropdown -->
            <Dropdown>
              <Button size="small">更多 <DownOutlined /></Button>
              <template #overlay>
                <Menu @click="handleMenuClick(record, $event)">
                  <Menu.Item key="reexport">
                    <ExportOutlined /> 重新导出
                  </Menu.Item>
                  <Menu.Item key="recalibrate">
                    <AimOutlined /> 重新校准
                  </Menu.Item>
                  <Menu.Divider />
                  <Menu.Item v-if="record.stage === 'shadow'" key="shadow-report">
                    <ExperimentOutlined /> 影子报告
                  </Menu.Item>
                  <Menu.Item key="stage-history">
                    <HistoryOutlined /> 阶段历史
                  </Menu.Item>
                  <Menu.Divider />
                  <Menu.Item v-if="record.stage !== 'retired'" key="retire" danger>
                    退役
                  </Menu.Item>
                  <Menu.Item v-if="!record.is_active" key="delete" danger>
                    <DeleteOutlined /> 删除
                  </Menu.Item>
                </Menu>
              </template>
            </Dropdown>
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

  <!-- Reexport Modal -->
  <Modal
    v-model:open="reexportModalVisible"
    title="重新导出模型"
    :confirmLoading="reexportLoading"
    @ok="submitReexport"
    okText="确认导出"
    cancelText="取消"
  >
    <Form layout="vertical" style="margin-top: 16px">
      <Form.Item label="导出格式">
        <Select v-model:value="reexportForm.export_format" style="width: 100%">
          <Select.Option value="openvino">OpenVINO</Select.Option>
          <Select.Option value="onnx">ONNX</Select.Option>
          <Select.Option value="torch">Torch</Select.Option>
        </Select>
      </Form.Item>
      <Form.Item label="量化">
        <Select v-model:value="reexportForm.quantization" style="width: 100%">
          <Select.Option value="fp32">FP32 (无量化)</Select.Option>
          <Select.Option value="fp16">FP16 (半精度)</Select.Option>
          <Select.Option value="int8">INT8 (仅 OpenVINO)</Select.Option>
        </Select>
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
            <span style="color: #8890a0">&rarr;</span>
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
