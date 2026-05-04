<script setup lang="ts">
import { ref, computed, reactive } from 'vue'
import {
  Card, Table, Button, Tag, Space, Modal, Form, Select, Input,
  Descriptions, Drawer, Steps, Tooltip, Dropdown, Menu, message,
} from 'ant-design-vue'
import {
  ReloadOutlined, CheckOutlined, RollbackOutlined, DeleteOutlined,
  ExperimentOutlined, HistoryOutlined, LoadingOutlined,
  DownOutlined, ExportOutlined, AimOutlined, SwapOutlined,
} from '@ant-design/icons-vue'
import { useRouter } from 'vue-router'
import {
  activateModel, rollbackModel, deleteModel,
  promoteModel, retireModel,
  getStageHistory, getShadowReport,
  reexportModel, recalibrateModel,
} from '../../api'
import { STAGE_MAP, VALID_TRANSITIONS, STAGE_LABELS } from '../../composables/useModelState'
import { useWebSocket } from '../../composables/useWebSocket'
import { extractErrorMessage } from '../../utils/error'
import type { ModelInfo, ModelVersionEvent, CameraSummary } from '../../types/api'

// ── Release pipeline stage stepper config ──
// 后端 stage 字段为小写 enum 字符串（candidate/shadow/canary/production/retired）。
// STAGES 定义 4 段主线流程,retired 是终态分支,在行级单独渲染为"已退役"覆盖层。
const STAGES: { key: string; label: string }[] = [
  { key: 'candidate', label: '候选' },
  { key: 'shadow', label: '影子' },
  { key: 'canary', label: '灰度' },
  { key: 'production', label: '生产' },
]
const STAGE_INDEX: Record<string, number> = STAGES.reduce(
  (acc, s, i) => ((acc[s.key] = i), acc),
  {} as Record<string, number>,
)
function stepClass(stageKey: string, currentStage: string): 'done' | 'active' | 'pending' {
  const cur = STAGE_INDEX[currentStage]
  const idx = STAGE_INDEX[stageKey]
  if (cur === undefined || idx === undefined) return 'pending'
  if (idx < cur) return 'done'
  if (idx === cur) return 'active'
  return 'pending'
}
function shortVid(vid: string): string {
  if (!vid) return '?'
  return vid.length > 12 ? vid.slice(0, 12) : vid
}

const props = defineProps<{
  models: ModelInfo[]
  cameras: CameraSummary[]
}>()

const emit = defineEmits<{
  changed: []
}>()

const router = useRouter()

// ── State ──
const activatingModel = ref<string | null>(null)
const promotingModel = ref<string | null>(null)
const promoteModalVisible = ref(false)
const promoteForm = ref({ version_id: '', target_stage: '', triggered_by: '', reason: '', canary_camera_id: '' })
const shadowReport = ref<any>(null)
const shadowReportVisible = ref(false)
const shadowReportLoading = ref(false)
const stageHistoryVisible = ref(false)
const stageHistory = ref<ModelVersionEvent[]>([])
const stageHistoryLoading = ref(false)
const reexportModalVisible = ref(false)
const reexportForm = ref({ version_id: '', export_format: 'openvino', quantization: 'fp16' })
const reexportLoading = ref(false)
const recalibrateLoading = ref<string | null>(null)

// ── Release pipeline live progress (model_release WS topic) ──
// 后端在 release_pipeline.transition() commit 成功后会广播一条
// stage_transition 事件,这里转成 toast + 行内 stepper 流光动画,
// 让操作员看到金丝雀/生产推进的真实进度,不必手动 refresh。
// 用 useWebSocket 自带的 onUnmounted 清理,这里不需要再手动 unsubscribe。
// transitioningEvents 同时支持多行同时收到事件(多版本并行推进)。
type TransitionInfo = { from: string; to: string; ts: number }
const transitioningEvents = reactive<Record<string, TransitionInfo>>({})
const FLOW_DURATION_MS = 1200
useWebSocket({
  topics: ['model_release'],
  onMessage: (_topic, data) => {
    const payload = data as {
      type?: string
      model_version_id?: string
      from_stage?: string
      to_stage?: string
      triggered_by?: string
    }
    if (payload?.type !== 'stage_transition') return
    const vid = payload.model_version_id ?? ''
    const from = (payload.from_stage ?? '').toLowerCase()
    const to = (payload.to_stage ?? '').toLowerCase()
    const stageLabel = (s: string) => STAGE_MAP[s]?.text || s || '?'
    message.info(`模型 ${shortVid(vid)} 进入 ${stageLabel(to)} 阶段`)
    // 记录此行的过渡事件,触发 stepper 流光 1.2s,然后清除并刷新表格
    transitioningEvents[vid] = { from, to, ts: Date.now() }
    setTimeout(() => {
      const cur = transitioningEvents[vid]
      // 只有当事件还是同一次过渡时才清除(避免短时间内多次过渡互相覆盖)
      if (cur && cur.ts === transitioningEvents[vid]?.ts) {
        delete transitioningEvents[vid]
      }
      emit('changed')
    }, FLOW_DURATION_MS)
  },
})

// 计算行的过渡事件:用于 stepper 渲染流光段。
function transitionFor(vid: string): TransitionInfo | null {
  return transitioningEvents[vid] || null
}

// 渲染 stepper 段间连接器的 class:
// - flowing: 当前正在过渡且这段 segment (i 到 i+1) 是 from→to 的范围
// - done: 两端都已通过(idx <= currentStage - 1)
// - pending: 否则
function connectorClass(i: number, record: Record<string, any>): string {
  const cur = STAGE_INDEX[record.stage as string]
  const t = transitionFor(record.model_version_id as string)
  if (t) {
    const fIdx = STAGE_INDEX[t.from]
    const tIdx = STAGE_INDEX[t.to]
    if (fIdx !== undefined && tIdx !== undefined) {
      const lo = Math.min(fIdx, tIdx)
      const hi = Math.max(fIdx, tIdx)
      if (i >= lo && i < hi) {
        return tIdx > fIdx ? 'connector flowing forward' : 'connector flowing backward'
      }
    }
  }
  if (cur !== undefined && i < cur) return 'connector done'
  return 'connector pending'
}

// 行级 class:retired 灰化整行
// 入参类型放宽到 Record<string, any>:ant-design-vue 的 row-class-name 回调
// 在内部按通用对象传 record,严格 TS 模式下不能直接绑定 ModelInfo 签名。
function rowClassName(record: Record<string, any>): string {
  return record.stage === 'retired' ? 'row-retired' : ''
}

// ── T3: Expandable row — stage history ──
// 展开时按需拉取该 model_version_id 的 stage_history(已有 API 端点),
// 不引入新依赖、不修改 api/models.ts。失败/空态在展开内提示。
const expandedRowKeys = ref<string[]>([])
const stageHistoryByVersion = reactive<Record<string, ModelVersionEvent[]>>({})
const stageHistoryLoadingByVersion = reactive<Record<string, boolean>>({})
const stageHistoryErrorByVersion = reactive<Record<string, string>>({})

async function ensureStageHistory(vid: string) {
  if (stageHistoryByVersion[vid] || stageHistoryLoadingByVersion[vid]) return
  stageHistoryLoadingByVersion[vid] = true
  delete stageHistoryErrorByVersion[vid]
  try {
    const res = await getStageHistory(vid)
    stageHistoryByVersion[vid] = res.events || []
  } catch (e) {
    stageHistoryErrorByVersion[vid] = extractErrorMessage(e, '获取阶段历史失败')
  } finally {
    stageHistoryLoadingByVersion[vid] = false
  }
}

function onExpandedRowsChange(keys: (string | number)[]) {
  expandedRowKeys.value = keys.map(k => String(k))
  for (const vid of expandedRowKeys.value) {
    ensureStageHistory(vid)
  }
}

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
  return [...props.models].sort((a: ModelInfo, b: ModelInfo) =>
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
      } catch (e) {
        message.error(extractErrorMessage(e, '激活失败'))
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
        message.success(`已回滚到 ${res.activated}`)
        emit('changed')
      } catch (e) {
        message.error(extractErrorMessage(e, '回滚失败'))
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
      } catch (e) {
        message.error(extractErrorMessage(e, '删除失败'))
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
  } catch (e) {
    message.error(extractErrorMessage(e, '推进失败'))
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
      } catch (e) {
        message.error(extractErrorMessage(e, '退役失败'))
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
    shadowReport.value = res
  } catch (e) {
    message.error(extractErrorMessage(e, '获取影子报告失败'))
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
    stageHistory.value = res.events || []
  } catch (e) {
    message.error(extractErrorMessage(e, '获取历史失败'))
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
  } catch (e) {
    message.error(extractErrorMessage(e, '重新导出失败'))
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
        const n = res.n_samples
        message.success(`重新校准成功 (${n} 个样本)`)
        emit('changed')
      } catch (e) {
        message.error(extractErrorMessage(e, '重新校准失败'))
      } finally {
        recalibrateLoading.value = null
      }
    },
  })
}

// ── Dropdown menu dispatcher ──

function handleOpenABCompare(record: any) {
  router.push({
    path: '/models',
    query: { tab: 'comparison', camera: record.camera_id, shadow: record.model_version_id },
  })
}

function handleMenuClick(record: any, { key }: { key: string | number }) {
  switch (key) {
    case 'reexport': handleReexport(record); break
    case 'recalibrate': handleRecalibrate(record); break
    case 'retire': handleRetire(record); break
    case 'delete': handleDeleteModel(record); break
    case 'shadow-report': handleViewShadowReport(record); break
    case 'ab-compare': handleOpenABCompare(record); break
    case 'stage-history': handleViewStageHistory(record); break
  }
}

// ── Table columns ──
// '阶段' 列加宽到 240px 以容纳紧凑 4 段 stepper(候选/影子/灰度/生产)。
const columns = [
  { title: '版本 ID', dataIndex: 'model_version_id', key: 'version_id', ellipsis: true },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id', width: 90 },
  { title: '类型', dataIndex: 'model_type', key: 'model_type', width: 100 },
  { title: '阶段', key: 'stage', width: 240 },
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
      :row-class-name="rowClassName"
      :expanded-row-keys="expandedRowKeys"
      @expanded-rows-change="onExpandedRowsChange"
    >
      <template #expandedRowRender="{ record }">
        <div class="stage-history-panel">
          <div v-if="stageHistoryLoadingByVersion[record.model_version_id]" class="stage-history-empty">
            <LoadingOutlined spin /> 加载阶段历史...
          </div>
          <div v-else-if="stageHistoryErrorByVersion[record.model_version_id]" class="stage-history-empty error">
            {{ stageHistoryErrorByVersion[record.model_version_id] }}
          </div>
          <div v-else-if="stageHistoryByVersion[record.model_version_id]?.length">
            <div
              v-for="(event, idx) in stageHistoryByVersion[record.model_version_id]"
              :key="idx"
              class="stage-history-item"
            >
              <Space>
                <Tag :color="(STAGE_MAP[event.from_stage ?? ''] || { color: 'default' }).color">
                  {{ (STAGE_MAP[event.from_stage ?? ''] || { text: event.from_stage || '?' }).text }}
                </Tag>
                <span class="arrow">&rarr;</span>
                <Tag :color="(STAGE_MAP[event.to_stage] || { color: 'default' }).color">
                  {{ (STAGE_MAP[event.to_stage] || { text: event.to_stage }).text }}
                </Tag>
                <span class="meta">{{ event.triggered_by }}</span>
                <span v-if="event.reason" class="meta">原因: {{ event.reason }}</span>
              </Space>
              <span class="ts">
                {{ event.timestamp ? event.timestamp.replace('T', ' ').substring(0, 19) : '' }}
              </span>
            </div>
          </div>
          <div v-else class="stage-history-empty">暂无阶段变更记录</div>
        </div>
      </template>
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'version_id'">
          <span style="font-family: monospace; font-size: 12px">{{ record.model_version_id }}</span>
        </template>
        <template v-if="column.key === 'stage'">
          <div
            class="stage-stepper"
            :class="{ 'is-retired': record.stage === 'retired' }"
            :data-vid="record.model_version_id"
          >
            <template v-if="record.stage === 'retired'">
              <Tag color="default" class="retired-tag">已退役</Tag>
            </template>
            <template v-else>
              <template v-for="(s, i) in STAGES" :key="s.key">
                <div class="stage-step" :class="stepClass(s.key, record.stage)">
                  <div class="stage-dot">
                    <span
                      v-if="
                        transitionFor(record.model_version_id) &&
                        transitionFor(record.model_version_id)?.to === s.key
                      "
                      class="stage-dot-pulse"
                    ></span>
                  </div>
                  <div class="stage-label">{{ s.label }}</div>
                </div>
                <div
                  v-if="i < STAGES.length - 1"
                  :key="`c-${s.key}`"
                  :class="connectorClass(i, record)"
                ></div>
              </template>
            </template>
          </div>
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
                  <Menu.Item v-if="record.stage === 'shadow' || record.stage === 'canary'" key="shadow-report">
                    <ExperimentOutlined /> 影子报告
                  </Menu.Item>
                  <Menu.Item v-if="record.stage === 'shadow' || record.stage === 'canary'" key="ab-compare">
                    <SwapOutlined /> A/B 详细对比
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
            v-for="stage in (VALID_TRANSITIONS[models.find(m => m.model_version_id === promoteForm.version_id)?.stage ?? ''] || [])"
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
        <span :style="{ color: shadowReport.false_positive_delta > 0 ? '#e5484d' : '#15a34a' }">
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
        style="padding: 12px; margin-bottom: 8px; background: rgba(10, 10, 15, 0.05); border-radius: 6px"
      >
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px">
          <Space>
            <Tag :color="(STAGE_MAP[event.from_stage ?? ''] || { color: 'default' }).color">
              {{ (STAGE_MAP[event.from_stage ?? ''] || { text: event.from_stage || '?' }).text }}
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

<style scoped>
/* ── Stage stepper (4 段紧凑流程图) ──
   每行的"阶段"列内联显示 候选→影子→灰度→生产 的可视化进度。
   宽度约 220px,单行高度 36px,与表格 size="small" 协调。 */
.stage-stepper {
  display: flex;
  align-items: center;
  gap: 0;
  width: 100%;
  min-width: 200px;
  max-width: 240px;
  height: 36px;
}
.stage-stepper.is-retired {
  justify-content: flex-start;
}
.retired-tag {
  margin: 0;
}
.stage-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  width: 36px;
}
.stage-dot {
  position: relative;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: 2px solid #c4c8d4;
  background: transparent;
  box-sizing: border-box;
  transition: background 0.2s, border-color 0.2s, box-shadow 0.2s;
}
.stage-step.done .stage-dot {
  background: #1677ff;
  border-color: #1677ff;
}
.stage-step.active .stage-dot {
  background: #1677ff;
  border-color: #1677ff;
  box-shadow: 0 0 0 3px rgba(22, 119, 255, 0.18);
}
.stage-step.pending .stage-dot {
  background: transparent;
  border-color: #c4c8d4;
}
.stage-label {
  font-size: 11px;
  line-height: 1.1;
  margin-top: 4px;
  white-space: nowrap;
  color: #8c8c8c;
}
.stage-step.done .stage-label,
.stage-step.active .stage-label {
  color: #1f1f1f;
  font-weight: 500;
}
.stage-step.active .stage-label {
  color: #1677ff;
}

/* connector: stepper 段间的连线;高 2px,绝对定位在 dot 中心高度 */
.connector {
  flex: 1 1 auto;
  height: 2px;
  background: #d9d9d9;
  align-self: flex-start;
  margin-top: 7px; /* 12px dot / 2 + 1px(border) ≈ dot 中心 */
  position: relative;
  overflow: hidden;
  min-width: 8px;
}
.connector.done {
  background: #1677ff;
}
.connector.pending {
  background: #d9d9d9;
}

/* T2: 流光动画 — 收到 stage_transition WS 事件时,from→to 段间出现 1.2s 流光 */
.connector.flowing {
  background: linear-gradient(90deg, #1677ff 0%, #1677ff 100%);
}
.connector.flowing::after {
  content: '';
  position: absolute;
  top: 0;
  left: -50%;
  width: 50%;
  height: 100%;
  background: linear-gradient(
    90deg,
    rgba(255, 255, 255, 0) 0%,
    rgba(255, 255, 255, 0.85) 50%,
    rgba(255, 255, 255, 0) 100%
  );
  animation: stage-flow 1.2s ease-in-out forwards;
}
.connector.flowing.backward::after {
  /* 回退过渡:流光从右往左 */
  animation-name: stage-flow-back;
}
@keyframes stage-flow {
  from {
    transform: translateX(0);
  }
  to {
    transform: translateX(300%);
  }
}
@keyframes stage-flow-back {
  from {
    transform: translateX(300%);
  }
  to {
    transform: translateX(0);
  }
}

/* 目标 dot 的 pulse — 强调 to_stage,持续 1.2s 后由父级清除 transitioningEvents 移除 */
.stage-dot-pulse {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: rgba(22, 119, 255, 0.45);
  animation: stage-dot-ping 1.2s ease-out forwards;
  pointer-events: none;
}
@keyframes stage-dot-ping {
  0% {
    transform: translate(-50%, -50%) scale(1);
    opacity: 0.7;
  }
  100% {
    transform: translate(-50%, -50%) scale(2.6);
    opacity: 0;
  }
}

/* Retired 行整行变灰 */
:deep(.row-retired) {
  opacity: 0.55;
}
:deep(.row-retired) td {
  background: rgba(0, 0, 0, 0.02);
}

/* T3: 展开行内的阶段历史面板 */
.stage-history-panel {
  padding: 8px 12px;
  background: rgba(10, 10, 15, 0.03);
  border-radius: 4px;
}
.stage-history-empty {
  text-align: center;
  padding: 16px;
  color: #8890a0;
  font-size: 12px;
}
.stage-history-empty.error {
  color: #e5484d;
}
.stage-history-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 4px;
  border-bottom: 1px dashed rgba(0, 0, 0, 0.06);
  font-size: 12px;
}
.stage-history-item:last-child {
  border-bottom: none;
}
.stage-history-item .arrow {
  color: #8890a0;
}
.stage-history-item .meta {
  color: #6b7280;
  font-size: 11px;
}
.stage-history-item .ts {
  color: #8890a0;
  font-size: 11px;
  font-family: monospace;
}
</style>
