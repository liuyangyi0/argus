<script setup lang="ts">
import { ref, computed, reactive, onUnmounted, watch } from 'vue'
import {
  Card, Button, Tag, Space, Modal, Form, Select, Input,
  Descriptions, Drawer, Steps, Tooltip, Dropdown, Menu, message,
} from 'ant-design-vue'
import {
  ReloadOutlined, RollbackOutlined, DeleteOutlined,
  ExperimentOutlined, HistoryOutlined, LoadingOutlined,
  DownOutlined, ExportOutlined, AimOutlined, SwapOutlined,
} from '@ant-design/icons-vue'
import { useRouter } from 'vue-router'
import {
  deleteModel,
  promoteModel, retireModel, rollbackModel, getRollbackReadiness,
  getStageHistory, getShadowReport, getReleaseOperation,
  acknowledgeReleaseOperation,
  reexportModel, recalibrateModel,
} from '../../api'
import { STAGE_MAP, VALID_TRANSITIONS, STAGE_LABELS } from '../../composables/useModelState'
import { useWebSocket } from '../../composables/useWebSocket'
import { extractErrorMessage } from '../../utils/error'
import { ApiError } from '../../types/api'
import type {
  CameraSummary,
  ModelInfo,
  ModelVersionEvent,
  ModelReleaseMutationResponse,
  ReleaseOperation,
  ReleaseReadiness,
  ShadowReportResponse,
} from '../../types/api'
import type { ModelReleaseRequest, RollbackReadiness } from '../../api/models'
import ResponsiveTable from '../common/ResponsiveTable.vue'

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
  releaseReadiness?: Record<string, ReleaseReadiness>
  focusedVersionId?: string
}>()

const emit = defineEmits<{
  changed: []
}>()

const router = useRouter()

// ── State ──
const promotingModel = ref<string | null>(null)
const promoteModalVisible = ref(false)
const promoteForm = ref({ version_id: '', target_stage: '', triggered_by: '', reason: '', canary_camera_id: '' })
const shadowReport = ref<ShadowReportResponse | null>(null)
const shadowReportVisible = ref(false)
const shadowReportLoading = ref(false)
const stageHistoryVisible = ref(false)
const stageHistory = ref<ModelVersionEvent[]>([])
const stageHistoryLoading = ref(false)
const reexportModalVisible = ref(false)
const reexportForm = ref({ version_id: '', export_format: 'openvino', quantization: 'fp16' })
const reexportLoading = ref(false)
const recalibrateLoading = ref<string | null>(null)
const rollbackVisible = ref(false)
const rollbackTarget = ref('')
const rollbackReason = ref('')
const rollbackChecking = ref(false)
const rollbackSubmitting = ref(false)
const rollbackReadiness = ref<RollbackReadiness | null>(null)
const rollbackError = ref('')
let rollbackGeneration = 0
const rollbackRevisionMatches = computed(() => {
  const ready = rollbackReadiness.value
  const target = props.models.find(model => model.model_version_id === rollbackTarget.value)
  const source = props.models.find(model => model.model_version_id === ready?.source_model_version_id)
  return !!ready && ready.model_version_id === rollbackTarget.value && target?.stage === 'retired'
    && target.release_revision === ready.expected_revision
    && source?.stage === 'production' && source.is_active
    && source.release_revision === ready.source_release_revision
})
const canSubmitRollback = computed(() => !!rollbackReadiness.value?.rollback_allowed
  && rollbackRevisionMatches.value && !rollbackChecking.value && !rollbackSubmitting.value
  && !!rollbackReason.value.trim())

async function checkRollback() {
  const generation = ++rollbackGeneration
  const versionId = rollbackTarget.value
  rollbackChecking.value = true
  rollbackReadiness.value = null
  rollbackError.value = ''
  try {
    const readiness = await getRollbackReadiness(versionId)
    if (generation !== rollbackGeneration || !rollbackVisible.value) return
    rollbackReadiness.value = readiness
  } catch (error) {
    if (generation === rollbackGeneration && rollbackVisible.value) {
      rollbackError.value = extractErrorMessage(error, '回滚检查失败')
    }
  } finally {
    if (generation === rollbackGeneration) rollbackChecking.value = false
  }
}

function openRollback(versionId: string) {
  rollbackSubmitting.value = false
  rollbackTarget.value = versionId
  rollbackReason.value = ''
  rollbackVisible.value = true
  void checkRollback()
}

function closeRollback() {
  ++rollbackGeneration
  rollbackVisible.value = false
  rollbackReadiness.value = null
  rollbackChecking.value = false
}

async function submitRollback() {
  const ready = rollbackReadiness.value
  if (!canSubmitRollback.value || !ready?.source_model_version_id || ready.source_release_revision == null) return
  rollbackSubmitting.value = true
  const generation = rollbackGeneration
  const versionId = rollbackTarget.value
  const isCurrent = () => !releaseTrackingDisposed && generation === rollbackGeneration
    && rollbackVisible.value && rollbackTarget.value === versionId
  try {
    const result = await rollbackModel(versionId, {
      source_model_version_id: ready.source_model_version_id,
      source_release_revision: ready.source_release_revision,
      expected_revision: ready.expected_revision,
      reason: rollbackReason.value.trim(),
    })
    if (isCurrent()) {
      notifyReleaseResult(result, '模型回滚')
      rollbackSubmitting.value = false
      closeRollback()
    } else if (result.operation) {
      scheduleReleaseOperationRefresh(result.operation)
    }
    if (!releaseTrackingDisposed) emit('changed')
  } catch (error) {
    const operation = releaseOperationFromError(error)
    if (operation) scheduleReleaseOperationRefresh(operation)
    if (isCurrent()) rollbackError.value = extractErrorMessage(error, '回滚提交失败')
  } finally {
    if (isCurrent()) rollbackSubmitting.value = false
  }
}
const releaseOperations = reactive<Record<string, ReleaseOperation>>({})
const releaseOperationTimers = new Map<string, ReturnType<typeof setTimeout>>()
let releaseTrackingDisposed = false

function operationRevision(operation: ReleaseOperation): number {
  return operation.applied_revision ?? operation.expected_revision
}

function isCurrentOperation(operation: ReleaseOperation, record?: { release_revision?: number | null }): boolean {
  const revision = record?.release_revision
  return typeof revision !== 'number' || operationRevision(operation) >= revision
}

function scheduleReleaseOperationRefresh(operation: ReleaseOperation): boolean {
  if (releaseTrackingDisposed) return false
  const existing = releaseOperationTimers.get(operation.operation_id)
  if (existing) clearTimeout(existing)
  releaseOperationTimers.delete(operation.operation_id)
  const model = props.models.find(item => item.model_version_id === operation.model_version_id)
  const previous = releaseOperations[operation.model_version_id]
  if (!isCurrentOperation(operation, model)) return false
  if (previous) {
    if (operationRevision(operation) < operationRevision(previous)) return false
    if (operation.operation_id === previous.operation_id) {
      if (previous.status !== 'pending' && operation.status === 'pending') return false
    } else if (operationRevision(operation) === operationRevision(previous)) {
      const previousTime = Date.parse(previous.requested_at || '')
      const incomingTime = Date.parse(operation.requested_at || '')
      if (Number.isFinite(previousTime) && Number.isFinite(incomingTime) && incomingTime < previousTime) return false
      if (!Number.isFinite(incomingTime) && previous.status === 'pending' && operation.status !== 'pending') return false
    }
    if (previous.operation_id !== operation.operation_id) {
      const previousTimer = releaseOperationTimers.get(previous.operation_id)
      if (previousTimer) clearTimeout(previousTimer)
      releaseOperationTimers.delete(previous.operation_id)
    }
  }
  releaseOperations[operation.model_version_id] = operation
  if (operation.status !== 'pending') {
    acknowledgeReleaseOperation(operation)
    releaseOperationTimers.delete(operation.operation_id)
    return true
  }
  const timer = setTimeout(async () => {
    releaseOperationTimers.delete(operation.operation_id)
    try {
      const current = await getReleaseOperation(operation.operation_id)
      scheduleReleaseOperationRefresh(current)
      if (current.status !== 'pending') emit('changed')
    } catch {
      // Keep the accepted operation visible. The WS channel and the next
      // explicit registry refresh remain independent recovery paths.
      scheduleReleaseOperationRefresh(operation)
    }
  }, 2_000)
  releaseOperationTimers.set(operation.operation_id, timer)
  return true
}

onUnmounted(() => {
  ++rollbackGeneration
  releaseTrackingDisposed = true
  for (const timer of releaseOperationTimers.values()) clearTimeout(timer)
  releaseOperationTimers.clear()
})

// The durable registry read carries the latest operation, including commands
// submitted in another browser or before this page was mounted. WS is a change
// notification, not the only place an accepted operation can be discovered.
watch(
  () => props.models.map(model => model.latest_release_operation),
  operations => {
    for (const operation of operations) {
      if (operation) scheduleReleaseOperationRefresh(operation)
    }
  },
  { immediate: true },
)

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
      operation?: ReleaseOperation
    }
    if (payload?.type?.startsWith('release_operation_') && payload.operation) {
      if (!scheduleReleaseOperationRefresh(payload.operation)) return
      if (payload.operation.status === 'applied') {
        message.success(`发布操作 ${shortVid(payload.operation.operation_id)} 已应用`)
      } else if (payload.operation.status === 'failed') {
        message.error(`发布失败：${payload.operation.error || '运行时未应用'}`)
      }
      emit('changed')
      return
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
  const classes: string[] = []
  if (record.stage === 'retired') classes.push('row-retired')
  if (
    props.focusedVersionId &&
    record.model_version_id === props.focusedVersionId
  ) {
    classes.push('row-focused')
  }
  return classes.join(' ')
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
  const ordered = [...props.models].sort((a: ModelInfo, b: ModelInfo) =>
    (order[a.stage] ?? 5) - (order[b.stage] ?? 5)
  )
  const focusId = props.focusedVersionId
  if (!focusId) return ordered

  const focusIdx = ordered.findIndex(model => model.model_version_id === focusId)
  if (focusIdx <= 0) return ordered

  const focused = ordered[focusIdx]
  return [
    focused,
    ...ordered.slice(0, focusIdx),
    ...ordered.slice(focusIdx + 1),
  ]
})

const RUNTIME_STATE_MAP: Record<string, { text: string; color: string }> = {
  applied: { text: '已应用到运行时', color: 'success' },
  waiting: { text: '等待摄像头启动', color: 'processing' },
  failed: { text: '应用失败', color: 'error' },
}

function runtimeStateDisplay(record: Record<string, any>): { text: string; color: string } {
  const cached = releaseOperations[String(record.model_version_id)]
  const operation = cached && isCurrentOperation(cached, record) ? cached : undefined
  const state = typeof record.runtime_state === 'string' ? record.runtime_state : ''
  // Registry runtime health remains authoritative after a successful release.
  // A historical operation cannot paint a stopped/failed runtime green.
  if (state === 'waiting' || state === 'failed') return RUNTIME_STATE_MAP[state]
  if (operation?.status === 'pending') return { text: '发布处理中', color: 'processing' }
  if (state) return RUNTIME_STATE_MAP[state] || { text: state, color: 'default' }
  if (operation?.status === 'failed') return { text: '发布失败', color: 'error' }
  if (operation?.status === 'applied') return { text: '发布已应用', color: 'success' }
  return { text: '-', color: 'default' }
}

function releaseOperationMeta(record: Record<string, any>): string | null {
  const operation = releaseOperations[String(record.model_version_id)]
  if (!operation || !isCurrentOperation(operation, record)) return null
  const revision = operation.applied_revision ?? operation.expected_revision
  const state = { pending: '发布处理中', applied: '发布已应用', failed: '发布失败' }[operation.status]
  return `${state} · ${shortVid(operation.operation_id)} · r${revision}${operation.error ? ` · ${operation.error}` : ''}`
}

function isProductionTransition(record: Record<string, any>): boolean {
  return (VALID_TRANSITIONS[String(record.stage)] || [])[0] === 'production'
}

function productionReadiness(record: Record<string, any>): ReleaseReadiness | null {
  return props.releaseReadiness?.[String(record.model_version_id)] ?? null
}

function productionReadinessReason(record: Record<string, any>): string {
  const readiness = productionReadiness(record)
  if (!readiness) {
    return '服务端发布 readiness 尚未返回；按 fail-closed 策略保持 NO-GO'
  }
  if (readiness.reasons.length > 0) return readiness.reasons.join('；')
  if (readiness.status === 'pending') return '生产发布操作正在处理，禁止重复提交'
  if (readiness.status === 'failed') return '最近一次生产发布失败，请先处理失败原因'
  return '生产发布必须通过服务端 baseline v3、回归指标与可信 72 小时 soak 证据校验'
}

function productionReadinessLabel(record: Record<string, any>): string {
  const readiness = productionReadiness(record)
  if (readiness?.status === 'pending') return '生产发布处理中'
  if (readiness?.status === 'failed') return '生产发布失败'
  return '生产 NO-GO'
}

function productionReadinessMeta(record: Record<string, any>): string | null {
  const appliedProduction = record.stage === 'production'
  if (!isProductionTransition(record) && !appliedProduction) return null
  const readiness = productionReadiness(record)
  if (!readiness) return 'Readiness 未加载 · NO-GO'
  const failedGates = readiness.gates.filter(gate => gate.blocking && !gate.passed).length
  if (appliedProduction) {
    return readiness.production_ready
      ? 'Readiness · 已应用且安全门通过'
      : `Readiness · 已应用但 NO-GO · ${failedGates} 项未通过`
  }
  if (readiness.status === 'pending') return 'Readiness · 发布处理中'
  if (readiness.status === 'failed') return `Readiness · 失败 · ${failedGates} 项未通过`
  return readiness.production_ready
    ? 'Readiness · 已满足（需受控证据提交）'
    : `Readiness · NO-GO · ${failedGates} 项未通过`
}

const promoteModelRecord = computed(() =>
  props.models.find(m => m.model_version_id === promoteForm.value.version_id) || null
)

function promoteOwnCameraId(): string {
  return promoteModelRecord.value?.camera_id || ''
}

function handlePromoteTargetChange(stage: unknown) {
  promoteForm.value.canary_camera_id = stage === 'canary' ? promoteOwnCameraId() : ''
}

function notifyRuntimeResult(
  result: { runtime_synced?: boolean } | null | undefined,
  successText: string,
) {
  if (result?.runtime_synced === false) {
    message.warning(`${successText}，但运行时尚未同步；请检查运行态`)
    return
  }
  message.success(successText)
}

function notifyReleaseResult(result: ModelReleaseMutationResponse, successText: string): void {
  const operation = result.operation
  if (!operation) {
    notifyRuntimeResult(result, successText)
    return
  }
  if (!scheduleReleaseOperationRefresh(operation)) return
  if (operation.status === 'pending') {
    message.info(`${successText}，发布请求已受理（${shortVid(operation.operation_id)}）`)
    return
  }
  if (operation.status === 'failed') {
    message.error(`${successText}失败：${operation.error || '运行时未应用'}`)
    return
  }
  message.success(`${successText}，运行时已确认（r${operation.applied_revision ?? operation.expected_revision}）`)
}

function releaseOperationFromError(error: unknown): ReleaseOperation | null {
  if (!(error instanceof ApiError) || !error.details || typeof error.details !== 'object') return null
  const operation = (error.details as Record<string, unknown>).operation
  if (!operation || typeof operation !== 'object') return null
  const candidate = operation as Partial<ReleaseOperation>
  if (!candidate.operation_id || !candidate.model_version_id || !candidate.status) return null
  return candidate as ReleaseOperation
}

// ── Actions: delete ──

function handleDeleteModel(record: any) {
  Modal.confirm({
    title: '删除模型',
    content: `确定删除模型 ${record.model_version_id} 的注册记录？模型文件将在确认没有引用后稍后安全清理。`,
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
  const defaultTarget = transitions.find(stage => stage !== 'production') ?? transitions[0]
  promoteForm.value = {
    version_id: record.model_version_id,
    target_stage: defaultTarget,
    triggered_by: '',
    reason: '',
    canary_camera_id: defaultTarget === 'canary' ? record.camera_id : '',
  }
  promoteModalVisible.value = true
}

async function submitPromote() {
  if (promoteForm.value.target_stage === 'canary') {
    const ownCameraId = promoteOwnCameraId()
    if (!ownCameraId) {
      message.warning('金丝雀阶段需要模型所属摄像头')
      return
    }
    promoteForm.value.canary_camera_id = ownCameraId
  }
  promotingModel.value = promoteForm.value.version_id
  try {
    const payload: ModelReleaseRequest = {
      target_stage: promoteForm.value.target_stage,
    }
    if (promoteForm.value.triggered_by) payload.triggered_by = promoteForm.value.triggered_by
    if (promoteForm.value.reason) payload.reason = promoteForm.value.reason
    if (promoteForm.value.canary_camera_id) payload.canary_camera_id = promoteForm.value.canary_camera_id
    const currentRevision = promoteModelRecord.value?.release_revision
    if (typeof currentRevision === 'number') payload.expected_revision = currentRevision

    const res = await promoteModel(promoteForm.value.version_id, payload)
    notifyReleaseResult(
      res,
      `模型已推进到 ${STAGE_MAP[promoteForm.value.target_stage]?.text || promoteForm.value.target_stage}`,
    )
    promoteModalVisible.value = false
    emit('changed')
  } catch (e) {
    const operation = releaseOperationFromError(e)
    if (operation) scheduleReleaseOperationRefresh(operation)
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
        const data = typeof record.release_revision === 'number'
          ? { expected_revision: record.release_revision }
          : undefined
        const res = data
          ? await retireModel(record.model_version_id, data)
          : await retireModel(record.model_version_id)
        notifyReleaseResult(res, '模型已退役')
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

function enabledGateChecks() {
  return shadowReport.value?.gate_report?.checks?.filter(check => check.enabled) || []
}

function formatGateValue(name: string, value: number): string {
  if (!Number.isFinite(value)) return '-'
  if (name.includes('samples')) return String(Math.round(value))
  if (name.includes('latency')) return `${value.toFixed(1)} ms`
  return value.toFixed(3)
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
  const sourcePath = typeof record.model_path === 'string' ? record.model_path.toLowerCase() : ''
  const exportFormat = sourcePath.endsWith('.pt') ? 'torch'
    : sourcePath.endsWith('.onnx') ? 'onnx' : 'openvino'
  reexportForm.value = {
    version_id: record.model_version_id,
    export_format: exportFormat,
    quantization: exportFormat === 'openvino' ? 'fp16' : 'fp32',
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
    message.success('已创建重新导出的候选版本，请在模型列表中审核与发布')
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
    content: `使用基线图片重新校准模型 ${record.model_version_id}，并创建新的候选版本。候选版本需要单独审核与发布。`,
    okText: '确认',
    cancelText: '取消',
    async onOk() {
      recalibrateLoading.value = record.model_version_id
      try {
        const res = await recalibrateModel(record.model_version_id)
        const n = res.n_samples
        message.success(`已创建校准候选版本 (${n} 个样本)，请审核与发布`)
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
    path: '/models/comparison',
    query: { camera: record.camera_id, shadow: record.model_version_id },
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
  { title: '运行态', key: 'runtime_state', width: 170 },
  { title: '激活', key: 'is_active', width: 80 },
  { title: '创建时间', dataIndex: 'created_at', key: 'created_at', width: 150 },
  { title: '操作', key: 'action', width: 250 },
]
</script>

<template>
  <!-- Pipeline Steps visualization -->
  <Card size="small" style="margin-bottom: 16px">
    <Steps :current="-1" size="small" style="padding: 4px 0">
      <Steps.Step title="候选" :description="`${stageCounts.candidate} 个模型`" status="process" />
      <Steps.Step title="影子" :description="`${stageCounts.shadow} 个 · 旁路评分/不告警`" :status="stageCounts.shadow > 0 ? 'process' : 'wait'" />
      <Steps.Step title="金丝雀" :description="`${stageCounts.canary} 个 · 该摄像头主模型`" :status="stageCounts.canary > 0 ? 'process' : 'wait'" />
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
    <ResponsiveTable
      :columns="columns"
      :data-source="sortedModels"
      :pagination="{ pageSize: 15, showSizeChanger: false }"
      row-key="model_version_id"
      size="small"
      :row-class-name="rowClassName"
      :expanded-row-keys="expandedRowKeys"
      :min-width="1120"
      accessible-label="模型发布列表，可横向滚动"
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
        <template v-if="column.key === 'runtime_state'">
          <div class="runtime-state-cell">
            <Tag :color="runtimeStateDisplay(record).color">{{ runtimeStateDisplay(record).text }}</Tag>
            <small v-if="releaseOperationMeta(record)">{{ releaseOperationMeta(record) }}</small>
            <small
              v-if="productionReadinessMeta(record)"
              :title="productionReadinessReason(record)"
            >
              {{ productionReadinessMeta(record) }}
            </small>
          </div>
        </template>
        <template v-if="column.key === 'created_at'">
          {{ record.created_at ? record.created_at.replace('T', ' ').substring(0, 16) : '-' }}
        </template>
        <template v-if="column.key === 'action'">
          <Space :size="4">
            <!-- Promote -->
            <Tooltip :title="isProductionTransition(record)
              ? productionReadinessReason(record)
              : '推进阶段'">
              <Button
                v-if="record.stage !== 'production' && record.stage !== 'retired'"
                size="small"
                type="primary"
                ghost
                :disabled="isProductionTransition(record)"
                :title="isProductionTransition(record) ? productionReadinessReason(record) : '推进阶段'"
                :aria-label="isProductionTransition(record)
                  ? `${productionReadinessLabel(record)}：${productionReadinessReason(record)}`
                  : `推进模型 ${record.model_version_id}`"
                :loading="promotingModel === record.model_version_id"
                @click="handlePromote(record)"
              >
                {{ isProductionTransition(record) ? productionReadinessLabel(record) : '推进' }}
              </Button>
            </Tooltip>
            <Tooltip v-if="record.stage === 'canary'" title="回退到影子阶段">
              <Button size="small" @click="handlePromote(record)">回退</Button>
            </Tooltip>
            <Tooltip title="检查此历史生产版本是否可通过两阶段事务恢复">
              <Button
                v-if="record.stage === 'retired' && !record.is_active"
                size="small"
                danger
                :aria-label="`检查回滚到模型 ${record.model_version_id}`"
                @click="openRollback(record.model_version_id)"
              >
                <template #icon><RollbackOutlined /></template>
                检查回滚
              </Button>
            </Tooltip>
            <!-- More dropdown -->
            <Dropdown>
              <Button size="small">更多 <DownOutlined /></Button>
              <template #overlay>
                <Menu @click="handleMenuClick(record, $event)">
                  <template v-if="record.stage !== 'retired'">
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
                  </template>
                  <Menu.Item key="stage-history">
                    <HistoryOutlined /> 阶段历史
                  </Menu.Item>
                  <Menu.Divider v-if="record.stage !== 'retired' || !record.is_active" />
                  <template v-if="record.stage !== 'retired'">
                    <Menu.Item key="retire" danger>
                      退役
                    </Menu.Item>
                  </template>
                  <Menu.Item v-if="!record.is_active" key="delete" danger>
                    <DeleteOutlined /> 删除
                  </Menu.Item>
                </Menu>
              </template>
            </Dropdown>
          </Space>
        </template>
      </template>
    </ResponsiveTable>
    <div v-if="models.length === 0" style="text-align: center; padding: 32px; color: #666">
      暂无注册的模型版本
    </div>
  </Card>

  <Modal
    v-model:open="rollbackVisible"
    title="受控模型回滚"
    okText="确认回滚"
    cancelText="取消"
    :confirmLoading="rollbackSubmitting"
    :okButtonProps="{ disabled: !canSubmitRollback }"
    :cancelButtonProps="{ disabled: rollbackSubmitting }"
    :closable="!rollbackSubmitting"
    :keyboard="!rollbackSubmitting"
    :maskClosable="false"
    @ok="submitRollback"
    @cancel="closeRollback"
  >
    <p>目标版本：{{ rollbackTarget }}</p>
    <p>只恢复已验证的历史生产模型；请先选择其已批准场景。系统不会自动切换场景。</p>
    <p>模型应用成功后仍需 SceneGuard 重新确认机位，确认前检测与告警保持暂停。</p>
    <p v-if="rollbackChecking" role="status">正在复核历史发布证据与运行版本…</p>
    <p v-if="rollbackError" role="alert">{{ rollbackError }}</p>
    <template v-if="rollbackReadiness">
      <p v-if="rollbackReadiness.rollback_allowed && !rollbackRevisionMatches" role="alert">模型版本已变化，回滚检查已过期，请重新检查。</p>
      <p v-else-if="rollbackReadiness.rollback_allowed">模型回滚检查通过；不代表场景已放行。</p>
      <ul v-else><li v-for="reason in rollbackReadiness.reasons" :key="reason">{{ reason }}</li></ul>
      <p v-if="rollbackReadiness.source_model_version_id">将替换：{{ rollbackReadiness.source_model_version_id }}</p>
    </template>
    <Button :loading="rollbackChecking" :disabled="rollbackSubmitting" @click="checkRollback">重新检查</Button>
    <Form layout="vertical">
      <Form.Item label="回滚原因" required>
        <Input.TextArea v-model:value="rollbackReason" aria-label="回滚原因" :maxlength="4000" :disabled="rollbackSubmitting" />
      </Form.Item>
    </Form>
  </Modal>

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
        <Select v-model:value="promoteForm.target_stage" style="width: 100%" @change="handlePromoteTargetChange">
          <Select.Option
            v-for="stage in (VALID_TRANSITIONS[models.find(m => m.model_version_id === promoteForm.version_id)?.stage ?? ''] || [])"
            :key="stage"
            :value="stage"
            :disabled="stage === 'production'"
          >
            {{ STAGE_LABELS[stage] || stage }}
          </Select.Option>
        </Select>
      </Form.Item>
      <div v-if="promoteForm.target_stage === 'shadow'" class="stage-hint">
        影子阶段只做旁路评分，不触发告警。
      </div>
      <div v-else-if="promoteForm.target_stage === 'canary'" class="stage-hint">
        金丝雀阶段会成为该摄像头的主模型。
      </div>
      <Form.Item label="操作人">
        <Input v-model:value="promoteForm.triggered_by" placeholder="留空则使用当前登录用户" />
      </Form.Item>
      <Form.Item v-if="promoteForm.target_stage === 'canary'" label="金丝雀摄像头" required>
        <Select v-model:value="promoteForm.canary_camera_id" style="width: 100%" disabled>
          <Select.Option v-if="promoteOwnCameraId()" :key="promoteOwnCameraId()" :value="promoteOwnCameraId()">
            {{ promoteOwnCameraId() }}
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
    title="重新导出为候选版本"
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
    <template v-else-if="shadowReport">
      <Descriptions :column="1" bordered size="small">
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
      <div v-if="shadowReport.gate_report" style="margin-top: 12px">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px">
          <strong>推进到金丝雀检查</strong>
          <Tag :color="shadowReport.gate_report.ready_for_canary ? 'green' : 'orange'">
            {{ shadowReport.gate_report.ready_for_canary ? '满足' : '未满足' }}
          </Tag>
        </div>
        <div v-if="enabledGateChecks().length === 0" style="color: #666; font-size: 12px">
          未配置影子证据门槛
        </div>
        <Descriptions v-else :column="1" bordered size="small">
          <Descriptions.Item
            v-for="check in enabledGateChecks()"
            :key="check.name"
            :label="check.label"
          >
            <Space>
              <Tag :color="check.passed ? 'green' : 'red'">
                {{ check.passed ? '通过' : '未通过' }}
              </Tag>
              <span>
                {{ formatGateValue(check.name, check.value) }} /
                {{ formatGateValue(check.name, check.limit) }}
              </span>
            </Space>
          </Descriptions.Item>
        </Descriptions>
      </div>
    </template>
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
:deep(.row-focused) td {
  background: #fff7e6 !important;
  box-shadow: inset 0 1px 0 rgba(250, 173, 20, 0.35), inset 0 -1px 0 rgba(250, 173, 20, 0.35);
}
:deep(.row-focused td:first-child) {
  box-shadow:
    inset 3px 0 0 #faad14,
    inset 0 1px 0 rgba(250, 173, 20, 0.35),
    inset 0 -1px 0 rgba(250, 173, 20, 0.35);
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

.stage-hint {
  margin: -4px 0 12px;
  padding: 8px 10px;
  border: 1px solid #d6e4ff;
  border-radius: 6px;
  background: #f0f5ff;
  color: #1d39c4;
  font-size: 12px;
  line-height: 1.5;
}

.runtime-state-cell {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 3px;
}
.runtime-state-cell small {
  max-width: 150px;
  overflow: hidden;
  color: var(--argus-text-muted);
  font-family: var(--argus-mono);
  font-size: 10px;
  line-height: 1.3;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
