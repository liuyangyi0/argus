<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { Button, Card, Modal } from 'ant-design-vue'
import { RouterLink } from 'vue-router'
import {
  activateSceneProfile,
  approveSceneProfile,
  createSceneTrainingJob,
  createSceneWorkflow,
  getSceneWorkflow,
  startSceneBaselineJob,
} from '../../api/sceneWorkflow'
import type { SceneBaselineSource, SceneWorkflow, SceneWorkflowSnapshot } from '../../api/sceneWorkflow'
import type { CameraSummary } from '../../types/api'
import { useAuthStore } from '../../stores/useAuthStore'
import { useWebSocket } from '../../composables/useWebSocket'
import { toApiError } from '../../utils/error'
import AsyncState from '../common/AsyncState.vue'
import StatusBanner from '../common/StatusBanner.vue'

const props = defineProps<{ cameras: CameraSummary[] }>()
const auth = useAuthStore()
const cameraId = ref('')
const snapshot = ref<SceneWorkflowSnapshot | null>(null)
const loaded = ref(false)
const loading = ref(false)
const loadError = ref('')
const loadErrorStatus = ref<number | null>(null)
const actionError = ref('')
const successMessage = ref('')
const busy = ref('')
const workflowId = ref('')
const draftId = ref('')
const referencePath = ref('')
const referenceConfirmed = ref(false)
const sources = ref<SceneBaselineSource[]>([
  { video: '', annotations: '' }, { video: '', annotations: '' },
])
const sampleEvery = ref(30)
const maxImages = ref(2000)
const imageSize = ref(256)
let viewGeneration = 0
let requestRevision = 0
let actionRevision = 0
let lastAttemptAt = 0
let pollTimer: ReturnType<typeof setInterval> | null = null

const selectedWorkflow = computed(() => (
  snapshot.value?.workflows.find(item => item.workflow_id === workflowId.value) ?? null
))
const selectedReference = computed(() => (
  snapshot.value?.references.find(item => item.path === referencePath.value) ?? null
))
const canManage = computed(() => auth.hasRole(['admin', 'engineer']))
const canMutate = computed(() => (
  canManage.value && snapshot.value !== null && !loadError.value && !loading.value && !busy.value
))
const draftValid = computed(() => (
  /^[A-Za-z0-9][A-Za-z0-9_.-]{0,47}$/.test(draftId.value)
  && selectedReference.value !== null && referenceConfirmed.value
))
const sourceInputValid = computed(() => {
  const minimum = snapshot.value?.requirements.min_sources ?? 2
  return sources.value.length >= minimum && sources.value.length <= 32
    && sources.value.every(source => source.video.trim() && source.annotations.trim())
    && new Set(sources.value.map(source => source.video.trim())).size === sources.value.length
    && Number.isInteger(sampleEvery.value) && sampleEvery.value >= 1 && sampleEvery.value <= 10_000
    && Number.isInteger(maxImages.value) && maxImages.value >= 150 && maxImages.value <= 20_000
})
const trainingActive = computed(() => ['pending_confirmation', 'queued', 'running'].includes(
  selectedWorkflow.value?.training_status ?? '',
))
const trainingRetryAvailable = computed(() => (
  selectedWorkflow.value?.state === 'training_submitted'
  && !selectedWorkflow.value.approved && !!selectedWorkflow.value.training_job_id
  && ['failed', 'rejected'].includes(selectedWorkflow.value.training_status ?? '')
))
const canBuild = computed(() => (
  canMutate.value && selectedWorkflow.value !== null && sourceInputValid.value
  && ['draft', 'baseline_failed'].includes(selectedWorkflow.value.state) && !trainingActive.value
  && !selectedWorkflow.value.approved && !selectedWorkflow.value.training_job_id
))
const canTrain = computed(() => (
  canMutate.value && selectedWorkflow.value?.training_ready === true
  && selectedWorkflow.value.baseline_schema === 'argus.scene-profile-baseline.v3'
  && selectedWorkflow.value.train_count >= (snapshot.value?.requirements.min_train_images ?? 100)
  && selectedWorkflow.value.calibration_count >= (snapshot.value?.requirements.min_calibration_images ?? 50)
  && !selectedWorkflow.value.approved
  && ((selectedWorkflow.value.state === 'baseline_ready' && !selectedWorkflow.value.training_job_id)
    || trainingRetryAvailable.value)
  && Number.isInteger(imageSize.value) && imageSize.value >= 64 && imageSize.value <= 1024
))
const canApprove = computed(() => (
  canMutate.value && selectedWorkflow.value?.training_status === 'complete'
  && !!selectedWorkflow.value.model_version_id && !selectedWorkflow.value.approved
))
const selectionRevisionReady = computed(() => {
  if (!snapshot.value) return false
  const active = snapshot.value.profiles.filter(profile => profile.active)
  const revision = snapshot.value.active_profile_revision
  if (active.length > 1) return false
  return active.length === 0 ? revision === null
    : typeof revision === 'string' && /^[0-9a-f]{64}$/.test(revision)
})
const canSelectProfile = computed(() => (
  canMutate.value && selectedWorkflow.value?.approved === true && !selectedWorkflow.value.active
  && selectionRevisionReady.value
))

async function refresh() {
  const camera = cameraId.value
  if (!camera) return
  const revision = ++requestRevision
  const generation = viewGeneration
  loading.value = true
  lastAttemptAt = Date.now()
  try {
    const result = await getSceneWorkflow(camera)
    if (generation !== viewGeneration || revision !== requestRevision) return
    if (result.camera_id !== camera) throw new Error('场景工作流返回了不匹配的摄像头')
    snapshot.value = result
    if (!result.references.some(item => item.path === referencePath.value)) {
      referencePath.value = result.references[0]?.path ?? ''
      referenceConfirmed.value = false
    }
    if (!result.workflows.some(item => item.workflow_id === workflowId.value)) {
      workflowId.value = result.workflows[0]?.workflow_id ?? ''
    }
    loadError.value = ''
    loadErrorStatus.value = null
  } catch (error: unknown) {
    if (generation !== viewGeneration || revision !== requestRevision) return
    const failure = toApiError(error, '加载场景工作流失败')
    loadError.value = failure.message
    loadErrorStatus.value = failure.httpStatus
    if ([403, 404].includes(failure.httpStatus ?? 0)) snapshot.value = null
  } finally {
    if (generation === viewGeneration && revision === requestRevision) {
      loaded.value = true
      loading.value = false
    }
  }
}

useWebSocket<{ camera_id?: string }>({
  topics: ['models', 'model_release', 'tasks'],
  onMessage: (_topic, data) => {
    if (data?.camera_id && data.camera_id !== cameraId.value) return
    // Do not drop a mutation event just because an older GET is in flight.
    // The new request revision makes the authoritative refresh supersede it.
    void refresh()
  },
  fallbackPoll: async () => { if (!loading.value && !busy.value) await refresh() },
  fallbackInterval: 60_000,
  reconcileInterval: 60_000,
})

watch(() => props.cameras.map(camera => camera.camera_id), ids => {
  if (!ids.includes(cameraId.value)) cameraId.value = ids[0] ?? ''
}, { immediate: true })

watch(cameraId, () => {
  viewGeneration += 1
  requestRevision += 1
  actionRevision += 1
  snapshot.value = null
  loaded.value = false
  loading.value = false
  loadError.value = ''
  loadErrorStatus.value = null
  actionError.value = ''
  successMessage.value = ''
  busy.value = ''
  workflowId.value = ''
  draftId.value = ''
  referencePath.value = ''
  referenceConfirmed.value = false
  sources.value = [{ video: '', annotations: '' }, { video: '', annotations: '' }]
  void refresh()
}, { immediate: true })

watch(referencePath, () => { referenceConfirmed.value = false })
watch(() => selectedReference.value?.sha256, () => { referenceConfirmed.value = false })

async function runAction(
  key: string,
  operation: (camera: string) => Promise<unknown>,
  success: string,
): Promise<boolean> {
  if (!canMutate.value) return false
  const generation = viewGeneration
  const revision = ++actionRevision
  const camera = cameraId.value
  busy.value = key
  actionError.value = ''
  successMessage.value = ''
  try {
    await operation(camera)
    if (generation !== viewGeneration || revision !== actionRevision) return false
    successMessage.value = success
    await refresh()
    // A camera change can happen while this follow-up GET is pending too.
    // Its ignored response must not authorize the caller to clear the new
    // camera's draft or replace its workflow selection.
    return generation === viewGeneration && revision === actionRevision
  } catch (error: unknown) {
    if (generation === viewGeneration && revision === actionRevision) {
      actionError.value = toApiError(error, '场景工作流操作失败').message
    }
    return false
  } finally {
    if (generation === viewGeneration && revision === actionRevision) busy.value = ''
  }
}

async function createDraft() {
  const reference = selectedReference.value
  if (!draftValid.value || !reference) return
  const id = draftId.value
  const created = await runAction('create', camera => createSceneWorkflow(camera, {
    workflow_id: id,
    reference_path: reference.path,
    reference_sha256: reference.sha256,
    normal_reference_confirmed: true,
  }), '场景草稿已创建；尚未批准、激活或发布。')
  if (created) {
    workflowId.value = id
    draftId.value = ''
    referenceConfirmed.value = false
  }
}

async function buildBaseline() {
  const workflow = selectedWorkflow.value
  if (!workflow || !canBuild.value) return
  const payload = {
    sources: sources.value.map(source => ({ video: source.video.trim(), annotations: source.annotations.trim() })),
    sample_every: sampleEvery.value,
    max_images: maxImages.value,
  }
  await runAction('baseline', camera => startSceneBaselineJob(camera, workflow.workflow_id, payload),
    '严格 v3 基线任务已提交，需等待服务端完成全部审核与解码校验。')
}

async function train() {
  const workflow = selectedWorkflow.value
  if (!workflow || !canTrain.value) return
  await runAction('training', camera => createSceneTrainingJob(camera, workflow.workflow_id, {
    model_type: 'patchcore', image_size: imageSize.value,
  }), '训练任务已提交为待人工确认；请到“训练与评估”确认，不会自动训练或发布。')
}

function confirmationIdentity(workflow: SceneWorkflow): string {
  return JSON.stringify([
    workflow.workflow_id, workflow.profile_id, workflow.reference_sha256,
    workflow.baseline_schema, workflow.training_ready,
    workflow.train_count, workflow.calibration_count,
    workflow.training_job_id, workflow.training_status, workflow.model_version_id,
    workflow.approved, workflow.active,
  ])
}

function activeProfileIdentity(): string {
  return JSON.stringify(snapshot.value?.profiles.filter(profile => profile.active)
    .map(profile => profile.profile_id).sort() ?? [])
}

function requestApproval() {
  const workflow = selectedWorkflow.value
  if (!workflow || !canApprove.value) return
  const camera = cameraId.value
  const generation = viewGeneration
  const identity = confirmationIdentity(workflow)
  Modal.confirm({
    title: '批准场景参考与训练模型的绑定？',
    content: `请人工核对场景 ${workflow.profile_id} 与模型 ${workflow.model_version_id}。批准不会激活场景，也不会恢复检测或发布模型；独立回归和可信 72 小时 soak 仍不可省略。`,
    okText: '确认批准绑定（不激活）',
    cancelText: '取消',
    async onOk() {
      if (cameraId.value !== camera || viewGeneration !== generation) return
      const current = selectedWorkflow.value
      if (!current || !canApprove.value || current.workflow_id !== workflow.workflow_id
        || confirmationIdentity(current) !== identity) {
        actionError.value = '审批目标或状态已变化，请重新核对并再次确认。'
        return
      }
      await runAction('approve', id => approveSceneProfile(id, workflow.workflow_id),
        '场景模型绑定已批准；活动场景和旧模型保持不变，请继续现有受控发布流程。')
    },
  })
}

function requestProfileSelection() {
  const workflow = selectedWorkflow.value
  const currentSnapshot = snapshot.value
  if (!workflow || !currentSnapshot || !canSelectProfile.value) return
  const camera = cameraId.value
  const generation = viewGeneration
  const identity = confirmationIdentity(workflow)
  const activeIdentity = activeProfileIdentity()
  const expectedActiveProfileId = currentSnapshot.profiles.find(profile => profile.active)?.profile_id ?? null
  const expectedActiveProfileRevision = currentSnapshot.active_profile_revision
  Modal.confirm({
    title: '选择此场景作为活动配置（不切换运行模型）？',
    content: `只选择已批准的场景 ${workflow.profile_id} 及其严格 v3 基线，不会发布模型或恢复评分。运行时可能仍是旧模型，重载或重启时身份不匹配会停检。新模型仍必须经过现有发布服务；独立回归和可信 72 小时 soak 未通过前禁止 production。`,
    okText: '选择场景，保留现有运行模型',
    cancelText: '取消',
    async onOk() {
      if (cameraId.value !== camera || viewGeneration !== generation) return
      const current = selectedWorkflow.value
      if (!current || !canSelectProfile.value || current.workflow_id !== workflow.workflow_id
        || confirmationIdentity(current) !== identity
        || activeProfileIdentity() !== activeIdentity
        || snapshot.value?.active_profile_revision !== expectedActiveProfileRevision) {
        actionError.value = '所选场景或当前活动配置已变化，请重新核对并再次确认。'
        return
      }
      await runAction('select-profile', id => activateSceneProfile(
        id, workflow.workflow_id, expectedActiveProfileId, expectedActiveProfileRevision,
      ),
        '已选择场景配置，但未切换运行模型、恢复评分或发布；请检查运行时绑定并继续受控发布。')
    },
  })
}

onMounted(() => {
  void auth.fetchCurrentUser()
  pollTimer = setInterval(() => {
    if (!cameraId.value || loading.value || busy.value) return
    const active = snapshot.value?.workflows.some(workflow => (
      workflow.state === 'building_baseline'
      || ['pending_confirmation', 'queued', 'running'].includes(workflow.training_status ?? '')
    ))
    if (active || Date.now() - lastAttemptAt >= 60_000) void refresh()
  }, 5000)
})

onUnmounted(() => {
  viewGeneration += 1
  requestRevision += 1
  actionRevision += 1
  if (pollTimer !== null) clearInterval(pollTimer)
})
</script>

<template>
  <Card title="场景 → 合规基线 → 训练校准 → 受控发布" class="scene-workflow">
    <StatusBanner
      class="scene-workflow__notice"
      status="info"
      title="创建、训练、批准、激活与发布是不同步骤"
      description="本流程不自动信任新机位，不自动批准或激活场景，不替代独立回归和可信 72 小时 soak，也不改变 SceneGuard 放行条件。"
    />
    <label class="scene-workflow__field">
      <span>摄像头</span>
      <select v-model="cameraId" data-test="scene-camera" aria-label="场景工作流摄像头">
        <option v-for="camera in cameras" :key="camera.camera_id" :value="camera.camera_id">
          {{ camera.name || camera.camera_id }}（{{ camera.camera_id }}）
        </option>
      </select>
    </label>
    <AsyncState
      :loading="loading && !loaded"
      :error="!snapshot && ![403, 404].includes(loadErrorStatus ?? 0) ? loadError : ''"
      :forbidden="loadErrorStatus === 403"
      :not-found="loadErrorStatus === 404"
      :empty="!cameraId"
      empty-title="请先配置摄像头"
      @retry="refresh"
    >
      <StatusBanner
        v-if="loadError && snapshot"
        class="scene-workflow__notice"
        status="warning"
        title="工作流状态刷新失败：当前为上次数据，所有写入操作已禁用"
        :description="loadError"
      >
        <template #action><Button size="small" @click="refresh">重试</Button></template>
      </StatusBanner>
      <StatusBanner
        v-if="!canManage"
        class="scene-workflow__notice"
        status="warning"
        title="当前账号只可查看；场景与训练操作需要管理员或工程人员权限"
      />
      <StatusBanner v-if="actionError" class="scene-workflow__notice" status="error" title="操作失败" :description="actionError" />
      <StatusBanner v-if="successMessage" class="scene-workflow__notice" status="success" :title="successMessage" />

      <template v-if="snapshot">
        <StatusBanner
          v-if="snapshot.issues?.length"
          class="scene-workflow__notice"
          status="warning"
          title="部分工作流记录缺失或损坏，需管理员核查"
          :description="snapshot.issues.map(issue => `${issue.workflow_id}：${issue.message}`).join('；')"
        />
        <p class="scene-workflow__muted">
          正式基线：{{ snapshot.requirements.baseline_schema }}；至少 {{ snapshot.requirements.min_sources }} 个独立且经人工审核的正常录制 session，
          训练 ≥{{ snapshot.requirements.min_train_images }} 张，校准 ≥{{ snapshot.requirements.min_calibration_images }} 张。
        </p>
        <div class="scene-workflow__grid">
          <section class="scene-workflow__section">
            <h3>1. 明确参考并创建草稿</h3>
            <label class="scene-workflow__field">
              <span>工作流 / 新场景 ID</span>
              <input v-model="draftId" data-test="scene-draft-id" maxlength="48" :disabled="!canMutate" autocomplete="off" />
            </label>
            <label class="scene-workflow__field">
              <span>已存在的受控参考图片（不是实时首帧）</span>
              <select v-model="referencePath" data-test="scene-reference" :disabled="!canMutate">
                <option v-for="reference in snapshot.references" :key="reference.path" :value="reference.path">{{ reference.path }}</option>
              </select>
            </label>
            <p v-if="selectedReference" class="scene-workflow__digest">SHA256：{{ selectedReference.sha256 }}</p>
            <p v-else class="scene-workflow__muted">没有可选参考图片，请先准备受控参考；不会自动采用实时画面。</p>
            <label class="scene-workflow__check">
              <input v-model="referenceConfirmed" data-test="scene-reference-confirmed" type="checkbox" :disabled="!canMutate" />
              我已人工核对此参考为目标正常场景（仅创建草稿，不授予运行权限）
            </label>
            <Button data-test="scene-create" type="primary" :disabled="!canMutate || !draftValid" :loading="busy === 'create'" @click="createDraft">创建未批准草稿</Button>
          </section>

          <section class="scene-workflow__section">
            <h3>已登记场景</h3>
            <p v-if="snapshot.profiles.length === 0" class="scene-workflow__muted">暂无已登记场景；草稿不等于批准场景。</p>
            <ul v-else class="scene-workflow__list">
              <li v-for="profile in snapshot.profiles" :key="profile.profile_id">
                <strong>{{ profile.profile_id }}</strong> · {{ profile.approved ? '已批准' : '未批准' }} · {{ profile.active ? '已选场景（非运行时确认）' : '未选择' }}
                <div class="scene-workflow__muted">绑定模型：{{ profile.model_version_id || '未绑定' }}</div>
              </li>
            </ul>
          </section>
        </div>

        <section class="scene-workflow__section scene-workflow__selected">
          <label class="scene-workflow__field">
            <span>继续已有工作流</span>
            <select v-model="workflowId" data-test="scene-workflow-select" :disabled="!!busy">
              <option v-for="workflow in snapshot.workflows" :key="workflow.workflow_id" :value="workflow.workflow_id">
                {{ workflow.workflow_id }} · {{ workflow.state }}
              </option>
            </select>
          </label>
          <p v-if="!selectedWorkflow" class="scene-workflow__muted">请先创建场景草稿。</p>
          <template v-else>
            <dl class="scene-workflow__facts">
              <div><dt>状态</dt><dd>{{ selectedWorkflow.state }}</dd></div>
              <div><dt>创建记录</dt><dd>{{ selectedWorkflow.created_by }} · {{ selectedWorkflow.created_at }}</dd></div>
              <div><dt>基线格式</dt><dd>{{ selectedWorkflow.baseline_schema || '尚未构建' }}</dd></div>
              <div><dt>训练 / 校准样本</dt><dd>{{ selectedWorkflow.train_count }} / {{ selectedWorkflow.calibration_count }}</dd></div>
              <div><dt>基线任务</dt><dd>{{ selectedWorkflow.baseline_task_id || '尚未创建' }}</dd></div>
              <div><dt>训练任务</dt><dd>{{ selectedWorkflow.training_job_id || '尚未创建' }} · {{ selectedWorkflow.training_status || '未开始' }}</dd></div>
              <div><dt>绑定模型</dt><dd>{{ selectedWorkflow.model_version_id || '训练完成后由服务端绑定' }}</dd></div>
              <div><dt>批准 / 场景选择（非运行时状态）</dt><dd>{{ selectedWorkflow.approved ? '已批准' : '未批准' }} / {{ selectedWorkflow.active ? '已选场景' : '未选择' }}</dd></div>
            </dl>
            <StatusBanner v-if="selectedWorkflow.error" class="scene-workflow__notice" status="error" title="工作流失败" :description="selectedWorkflow.error" />
            <div v-if="selectedWorkflow.blockers.length" class="scene-workflow__blockers" role="status">
              <strong>服务端当前阻断原因</strong>
              <ul><li v-for="reason in selectedWorkflow.blockers" :key="reason">{{ reason }}</li></ul>
            </div>
            <div class="scene-workflow__grid">
              <section>
                <h3>2. 构建严格 v3 基线</h3>
                <p class="scene-workflow__muted">路径相对于 {{ snapshot.source_root_label }}；每份 annotations 必须记录完整人工正常审核及不同 source_session_id。服务端分配 train/calibration，不随机拆相邻帧。</p>
                <div v-for="(source, index) in sources" :key="index" class="scene-workflow__source">
                  <label class="scene-workflow__field"><span>正常录制 {{ index + 1 }} · 视频路径</span><input v-model="source.video" :data-test="`scene-source-video-${index}`" :disabled="!canMutate" /></label>
                  <label class="scene-workflow__field"><span>人工审核 annotations 路径</span><input v-model="source.annotations" :data-test="`scene-source-annotations-${index}`" :disabled="!canMutate" /></label>
                  <Button v-if="sources.length > snapshot.requirements.min_sources" size="small" :disabled="!canMutate" @click="sources.splice(index, 1)">移除录制 {{ index + 1 }}</Button>
                </div>
                <Button size="small" :disabled="!canMutate || sources.length >= 32" @click="sources.push({ video: '', annotations: '' })">添加独立正常录制</Button>
                <div class="scene-workflow__numbers">
                  <label class="scene-workflow__field"><span>采样间隔（原始帧）</span><input v-model.number="sampleEvery" type="number" min="1" max="10000" step="1" :disabled="!canMutate" /></label>
                  <label class="scene-workflow__field"><span>图片预算</span><input v-model.number="maxImages" type="number" min="150" max="20000" step="1" :disabled="!canMutate" /></label>
                </div>
                <Button data-test="scene-build" :disabled="!canBuild" :loading="busy === 'baseline'" @click="buildBaseline">提交严格基线任务</Button>
              </section>
              <section>
                <h3>3. 训练与校准</h3>
                <p class="scene-workflow__muted">只有服务端验证通过的 v3 基线可提交。训练仍需在训练页人工确认，校准使用独立 session。</p>
                <label class="scene-workflow__field"><span>PatchCore 输入尺寸</span><input v-model.number="imageSize" type="number" min="64" max="1024" step="1" :disabled="!canMutate" /></label>
                <p v-if="trainingRetryAvailable" class="scene-workflow__muted">上次训练已失败或被拒绝，可显式提交新任务复用同一冻结基线；历史任务保留，新任务仍需再次人工确认。提交结果未知或任务仍在运行时不可重复提交。</p>
                <Button data-test="scene-train" :disabled="!canTrain" :loading="busy === 'training'" @click="train">{{ trainingRetryAvailable ? '显式重试训练（仍需确认）' : '创建待确认训练任务' }}</Button>
                <h3>4. 人工批准绑定，再走现有发布门</h3>
                <p class="scene-workflow__muted">服务端检查训练完成及校准有效后才允许批准；批准不会切换当前活动场景或旧模型。</p>
                <Button data-test="scene-approve" :disabled="!canApprove" :loading="busy === 'approve'" @click="requestApproval">人工批准场景模型绑定</Button>
                <p class="scene-workflow__muted">批准后可单独选择场景配置。这只变更所选 profile 与基线，不代表运行模型已应用，也不恢复评分或告警。</p>
                <StatusBanner
                  v-if="selectedWorkflow.approved && !selectedWorkflow.active && !selectionRevisionReady"
                  class="scene-workflow__notice"
                  status="warning"
                  title="活动场景版本未知，选择操作保持禁用"
                  description="缺少明确的活动场景 ID/版本或状态不一致，请刷新后重新确认；不会无条件覆盖其他操作者的选择。"
                >
                  <template #action><Button size="small" @click="refresh">重新获取状态</Button></template>
                </StatusBanner>
                <Button data-test="scene-select-profile" :disabled="!canSelectProfile" :loading="busy === 'select-profile'" @click="requestProfileSelection">选择活动场景（不切模型）</Button>
                <p><RouterLink to="/models/training">前往训练与评估</RouterLink> · <RouterLink to="/models/registry">查看模型发布 readiness</RouterLink></p>
              </section>
            </div>
          </template>
        </section>
      </template>
    </AsyncState>
  </Card>
</template>

<style scoped>
.scene-workflow { margin-bottom: 20px; }
.scene-workflow__notice { margin-bottom: 14px; }
.scene-workflow__grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
.scene-workflow__section { min-width: 0; padding: 14px; border: 1px solid var(--line-1, #dce1e8); border-radius: var(--r-sm, 6px); }
.scene-workflow__selected { margin-top: 18px; }
.scene-workflow h3 { margin: 0 0 12px; color: var(--ink-1); font-size: 14px; }
.scene-workflow h3:not(:first-child) { margin-top: 22px; }
.scene-workflow__field { display: flex; flex-direction: column; gap: 6px; margin-bottom: 12px; color: var(--ink-2); font-size: 13px; }
.scene-workflow__field input, .scene-workflow__field select { width: 100%; min-width: 0; padding: 7px 10px; border: 1px solid var(--line-2, #c7cdd6); border-radius: 4px; background: var(--surface-1, #fff); color: var(--ink-1, #162033); font: inherit; }
.scene-workflow__field input:focus-visible, .scene-workflow__field select:focus-visible { outline: 2px solid var(--brand, #1d4ed8); outline-offset: 2px; }
.scene-workflow__field input:disabled, .scene-workflow__field select:disabled { opacity: 0.65; }
.scene-workflow__check { display: flex; align-items: flex-start; gap: 8px; margin-bottom: 12px; color: var(--ink-2); font-size: 13px; }
.scene-workflow__check input { margin-top: 3px; }
.scene-workflow__muted { color: var(--ink-3); font-size: 12px; line-height: 1.65; overflow-wrap: anywhere; }
.scene-workflow__digest { color: var(--ink-3); font-family: monospace; font-size: 11px; overflow-wrap: anywhere; }
.scene-workflow__list { margin: 0; padding-left: 18px; }
.scene-workflow__list li { margin-bottom: 10px; }
.scene-workflow__facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 20px; font-size: 12px; }
.scene-workflow__facts > div { min-width: 0; }
.scene-workflow__facts dt { color: var(--ink-3); }
.scene-workflow__facts dd { margin: 2px 0 8px; color: var(--ink-1); overflow-wrap: anywhere; }
.scene-workflow__source { padding: 10px 0; border-top: 1px solid var(--line-1, #dce1e8); }
.scene-workflow__numbers { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }
.scene-workflow__blockers { margin: 12px 0 18px; padding: 10px 12px; border-left: 3px solid var(--warning, #a16207); color: var(--ink-2); font-size: 12px; }
.scene-workflow__blockers ul { margin: 6px 0 0; padding-left: 18px; }
@media (max-width: 1000px) {
  .scene-workflow__grid, .scene-workflow__facts { grid-template-columns: minmax(0, 1fr); }
}
</style>
