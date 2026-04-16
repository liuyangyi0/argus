<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import {
  Card, Tag, Switch, Button, Tooltip, Space, Typography, message, Alert,
  InputNumber, Slider, Input, Popconfirm,
} from 'ant-design-vue'
import {
  ReloadOutlined, WarningOutlined,
  EditOutlined, SaveOutlined, CloseOutlined,
  PlusOutlined, DeleteOutlined,
} from '@ant-design/icons-vue'

import {
  getCrossCameraConfig,
  updateCrossCameraConfig,
  toggleModule,
  type CrossCameraConfigPayload,
  type CrossCameraOverlapPair,
} from '../../api'

defineOptions({ name: 'CrossCameraPanel' })

/* ── Draft types ── */
interface DraftScalars {
  corroboration_threshold: number
  max_age_seconds: number
  uncorroborated_severity_downgrade: number
}

interface DraftPair {
  camera_a: string
  camera_b: string
  homography: number[][]
}

/* ── Reactive state ── */
const cfg = ref<CrossCameraConfigPayload | null>(null)
const loading = ref(false)
const toggling = ref(false)
const saving = ref(false)
const error = ref<string | null>(null)

const editing = ref(false)
const draftScalars = ref<DraftScalars>({
  corroboration_threshold: 0.3,
  max_age_seconds: 5.0,
  uncorroborated_severity_downgrade: 1,
})
const draftPairs = ref<DraftPair[]>([])

/* ── Data loading ── */
async function load() {
  loading.value = true
  error.value = null
  try {
    cfg.value = await getCrossCameraConfig()
  } catch (e: any) {
    error.value = e?.response?.data?.msg || e?.message || '加载跨相机配置失败'
    cfg.value = null
  } finally {
    loading.value = false
  }
}

function resetDraftFromCfg(c: CrossCameraConfigPayload) {
  draftScalars.value = {
    corroboration_threshold: c.corroboration_threshold,
    max_age_seconds: c.max_age_seconds,
    uncorroborated_severity_downgrade: c.uncorroborated_severity_downgrade,
  }
  draftPairs.value = c.overlap_pairs.map(p => ({
    camera_a: p.camera_a,
    camera_b: p.camera_b,
    homography: p.homography.map(row => [...row]),
  }))
}

// Keep drafts in sync when cfg reloads and we're NOT editing.
watch(cfg, (next) => {
  if (next && !editing.value) resetDraftFromCfg(next)
})

/* ── Toggle ── */
async function handleToggle(checked: string | number | boolean) {
  if (!cfg.value) return
  const next = Boolean(checked)
  toggling.value = true
  try {
    await toggleModule('cross_camera.enabled', next)
    cfg.value.enabled = next
    message.success(next
      ? '跨相机关联已启用（已运行的摄像头需要重启才会生效）'
      : '跨相机关联已关闭')
    await load()
  } catch (e: any) {
    message.error(e?.response?.data?.msg || '切换失败')
  } finally {
    toggling.value = false
  }
}

/* ── Edit mode ── */
function enterEdit() {
  if (!cfg.value) return
  resetDraftFromCfg(cfg.value)
  editing.value = true
}

function cancelEdit() {
  editing.value = false
  if (cfg.value) resetDraftFromCfg(cfg.value)
}

function addPair() {
  draftPairs.value.push({
    camera_a: '',
    camera_b: '',
    homography: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  })
}

function removePair(index: number) {
  draftPairs.value.splice(index, 1)
}

function resetToIdentity(index: number) {
  draftPairs.value[index].homography = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
}

async function saveDraft() {
  if (!cfg.value) return
  saving.value = true
  try {
    const result: any = await updateCrossCameraConfig({
      overlap_pairs: draftPairs.value,
      corroboration_threshold: draftScalars.value.corroboration_threshold,
      max_age_seconds: draftScalars.value.max_age_seconds,
      uncorroborated_severity_downgrade: draftScalars.value.uncorroborated_severity_downgrade,
    })
    const updated = result?.correlator_updated ?? false
    message.success(updated
      ? '配置已保存，关联器已热更新'
      : '配置已保存（关联器未运行，需重启摄像头生效）')
    editing.value = false
    await load()
  } catch (e: any) {
    message.error(e?.response?.data?.msg || e?.message || '保存失败')
  } finally {
    saving.value = false
  }
}

/* ── Computed ── */
const runtimeSummary = computed(() => {
  const r = cfg.value?.runtime
  if (!r) return null
  if (!cfg.value?.enabled) return '分组关联器已关闭'
  if (r.total_pipelines === 0) return '无运行中的摄像头管线'
  if (r.correlator_present) {
    return `${r.total_pipelines} 路管线运行中，相关器已附加`
  }
  return `${r.total_pipelines} 路管线运行中，但相关器尚未附加（配置是后启用的 — 需重启摄像头）`
})

onMounted(load)
</script>

<template>
  <Card :loading="loading && !cfg" :bordered="false">
    <template #title>
      <Space>
        <span>跨相机关联校验器</span>
        <Tag v-if="cfg" :color="cfg.enabled ? 'green' : 'default'">
          {{ cfg.enabled ? '已启用' : '已关闭' }}
        </Tag>
      </Space>
    </template>
    <template #extra>
      <Button size="small" :loading="loading" @click="load">
        <template #icon><ReloadOutlined /></template>
        刷新
      </Button>
    </template>

    <Alert v-if="error" type="error" :message="error" show-icon style="margin-bottom: 16px" />

    <div v-if="cfg" class="cross-camera-panel">
      <!-- Master toggle + runtime status -->
      <div class="panel-section">
        <div class="toggle-row">
          <div class="toggle-info">
            <Typography.Text strong>启用跨相机关联</Typography.Text>
            <Typography.Text type="secondary" style="font-size: 12px">
              多个相机视野重叠时，由另一路相机交叉证实异常
            </Typography.Text>
          </div>
          <Switch
            :checked="cfg.enabled"
            :loading="toggling"
            @update:checked="handleToggle"
          />
        </div>
        <Typography.Text v-if="runtimeSummary" type="secondary" style="font-size: 12px">
          {{ runtimeSummary }}
        </Typography.Text>
        <Alert
          v-if="cfg.enabled && !cfg.runtime.correlator_present && cfg.runtime.total_pipelines > 0"
          type="warning"
          show-icon
          style="margin-top: 8px"
          message="切换后的 enabled 状态只会影响新启动的摄像头管线，已运行的管线需要重启才会挂载关联器。"
        >
          <template #icon><WarningOutlined /></template>
        </Alert>
      </div>

      <!-- Model configuration -->
      <div class="panel-section">
        <div class="vocab-header">
          <Typography.Text strong>模型配置</Typography.Text>
          <Space v-if="!editing">
            <Button size="small" @click="enterEdit">
              <template #icon><EditOutlined /></template>
              编辑
            </Button>
          </Space>
          <Space v-else>
            <Button size="small" @click="cancelEdit" :disabled="saving">
              <template #icon><CloseOutlined /></template>
              取消
            </Button>
            <Button size="small" type="primary" :loading="saving" @click="saveDraft">
              <template #icon><SaveOutlined /></template>
              保存并热推送
            </Button>
          </Space>
        </div>

        <!-- View mode -->
        <div v-if="!editing" class="kv-grid">
          <div class="kv-row">
            <Tooltip title="对面相机在映射位置必须达到的最小异常分数，否则视为未证实">
              <span class="k">证实阈值 (corroboration_threshold)</span>
            </Tooltip>
            <span class="v">{{ cfg.corroboration_threshold.toFixed(2) }}</span>
          </div>
          <div class="kv-row">
            <Tooltip title="另一路相机异常热力图的有效期；超过此时长视为缺失">
              <span class="k">最大滞后时间 (max_age_seconds)</span>
            </Tooltip>
            <span class="v">{{ cfg.max_age_seconds.toFixed(1) }} s</span>
          </div>
          <div class="kv-row">
            <Tooltip title="未证实告警向下降级的等级数 (0-2)">
              <span class="k">严重度降级 (uncorroborated_severity_downgrade)</span>
            </Tooltip>
            <span class="v">{{ cfg.uncorroborated_severity_downgrade }}</span>
          </div>
          <div class="kv-row">
            <Tooltip title="当前 overlap_pairs 列表的相机对数量">
              <span class="k">已配置相机对数量</span>
            </Tooltip>
            <span class="v">{{ cfg.overlap_pairs.length }}</span>
          </div>
        </div>

        <!-- Edit mode: sliders + inputs -->
        <div v-else class="edit-grid">
          <Typography.Text type="secondary" style="font-size: 12px; display: block">
            保存后立刻推送到运行中的关联器。enabled 和相机对变更需重启管线。
          </Typography.Text>

          <div class="edit-row">
            <div class="edit-label">
              <Tooltip title="对面相机在映射位置必须达到的最小异常分数">
                <span>证实阈值 <code>corroboration_threshold</code></span>
              </Tooltip>
            </div>
            <Slider
              v-model:value="draftScalars.corroboration_threshold"
              :min="0.1"
              :max="0.9"
              :step="0.05"
              :marks="{ 0.1: '0.1', 0.3: '0.3', 0.5: '0.5', 0.9: '0.9' }"
              class="edit-slider"
            />
            <InputNumber
              v-model:value="draftScalars.corroboration_threshold"
              :min="0.1"
              :max="0.9"
              :step="0.05"
              :precision="2"
              size="small"
              class="edit-number"
            />
          </div>

          <div class="edit-row">
            <div class="edit-label">
              <Tooltip title="另一路相机异常热力图的有效期（秒）">
                <span>最大滞后时间 <code>max_age_seconds</code></span>
              </Tooltip>
            </div>
            <Slider
              v-model:value="draftScalars.max_age_seconds"
              :min="1"
              :max="30"
              :step="0.5"
              :marks="{ 1: '1', 5: '5', 15: '15', 30: '30' }"
              class="edit-slider"
            />
            <InputNumber
              v-model:value="draftScalars.max_age_seconds"
              :min="1"
              :max="30"
              :step="0.5"
              :precision="1"
              size="small"
              class="edit-number"
            />
          </div>

          <div class="edit-row">
            <div class="edit-label">
              <Tooltip title="未证实告警向下降级的等级数 (0=不降级, 2=降两级)">
                <span>严重度降级 <code>severity_downgrade</code></span>
              </Tooltip>
            </div>
            <Slider
              v-model:value="draftScalars.uncorroborated_severity_downgrade"
              :min="0"
              :max="2"
              :step="1"
              :marks="{ 0: '0', 1: '1', 2: '2' }"
              class="edit-slider"
            />
            <InputNumber
              v-model:value="draftScalars.uncorroborated_severity_downgrade"
              :min="0"
              :max="2"
              :step="1"
              size="small"
              class="edit-number"
            />
          </div>
        </div>
      </div>

      <!-- Overlap pairs list -->
      <div class="panel-section">
        <Typography.Text strong>相机对（overlap_pairs）</Typography.Text>

        <!-- View mode -->
        <template v-if="!editing">
          <div v-if="cfg.overlap_pairs.length === 0" class="empty-pairs">
            尚未配置相机对，点击上方"编辑"按钮添加。
          </div>
          <div v-else class="pairs-list">
            <div
              v-for="(pair, idx) in cfg.overlap_pairs"
              :key="`${pair.camera_a}-${pair.camera_b}-${idx}`"
              class="pair-card"
            >
              <div class="pair-header">
                <Tag color="blue">{{ pair.camera_a }}</Tag>
                <span class="pair-arrow">&harr;</span>
                <Tag color="geekblue">{{ pair.camera_b }}</Tag>
              </div>
              <div class="matrix-label">Homography (3&times;3)</div>
              <div class="matrix-grid">
                <div
                  v-for="(row, ri) in pair.homography"
                  :key="`row-${ri}`"
                  class="matrix-row"
                >
                  <div
                    v-for="(cell, ci) in row"
                    :key="`cell-${ri}-${ci}`"
                    class="matrix-cell"
                  >
                    {{ cell.toFixed(3) }}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </template>

        <!-- Edit mode -->
        <template v-else>
          <div v-if="draftPairs.length === 0" class="empty-pairs">
            尚未配置相机对，点击下方按钮添加第一个。
          </div>
          <div v-else class="pairs-list">
            <div
              v-for="(pair, idx) in draftPairs"
              :key="`draft-${idx}`"
              class="pair-card pair-card--edit"
            >
              <div class="pair-edit-header">
                <div class="pair-cameras">
                  <div class="camera-field">
                    <span class="camera-label">Camera A</span>
                    <Input
                      v-model:value="pair.camera_a"
                      size="small"
                      placeholder="camera_id"
                      class="camera-input"
                    />
                  </div>
                  <span class="pair-arrow">&harr;</span>
                  <div class="camera-field">
                    <span class="camera-label">Camera B</span>
                    <Input
                      v-model:value="pair.camera_b"
                      size="small"
                      placeholder="camera_id"
                      class="camera-input"
                    />
                  </div>
                </div>
                <Popconfirm
                  title="确认删除此相机对？"
                  ok-text="删除"
                  cancel-text="取消"
                  @confirm="removePair(idx)"
                >
                  <Button size="small" danger>
                    <template #icon><DeleteOutlined /></template>
                  </Button>
                </Popconfirm>
              </div>

              <div class="matrix-edit-header">
                <span class="matrix-label">Homography (3&times;3)</span>
                <Button size="small" @click="resetToIdentity(idx)">
                  重置为单位矩阵
                </Button>
              </div>
              <div class="matrix-grid">
                <div
                  v-for="(row, ri) in pair.homography"
                  :key="`erow-${ri}`"
                  class="matrix-row"
                >
                  <InputNumber
                    v-for="(_, ci) in row"
                    :key="`ecell-${ri}-${ci}`"
                    v-model:value="pair.homography[ri][ci]"
                    :step="0.001"
                    :precision="6"
                    size="small"
                    class="matrix-cell-input"
                  />
                </div>
              </div>
            </div>
          </div>

          <Button
            type="dashed"
            block
            style="margin-top: 8px"
            @click="addPair"
          >
            <template #icon><PlusOutlined /></template>
            添加相机对
          </Button>
        </template>
      </div>
    </div>
  </Card>
</template>

<style scoped>
.cross-camera-panel {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.panel-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.toggle-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}
.toggle-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

/* ── Section header with edit/save/cancel ── */
.vocab-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

/* ── KV grid (view mode) ── */
.kv-grid {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 14px;
  background: var(--argus-card-bg-solid, rgba(0,0,0,0.02));
  border-radius: 6px;
}
.kv-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  gap: 12px;
}
.kv-row .k {
  color: var(--argus-text-muted);
  cursor: help;
  border-bottom: 1px dashed transparent;
}
.kv-row .k:hover {
  border-bottom-color: var(--argus-text-muted);
}
.kv-row .v {
  color: var(--argus-text);
  font-variant-numeric: tabular-nums;
}

/* ── Edit grid (slider + input rows) ── */
.edit-grid {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 10px 14px;
  background: var(--argus-card-bg-solid, rgba(0,0,0,0.02));
  border-radius: 6px;
}
.edit-row {
  display: grid;
  grid-template-columns: 200px 1fr 90px;
  align-items: center;
  gap: 12px;
}
.edit-label {
  font-size: 13px;
  color: var(--argus-text-muted);
  cursor: help;
}
.edit-label code {
  font-size: 11px;
  background: rgba(0,0,0,0.04);
  padding: 1px 4px;
  border-radius: 3px;
}
.edit-slider {
  margin: 0;
}
.edit-number {
  width: 90px;
}

/* ── Pair cards ── */
.empty-pairs {
  padding: 14px;
  background: var(--argus-card-bg-solid, rgba(0,0,0,0.02));
  border-radius: 6px;
  color: var(--argus-text-muted);
  font-size: 13px;
  text-align: center;
}
.pairs-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.pair-card {
  padding: 12px 14px;
  background: var(--argus-card-bg-solid, rgba(0,0,0,0.02));
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 6px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.pair-card--edit {
  border-color: rgba(22, 119, 255, 0.2);
}
.pair-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 500;
}
.pair-arrow {
  color: var(--argus-text-muted);
  font-size: 16px;
}

/* ── Pair edit header ── */
.pair-edit-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 12px;
}
.pair-cameras {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  flex: 1;
}
.camera-field {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.camera-label {
  font-size: 11px;
  color: var(--argus-text-muted);
}
.camera-input {
  width: 160px;
}

/* ── Matrix ── */
.matrix-label {
  font-size: 11px;
  color: var(--argus-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.matrix-edit-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.matrix-grid {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 12px;
}
.matrix-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2px;
}
.matrix-cell {
  padding: 4px 8px;
  background: rgba(0,0,0,0.03);
  border-radius: 3px;
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--argus-text);
}
.matrix-cell-input {
  width: 100%;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 12px;
}
</style>
