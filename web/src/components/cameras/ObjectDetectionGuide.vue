<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'

const props = defineProps<{ modelValue: string; disabled?: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [value: string] }>()
const form = reactive({ labels: '', confidence: 0.5, duration: 0, cooldown: 30,
  fps: 5, gap: 1, device: 'cpu', scenePolicy: 'none', profileId: '',
  collectionEnabled: false, collectionPolicy: 'periodic', interval: 10, maxSamples: 1000, storageMb: 1024 })
const error = ref('')

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}
const parsed = computed(() => {
  try {
    const value: unknown = JSON.parse(props.modelValue)
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null
    return value as Record<string, unknown>
  } catch { return null }
})
function number(value: unknown, fallback: number): number { return typeof value === 'number' ? value : fallback }
watch(parsed, value => {
  if (!value) return
  const rules = record(value.rules), sampling = record(value.sampling)
  const resources = record(value.resources), scene = record(value.scene)
  form.labels = Array.isArray(rules.labels) ? rules.labels.filter((item): item is string => typeof item === 'string').join(', ') : ''
  form.confidence = number(rules.confidence_threshold, 0.5)
  form.duration = number(rules.min_duration_seconds, 0)
  form.cooldown = number(rules.cooldown_seconds, 30)
  form.fps = number(sampling.fps_target, 5)
  form.gap = number(sampling.max_gap_seconds, 1)
  form.device = resources.device === 'cuda' ? 'cuda' : 'cpu'
  form.scenePolicy = scene.policy === 'require_profile' ? 'require_profile' : 'none'
  form.profileId = typeof scene.profile_id === 'string' ? scene.profile_id : ''
  const collection = record(value.collection)
  form.collectionEnabled = collection.enabled === true
  form.collectionPolicy = typeof collection.policy === 'string' ? collection.policy : 'periodic'
  form.interval = number(collection.interval_seconds, 10)
  form.maxSamples = number(collection.max_samples, 1000)
  form.storageMb = number(collection.max_storage_mb, 1024)
  error.value = ''
}, { immediate: true })

function apply() {
  if (props.disabled || !parsed.value) return
  const labels = form.labels.split(/[,，]/).map(value => value.trim()).filter(Boolean)
  if (new Set(labels).size !== labels.length) { error.value = '类别不能重复'; return }
  const ranges: [number, number, number][] = [
    [form.confidence, 0, 1], [form.duration, 0, 3600], [form.cooldown, 0, 86400],
    [form.fps, 0.001, 120], [form.gap, 0.001, 30],
  ]
  if (ranges.some(([value, low, high]) => !Number.isFinite(value) || value < low || value > high)) {
    error.value = '请检查阈值、持续时间、冷却和采样范围'; return
  }
  if (form.scenePolicy === 'require_profile' && !form.profileId.trim()) {
    error.value = '场景绑定需要明确的已批准 profile 标识'; return
  }
  const value = parsed.value
  const next = { ...value,
    rules: { ...record(value.rules), labels, confidence_threshold: form.confidence,
      min_duration_seconds: form.duration, cooldown_seconds: form.cooldown },
    sampling: { mode: 'latest', ...record(value.sampling), fps_target: form.fps, max_gap_seconds: form.gap },
    resources: { ...record(value.resources), device: form.device },
    scene: { ...record(value.scene), policy: form.scenePolicy,
      profile_id: form.scenePolicy === 'require_profile' ? form.profileId.trim() : null },
  }
  error.value = ''
  emit('update:modelValue', JSON.stringify(next, null, 2))
}
function applyCollection() {
  if (props.disabled || !parsed.value) return
  if (!['periodic', 'events'].includes(form.collectionPolicy)) {
    error.value = '此采集策略尚未实现，不会自动替换为定时采集'; return
  }
  if (!Number.isFinite(form.interval) || form.interval < 0.1 || form.interval > 86400
    || !Number.isInteger(form.maxSamples) || form.maxSamples < 1 || form.maxSamples > 100000
    || !Number.isInteger(form.storageMb) || form.storageMb < 1 || form.storageMb > 102400) {
    error.value = '请检查采集间隔、样本上限和磁盘预算'; return
  }
  error.value = ''
  emit('update:modelValue', JSON.stringify({ ...parsed.value, collection: {
    ...record(parsed.value.collection), enabled: form.collectionEnabled, policy: form.collectionPolicy,
    interval_seconds: form.interval, max_samples: form.maxSamples, max_storage_mb: form.storageMb,
  } }, null, 2))
}
</script>

<template>
  <section aria-label="通用目标检测配置引导" class="object-guide">
    <h4>目标检测配置引导</h4>
    <p class="note">先配置任务，再选择兼容模型、冻结执行包并旁路验证。空类别表示使用模型全部类别；选择类别必须与模型标签一致。</p>
    <p v-if="!parsed" role="alert">高级 JSON 无效；请先修正 JSON。引导不会覆盖无效内容或丢弃草稿。</p>
    <p v-if="error" role="alert">{{ error }}</p>
    <fieldset :disabled="disabled || !parsed">
      <div class="grid">
        <label>目标类别（逗号分隔）<input v-model="form.labels" data-test="guide-labels" placeholder="person, package" /></label>
        <label>置信度阈值<input v-model.number="form.confidence" type="number" min="0" max="1" step="0.01" data-test="guide-confidence" /></label>
        <label>连续确认时长（秒）<input v-model.number="form.duration" type="number" min="0" max="3600" step="0.1" /></label>
        <label>重复事件冷却（秒）<input v-model.number="form.cooldown" type="number" min="0" max="86400" /></label>
        <label>目标采样率（FPS）<input v-model.number="form.fps" type="number" min="0.001" max="120" step="0.1" /></label>
        <label>最大帧间隔（秒）<input v-model.number="form.gap" type="number" min="0.001" max="30" step="0.1" /></label>
        <label>计算设备<select v-model="form.device" data-test="guide-device"><option value="cpu">CPU（显式选择）</option><option value="cuda">CUDA（不可静默降级）</option></select></label>
        <label>场景约束<select v-model="form.scenePolicy"><option value="none">通用全画面（无场景绑定）</option><option value="require_profile">要求已批准场景 profile</option></select></label>
        <label v-if="form.scenePolicy === 'require_profile'">已批准 profile 标识<input v-model="form.profileId" maxlength="128" /></label>
      </div>
      <p class="note">区域坐标等高级配置在下方 JSON 中保留；填写 profile 标识不会批准场景，也不会放宽 SceneGuard 或发布门。</p>
      <button type="button" data-test="apply-object-guide" @click="apply">应用引导到配置草稿</button>
      <h4>验证数据采集策略</h4>
      <div class="grid">
        <label>随任务自动采集<input v-model="form.collectionEnabled" type="checkbox" data-test="guide-collection-enabled" /></label>
        <label>采集策略<select v-model="form.collectionPolicy" data-test="guide-collection-policy"><option value="periodic">定时抽样</option><option value="events">已提交事件的精确关键帧</option><option value="disagreement" disabled>模型分歧（暂未实现）</option><option value="low_confidence" disabled>低置信度（暂未实现）</option></select></label>
        <label>最小采集间隔（秒）<input v-model.number="form.interval" type="number" min="0.1" max="86400" step="0.1" data-test="guide-collection-interval" /></label>
        <label>样本上限<input v-model.number="form.maxSamples" type="number" min="1" max="100000" data-test="guide-collection-limit" /></label>
        <label>任务图片总预算（MB）<input v-model.number="form.storageMb" type="number" min="1" max="102400" /></label>
      </div>
      <p class="note">事件策略只收集本次已应用执行会话中的新事件关键帧，不采集事件前后录像，也不补抓未来帧。源或执行身份变化时停止；没有样本不代表现场正常。样本仍需独立人工复核，策略保存不会启动摄像头、训练或发布。</p>
      <button type="button" data-test="apply-collection-guide" @click="applyCollection">应用采集策略到配置草稿</button>
    </fieldset>
  </section>
</template>

<style scoped>
.object-guide { margin: 12px 0; padding: 12px; border: 1px solid var(--border-default); border-radius: var(--r-md); }
h4 { margin: 0; } .note { color: var(--text-secondary); font-size: 12px; line-height: 1.6; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 8px 16px; }
fieldset { padding: 0; margin: 0; border: 0; } label { display: grid; gap: 5px; font-size: 12px; }
input, select, button { padding: 8px; color: var(--text-primary); background: var(--bg); border: 1px solid var(--border-default); border-radius: 6px; }
button { cursor: pointer; } button:disabled { opacity: .6; cursor: not-allowed; }
:is(input, select, button):focus-visible { outline: 2px solid var(--status-info); outline-offset: 2px; }
[role="alert"] { color: var(--status-danger); }
</style>
