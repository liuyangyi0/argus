<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { message } from 'ant-design-vue'
import { getBaselineCollections } from '../../api/baselines'
import type { BaselineCollection, DatasetSelection } from '../../types/api'

const props = defineProps<{
  /** Optional camera filter — when set, only that camera's collections are shown. */
  cameraId?: string | null
  modelValue?: DatasetSelection | null
}>()
const emit = defineEmits<{
  (e: 'update:modelValue', value: DatasetSelection | null): void
}>()

const all = ref<BaselineCollection[]>([])
const loading = ref(false)
// selection key = `${camera_id}::${zone_id}::${version}`
const selectedKeys = ref<Set<string>>(new Set())

async function load() {
  loading.value = true
  try {
    const res = await getBaselineCollections()
    all.value = res?.collections ?? []
  } catch (e: any) {
    message.error(`加载采集集合失败: ${e?.message || e}`)
    all.value = []
  } finally {
    loading.value = false
  }
}

onMounted(load)

const filtered = computed<BaselineCollection[]>(() => {
  return all.value
    .filter((c) => c.status !== 'failed') // failed runs cannot be used for training
    .filter((c) => !props.cameraId || c.camera_id === props.cameraId)
})

const totalFrames = computed(() => {
  let sum = 0
  for (const c of filtered.value) {
    if (selectedKeys.value.has(itemKey(c))) sum += c.image_count || 0
  }
  return sum
})

function itemKey(c: BaselineCollection): string {
  return `${c.camera_id}::${c.zone_id}::${c.version}`
}

function toggle(c: BaselineCollection) {
  const k = itemKey(c)
  const next = new Set(selectedKeys.value)
  if (next.has(k)) next.delete(k)
  else next.add(k)
  selectedKeys.value = next
}

function isSelected(c: BaselineCollection) {
  return selectedKeys.value.has(itemKey(c))
}

watch([selectedKeys, filtered], () => {
  if (selectedKeys.value.size === 0) {
    emit('update:modelValue', null)
    return
  }
  const items = filtered.value
    .filter((c) => selectedKeys.value.has(itemKey(c)))
    .map((c) => ({
      camera_id: c.camera_id,
      zone_id: c.zone_id,
      version: c.version,
      session_label: c.session_label || undefined,
    }))
  emit('update:modelValue', { items, total_frames: totalFrames.value })
}, { deep: true })

// When cameraId changes, drop selections that no longer apply.
watch(() => props.cameraId, (cid) => {
  if (!cid) return
  const next = new Set<string>()
  for (const k of selectedKeys.value) {
    const [c] = k.split('::')
    if (c === cid) next.add(k)
  }
  selectedKeys.value = next
})

const groups = computed(() => {
  const out = new Map<string, BaselineCollection[]>()
  for (const c of filtered.value) {
    const g = `${c.camera_id} / ${c.zone_id}`
    if (!out.has(g)) out.set(g, [])
    out.get(g)!.push(c)
  }
  return Array.from(out.entries())
})
</script>

<template>
  <div class="dataset-selector">
    <div class="header">
      <span class="title">数据集（多选合并训练）</span>
      <span class="hint">已选 {{ selectedKeys.size }} 项 · 合并 {{ totalFrames }} 帧</span>
    </div>

    <a-spin :spinning="loading">
      <div v-if="filtered.length === 0 && !loading" class="empty">
        {{ props.cameraId ? '当前摄像头没有可用的采集版本' : '暂无可用的采集版本' }}
      </div>

      <div v-for="[label, items] in groups" :key="label" class="group">
        <div class="group-label">{{ label }}</div>
        <div
          v-for="c in items"
          :key="`${c.version}-${c.session_label || ''}`"
          class="row"
          :class="{ selected: isSelected(c) }"
          @click="toggle(c)"
        >
          <a-checkbox :checked="isSelected(c)" @click.stop="toggle(c)" />
          <div class="row-body">
            <div class="row-line1">
              <span class="version">{{ c.version }}</span>
              <a-tag v-if="c.is_current" color="blue">当前</a-tag>
              <a-tag v-if="c.session_label">{{ c.session_label }}</a-tag>
              <a-tag v-if="c.state">{{ c.state }}</a-tag>
            </div>
            <div class="row-line2">
              <span>{{ c.image_count }} 帧</span>
              <span v-if="c.acceptance_rate != null">
                · 接受率 {{ (c.acceptance_rate * 100).toFixed(1) }}%
              </span>
              <span v-if="c.captured_at">· {{ c.captured_at.replace('T', ' ').slice(0, 19) }}</span>
            </div>
          </div>
        </div>
      </div>
    </a-spin>
  </div>
</template>

<style scoped>
.dataset-selector {
  border: 1px solid var(--line-2, #e5e6eb);
  border-radius: 6px;
  padding: 12px;
  background: #fafafa;
}
.header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 10px;
}
.title {
  font-weight: 600;
  font-size: 13px;
}
.hint {
  font-size: 12px;
  color: var(--ink-3, #595959);
}
.empty {
  padding: 20px;
  text-align: center;
  color: var(--ink-3, #8890a0);
  font-size: 12px;
}
.group + .group {
  margin-top: 12px;
}
.group-label {
  font-size: 11px;
  color: var(--ink-3, #595959);
  letter-spacing: 0.3px;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 4px;
  cursor: pointer;
  border: 1px solid transparent;
  background: #fff;
}
.row + .row {
  margin-top: 4px;
}
.row:hover {
  border-color: #1890ff66;
}
.row.selected {
  border-color: #1890ff;
  background: #e6f4ff;
}
.row-body {
  flex: 1;
  min-width: 0;
}
.row-line1 {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
}
.version {
  font-weight: 600;
}
.row-line2 {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  font-size: 11px;
  color: var(--ink-3, #595959);
  margin-top: 2px;
}
</style>
