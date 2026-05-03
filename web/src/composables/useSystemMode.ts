import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { getSystemModeSummary } from '../api/cameras'
import type { PipelineMode, SystemModeSummary } from '../types/api'

const COLOR_MAP: Record<PipelineMode, string> = {
  active: '#52c41a',
  maintenance: '#faad14',
  learning: '#1890ff',
  collection: '#722ed1',
  training: '#fa8c16',
}

const LABEL_MAP: Record<PipelineMode, string> = {
  active: '运行中',
  maintenance: '维护',
  learning: '学习中',
  collection: '采集中',
  training: '训练中',
}

const GLOBAL_BANNER_LABEL: Record<SystemModeSummary['global_state'], string> = {
  normal: '运行中',
  capturing: '采集中（检测已暂停）',
  training: '训练中（GPU 占用，检测已暂停）',
  maintenance: '维护模式',
}

const GLOBAL_BANNER_COLOR: Record<SystemModeSummary['global_state'], string> = {
  normal: '#52c41a',
  capturing: '#722ed1',
  training: '#fa8c16',
  maintenance: '#faad14',
}

/** Lightweight polling composable used by the navbar banner and any
 *  component that wants pipeline_mode info per camera. */
export function useSystemMode(intervalMs = 3000) {
  const cameras = ref<Record<string, PipelineMode>>({})
  const globalState = ref<SystemModeSummary['global_state']>('normal')
  let timer: ReturnType<typeof setInterval> | null = null

  async function refresh() {
    try {
      const data = await getSystemModeSummary()
      cameras.value = (data?.cameras ?? {}) as Record<string, PipelineMode>
      globalState.value = data?.global_state ?? 'normal'
    } catch {
      // Network blip: keep the last good snapshot rather than dropping to default.
    }
  }

  onMounted(() => {
    refresh()
    timer = setInterval(refresh, intervalMs)
  })
  onBeforeUnmount(() => {
    if (timer !== null) clearInterval(timer)
  })

  const showBanner = computed(() => globalState.value !== 'normal')
  const bannerLabel = computed(() => GLOBAL_BANNER_LABEL[globalState.value])
  const bannerColor = computed(() => GLOBAL_BANNER_COLOR[globalState.value])

  return {
    cameras,
    globalState,
    showBanner,
    bannerLabel,
    bannerColor,
    refresh,
  }
}

/** Per-mode colour + label lookup used by ModeBadge.vue. */
export function modeBadge(mode?: PipelineMode | string | null) {
  if (!mode || !(mode in COLOR_MAP)) {
    return { color: '#bfbfbf', label: '未知', visible: false }
  }
  const m = mode as PipelineMode
  return {
    color: COLOR_MAP[m],
    label: LABEL_MAP[m],
    // Hide the badge for the boring "active" case to keep video walls clean
    visible: m !== 'active',
  }
}
