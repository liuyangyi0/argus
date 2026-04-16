import { ref, computed, watch } from 'vue'
import { message } from 'ant-design-vue'
import {
  getReplayMetadata,
  getReplaySignals,
  getReplayVideoUrl,
  getReplayReference,
  pinReplayFrame,
} from '../api'

export function useReplayController(alertId: string) {
  // core state
  const metadata = ref<any>(null)
  const signals = ref<any>(null)
  const currentIndex = ref(0)
  const playing = ref(false)
  const speed = ref(1)
  const loading = ref(true)

  // video dom ref
  const videoEl = ref<HTMLVideoElement | null>(null)
  const videoError = ref('')
  const pendingSeekIndex = ref<number | null>(null)

  // overlays
  const showHeatmap = ref(true)
  const showBoxes = ref(false)
  const showTrajectory = ref(false)
  const showHud = ref(true)
  const heatmapOpacity = ref(0.4)

  // reference frames
  const referenceFrame = ref<string | null>(null)
  const referenceDate = ref('')
  const loadingRef = ref(false)
  const selectedRefOption = ref('yesterday')

  // clips
  const clipStart = ref<number | null>(null)
  const clipEnd = ref<number | null>(null)

  // computed properties
  const fps = computed(() => metadata.value?.fps || 15)
  const videoUrl = computed(() => getReplayVideoUrl(alertId))
  const hasHeatmaps = computed(() => signals.value?.has_heatmaps || false)

  const currentTimestamp = computed(() => {
    const ts = signals.value?.timestamps?.[currentIndex.value]
    if (!ts) return ''
    return new Date(ts * 1000).toLocaleTimeString('zh-CN')
  })

  // data loading
  async function loadData() {
    loading.value = true
    try {
      const [metaRes, sigRes] = await Promise.all([
        getReplayMetadata(alertId),
        getReplaySignals(alertId),
      ])
      metadata.value = metaRes
      signals.value = sigRes

      if (metadata.value?.trigger_frame_index !== undefined) {
        pendingSeekIndex.value = metadata.value.trigger_frame_index
        currentIndex.value = metadata.value.trigger_frame_index
      }
      loadReference()
    } catch (e) {
      message.error('回放数据加载失败')
      metadata.value = null
    } finally {
      loading.value = false
    }
  }

  async function loadReference(date?: string) {
    loadingRef.value = true
    try {
      const params: any = {}
      if (date) params.date = date
      const res = await getReplayReference(alertId, params)
      if (res.available && res.frame_base64) {
        referenceFrame.value = `data:image/jpeg;base64,${res.frame_base64}`
      } else {
        referenceFrame.value = null
      }
      referenceDate.value = res.source_date || ''
    } catch {
      referenceFrame.value = null
    } finally {
      loadingRef.value = false
    }
  }

  // reference option change
  function onRefOptionChange(value: any) {
    selectedRefOption.value = value
    if (value === 'custom') {
      const dateStr = prompt('输入日期 (YYYY-MM-DD):')
      if (dateStr && /^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
        loadReference(dateStr)
      }
      return
    }
    const triggerTs = metadata.value?.trigger_timestamp
    if (!triggerTs) return
    const trigger = new Date(triggerTs * 1000)
    let refDate = new Date(trigger)
    if (value === 'yesterday') {
      refDate.setDate(refDate.getDate() - 1)
    } else if (value === 'last_week') {
      refDate.setDate(refDate.getDate() - 7)
    } else if (value === 'prev_week') {
      refDate.setDate(refDate.getDate() - 14)
    } else {
      refDate.setDate(refDate.getDate() - 1)
    }
    const dateStr = refDate.toISOString().slice(0, 10)
    loadReference(dateStr)
  }

  // playback controls
  function safePlay() {
    if (!videoEl.value) return
    videoEl.value.play().catch((err: DOMException) => {
      if (err.name === 'AbortError') return
      videoError.value = videoErrorMessage(err)
      playing.value = false
    })
  }

  function togglePlay() {
    if (!videoEl.value) return
    if (videoEl.value.paused) {
      safePlay()
    } else {
      videoEl.value.pause()
    }
  }

  function stepFrame(delta: number) {
    if (!videoEl.value) return
    videoEl.value.pause()
    videoEl.value.currentTime = Math.max(0, videoEl.value.currentTime + delta / fps.value)
  }

  function seekTo(index: number) {
    if (!videoEl.value) return
    videoEl.value.pause()
    videoEl.value.currentTime = index / fps.value
    currentIndex.value = index
  }

  function goToStart() { seekTo(0) }
  function goToEnd() { seekTo((metadata.value?.frame_count || 1) - 1) }

  async function handlePinFrame() {
    const label = prompt('帧标签:')
    if (!label) return
    try {
      await pinReplayFrame(alertId, { index: currentIndex.value, label })
      if (signals.value) {
        if (!signals.value.key_frames) signals.value.key_frames = []
        const idx = currentIndex.value
        if (!signals.value.key_frames.some((kf: any) => kf.index === idx)) {
          signals.value.key_frames.push({ index: idx, label, type: 'user' })
        }
      }
      message.success('已标记帧')
    } catch {
      message.error('标记失败')
    }
  }

  function videoErrorMessage(err: MediaError | DOMException): string {
    if (err instanceof MediaError) {
      const msgs: Record<number, string> = { 1: '视频加载被中止', 2: '网络错误', 3: '视频解码失败', 4: '视频格式不支持' }
      return msgs[err.code] || `视频错误 (code=${err.code})`
    }
    if (err instanceof DOMException) {
      if (err.name === 'NotAllowedError') return '浏览器阻止自动播放，请点击视频区域后重试'
      if (err.name === 'NotSupportedError') return '视频格式不支持'
      return `播放失败: ${err.message || err.name}`
    }
    return `播放失败`
  }

  watch(speed, (s) => {
    if (videoEl.value) {
      videoEl.value.playbackRate = s
    }
  })

  // Shortcuts logic (exposed so component can addEventListener on mount)
  function handleKeydown(e: KeyboardEvent) {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
    switch (e.key) {
      case ' ': e.preventDefault(); togglePlay(); break
      case 'ArrowLeft': stepFrame(-1); break
      case 'ArrowRight': stepFrame(1); break
      case 'k': case 'K': if (videoEl.value) videoEl.value.pause(); break
      case 'j': case 'J': speed.value = Math.max(0.25, speed.value / 2); if (videoEl.value?.paused) safePlay(); break
      case 'l': case 'L': speed.value = Math.min(4, speed.value * 2); if (videoEl.value?.paused) safePlay(); break
      case 'Home': goToStart(); break
      case 'End': goToEnd(); break
      case '[': clipStart.value = currentIndex.value; break
      case ']': clipEnd.value = currentIndex.value; break
    }
  }

  return {
    metadata, signals, currentIndex, playing, speed, loading,
    videoEl, videoError, pendingSeekIndex,
    showHeatmap, showBoxes, showTrajectory, showHud, heatmapOpacity,
    referenceFrame, referenceDate, loadingRef, selectedRefOption,
    clipStart, clipEnd,
    fps, videoUrl, hasHeatmaps, currentTimestamp,
    loadData, togglePlay, stepFrame, seekTo, goToStart, goToEnd,
    handlePinFrame, handleKeydown, onRefOptionChange
  }
}
