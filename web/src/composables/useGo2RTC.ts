import { ref, onUnmounted, type Ref } from 'vue'
import { logger } from '../utils/logger'
import { StreamManager } from '../services/streaming/StreamManager'
import { PlayerFactory, type PlayerInstance } from '../services/streaming/PlayerFactory'

export type StreamStatus = 'idle' | 'connecting' | 'playing' | 'error' | 'fallback'

const FALLBACK_REASON_LABELS: Record<string, string> = {
  connection_budget_exhausted: '实时连接数已满',
  stream_info_unavailable: '流媒体信息不可用',
  webrtc_failed: 'WebRTC 连接失败',
  mse_failed: 'MSE 连接失败',
  go2rtc_unavailable: 'go2rtc 不可用',
  video_element_unavailable: '视频播放器尚未就绪',
  player_creation_exception: '播放器初始化失败',
}

export function streamFallbackReasonLabel(reason: string | null | undefined): string {
  if (!reason) return '实时流不可用'
  return FALLBACK_REASON_LABELS[reason] ?? `实时流降级（${reason}）`
}

export function useGo2RTC(cameraId: Ref<string> | string) {
  const videoRef = ref<HTMLVideoElement | null>(null)
  const mjpegRef = ref<HTMLImageElement | null>(null)
  const status = ref<StreamStatus>('idle')
  const fallbackReason = ref<string | null>(null)

  let playerInstance: PlayerInstance | null = null
  let startController: AbortController | null = null
  let generation = 0
  let _counted = false

  const getCameraId = () => typeof cameraId === 'string' ? cameraId : cameraId.value

  function releaseBudget() {
    if (_counted) {
      StreamManager.releaseSlot()
      _counted = false
    }
  }

  async function start() {
    stop() // internally increments generation

    // Request budget for WebRTC/MSE connection
    if (!StreamManager.requestSlot()) {
      logger.warn(`[go2rtc] connection budget exhausted limit(${StreamManager.MAX_STREAMS})`)
      fallbackReason.value = 'connection_budget_exhausted'
      status.value = 'fallback'
      return
    }
    _counted = true
    const thisGen = ++generation
    status.value = 'connecting'
    fallbackReason.value = null

    const isCancelled = () => thisGen !== generation
    const controller = new AbortController()
    startController = controller

    let candidate: PlayerInstance | null = null
    try {
      candidate = await PlayerFactory.create(
        getCameraId(),
        videoRef.value,
        isCancelled,
        reason => { if (!isCancelled()) fallbackReason.value = reason },
        controller.signal,
      )
    } catch (e) {
      if (isCancelled()) return
      logger.warn(`[go2rtc] player creation failed for ${getCameraId()}, falling back to MJPEG`, e)
      fallbackReason.value = 'player_creation_exception'
    }

    if (isCancelled()) {
      candidate?.destroy()
      return
    }
    // An older asynchronous start must never clear or replace a newer owner.
    playerInstance = candidate

    if (playerInstance) {
      status.value = playerInstance.status
      playerInstance.onStatusChange = (newStatus) => {
        if (isCancelled()) return
        status.value = newStatus
      }
      playerInstance.onReconnectRequest = () => {
        if (!isCancelled()) {
          start()
        }
      }
    } else {
      // Fallback
      releaseBudget()
      status.value = 'fallback'
    }
  }

  function stop() {
    generation++
    // Cancel pending connection creation before freeing its connection budget.
    // A cancelled factory cannot attach a late media stream to the next start.
    startController?.abort()
    startController = null
    if (playerInstance) {
      playerInstance.destroy()
      playerInstance = null
    }
    if (videoRef.value) {
      videoRef.value.srcObject = null
      videoRef.value.src = ''
    }
    if (mjpegRef.value) {
      mjpegRef.value.src = ''
    }
    releaseBudget()
    status.value = 'idle'
    fallbackReason.value = null
  }

  onUnmounted(stop)

  return {
    videoRef,
    mjpegRef,
    status,
    fallbackReason,
    start,
    stop,
  }
}
