import { ref, onUnmounted, type Ref } from 'vue'
import { logger } from '../utils/logger'
import { StreamManager } from '../services/streaming/StreamManager'
import { PlayerFactory, type PlayerInstance } from '../services/streaming/PlayerFactory'

export type StreamStatus = 'idle' | 'connecting' | 'playing' | 'error' | 'fallback'

export function useGo2RTC(cameraId: Ref<string> | string) {
  const videoRef = ref<HTMLVideoElement | null>(null)
  const mjpegRef = ref<HTMLImageElement | null>(null)
  const status = ref<StreamStatus>('idle')

  let playerInstance: PlayerInstance | null = null
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
      status.value = 'error'
      return
    }
    _counted = true
    const thisGen = ++generation
    status.value = 'connecting'

    const isCancelled = () => thisGen !== generation

    playerInstance = await PlayerFactory.create(getCameraId(), videoRef.value, isCancelled)
    
    if (isCancelled()) return

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
  }

  onUnmounted(stop)

  return {
    videoRef,
    mjpegRef,
    status,
    start,
    stop,
  }
}
