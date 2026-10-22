import axios from 'axios'
import { logger } from '../../utils/logger'
import { STREAM_CONNECT_TIMEOUT_MS } from '../../config/constants'

export interface StreamInfo {
  camera_id: string
  go2rtc: boolean
  webrtc_ws?: string
  mse_ws?: string
  fallback: string
}

export interface PlayerInstance {
  status: 'idle' | 'connecting' | 'playing' | 'error' | 'fallback'
  onStatusChange: (status: 'idle' | 'connecting' | 'playing' | 'error' | 'fallback') => void
  onReconnectRequest: () => void
  connect(): Promise<boolean>
  destroy(): void
}

function isStreamInfo(value: unknown): value is StreamInfo {
  if (!value || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return (
    typeof record.camera_id === 'string'
    && typeof record.go2rtc === 'boolean'
    && typeof record.fallback === 'string'
    && (record.webrtc_ws === undefined || typeof record.webrtc_ws === 'string')
    && (record.mse_ws === undefined || typeof record.mse_ws === 'string')
  )
}

export class WebRTCPlayer implements PlayerInstance {
  status: 'idle' | 'connecting' | 'playing' | 'error' | 'fallback' = 'connecting'
  onStatusChange: (status: 'idle' | 'connecting' | 'playing' | 'error' | 'fallback') => void = () => {}
  onReconnectRequest: () => void = () => {}

  private pc: RTCPeerConnection | null = null
  private ws: WebSocket | null = null
  private settled = false
  private cameraId: string
  private wsUrl: string
  private videoEl: HTMLVideoElement
  private ownedStream: MediaStream | null = null
  private cancelConnection: (() => void) | null = null

  constructor(cameraId: string, wsUrl: string, videoEl: HTMLVideoElement) {
    this.cameraId = cameraId
    this.wsUrl = wsUrl
    this.videoEl = videoEl
  }

  private setStatus(s: typeof this.status) {
    this.status = s
    this.onStatusChange(s)
  }

  async connect(): Promise<boolean> {
    return new Promise((resolve) => {
      this.settled = false
      const settle = (ok: boolean) => { if (!this.settled) { this.settled = true; clearTimeout(timeout); resolve(ok) } }

      const timeout = setTimeout(() => {
        logger.debug(`[go2rtc] WebRTC timeout for ${this.cameraId}`)
        this.cleanup()
        settle(false)
      }, STREAM_CONNECT_TIMEOUT_MS)
      this.cancelConnection = () => {
        this.cleanup()
        settle(false)
      }
      const failConnection = () => {
        const wasPlaying = this.settled && this.status === 'playing'
        this.cleanup()
        if (!this.settled) settle(false)
        else if (wasPlaying) {
          // The initial connect promise is already resolved. Clearing media
          // alone would leave the owner reporting "playing" indefinitely.
          this.setStatus('error')
          this.onReconnectRequest()
        }
      }

      try {
        this.ws = new WebSocket(this.wsUrl)
        const ws = this.ws
        this.ws.onopen = () => {
          if (this.ws !== ws) return
          try {
            const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] })
            this.pc = pc
            pc.ontrack = (event) => {
              if (this.pc !== pc || this.ws !== ws) return
              if (this.videoEl && event.streams[0]) {
                this.ownedStream = event.streams[0]
                this.videoEl.srcObject = this.ownedStream
                this.setStatus('playing')
                settle(true)
              }
            }
            pc.onconnectionstatechange = () => {
              if (this.pc !== pc || this.ws !== ws) return
              if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
                this.onReconnectRequest()
              }
            }
            pc.onicecandidate = (event) => {
              if (this.pc !== pc || this.ws !== ws) return
              if (event.candidate && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'webrtc/candidate', value: event.candidate.candidate }))
              }
            }
            pc.addTransceiver('video', { direction: 'recvonly' })
            pc.addTransceiver('audio', { direction: 'recvonly' })
            void pc.createOffer().then(async (offer) => {
              if (this.pc !== pc || this.ws !== ws) return
              await pc.setLocalDescription(offer)
              if (this.pc !== pc || this.ws !== ws) return
              ws?.send(JSON.stringify({ type: 'webrtc/offer', value: offer.sdp }))
            }).catch(() => {
              if (this.pc !== pc || this.ws !== ws) return
              failConnection()
            })
          } catch {
            if (this.ws === ws) failConnection()
          }
        }

        this.ws.onmessage = (event) => {
          if (this.ws !== ws) return
          const pc = this.pc
          try {
            const msg: unknown = JSON.parse(event.data)
            if (!msg || typeof msg !== 'object') throw new Error('Invalid WebRTC message')
            const payload = msg as { type?: unknown; value?: unknown }
            if (payload.type === 'webrtc/answer' && pc) {
              if (typeof payload.value !== 'string') throw new Error('Invalid WebRTC answer')
              void pc.setRemoteDescription(new RTCSessionDescription({ type: 'answer', sdp: payload.value }))
                .catch(() => { if (this.pc === pc && this.ws === ws) failConnection() })
            } else if (payload.type === 'webrtc/candidate' && pc) {
              if (typeof payload.value !== 'string') throw new Error('Invalid WebRTC candidate')
              void pc.addIceCandidate(new RTCIceCandidate({ candidate: payload.value, sdpMid: '0' }))
                .catch(() => { if (this.pc === pc && this.ws === ws) failConnection() })
            }
          } catch {
            if (this.ws === ws) failConnection()
          }
        }

        this.ws.onerror = () => {
          if (this.ws === ws) failConnection()
        }

        this.ws.onclose = () => {
          if (this.ws !== ws) return
          if (!this.settled) {
            this.cleanup()
            settle(false)
          }
        }
      } catch {
        settle(false)
      }
    })
  }

  private cleanup() {
    if (this.pc) {
      this.pc.onconnectionstatechange = null
      this.pc.ontrack = null
      this.pc.onicecandidate = null
      this.pc.close()
      this.pc = null
    }
    if (this.ws) {
      this.ws.onopen = null
      this.ws.onclose = null
      this.ws.onerror = null
      this.ws.onmessage = null
      this.ws.close()
      this.ws = null
    }
    const stream = this.ownedStream
    if (stream) {
      for (const track of stream.getTracks()) track.stop()
      if (this.videoEl.srcObject === stream) this.videoEl.srcObject = null
      this.ownedStream = null
    }
  }

  destroy() {
    this.cancelConnection?.()
    this.cancelConnection = null
    this.cleanup()
  }
}

export class MSEPlayer implements PlayerInstance {
  status: 'idle' | 'connecting' | 'playing' | 'error' | 'fallback' = 'connecting'
  onStatusChange: (status: 'idle' | 'connecting' | 'playing' | 'error' | 'fallback') => void = () => {}
  onReconnectRequest: () => void = () => {}

  private mseWs: WebSocket | null = null
  private mediaSource: MediaSource | null = null
  private sourceBuffer: SourceBuffer | null = null
  private sourceOpenHandler: (() => void) | null = null
  private updateEndHandler: (() => void) | null = null
  private objectUrl: string | null = null
  private bufferQueue: ArrayBuffer[] = []
  private settled = false
  private MAX_MSE_BUFFER_QUEUE = 30
  private cameraId: string
  private wsUrl: string
  private videoEl: HTMLVideoElement
  private cancelConnection: (() => void) | null = null

  constructor(cameraId: string, wsUrl: string, videoEl: HTMLVideoElement) {
    this.cameraId = cameraId
    this.wsUrl = wsUrl
    this.videoEl = videoEl
  }

  private setStatus(s: typeof this.status) {
    this.status = s
    this.onStatusChange(s)
  }

  async connect(): Promise<boolean> {
    if (!('MediaSource' in window)) return false

    return new Promise((resolve) => {
      this.settled = false
      const settle = (ok: boolean) => { if (!this.settled) { this.settled = true; clearTimeout(timeout); resolve(ok) } }

      const timeout = setTimeout(() => {
        logger.warn(`[go2rtc] MSE timeout for ${this.cameraId}`)
        this.cleanup()
        settle(false)
      }, STREAM_CONNECT_TIMEOUT_MS)
      this.cancelConnection = () => {
        this.cleanup()
        settle(false)
      }

      try {
        if (!this.videoEl) {
          settle(false)
          return
        }
        this.mediaSource = new MediaSource()
        this.videoEl.srcObject = null
        this.objectUrl = URL.createObjectURL(this.mediaSource)
        this.videoEl.src = this.objectUrl

        this.sourceOpenHandler = () => {
          this.mseWs = new WebSocket(this.wsUrl)
          this.mseWs.binaryType = 'arraybuffer'

          this.mseWs.onmessage = (event) => {
            if (typeof event.data === 'string') {
              const msg = JSON.parse(event.data)
              if (msg.type === 'mse' && msg.value && this.mediaSource) {
                try {
                  this.sourceBuffer = this.mediaSource.addSourceBuffer(msg.value)
                  this.sourceBuffer.mode = 'segments'
                  this.updateEndHandler = () => {
                    if (
                      this.bufferQueue.length > 0
                      && this.sourceBuffer
                      && !this.sourceBuffer.updating
                    ) {
                      this.sourceBuffer.appendBuffer(this.bufferQueue.shift()!)
                    }
                  }
                  this.sourceBuffer.addEventListener('updateend', this.updateEndHandler)
                } catch {
                  this.cleanup()
                  settle(false)
                }
              }
            } else if (event.data instanceof ArrayBuffer && this.sourceBuffer) {
              if (!this.settled) {
                this.setStatus('playing')
                settle(true)
              }
              if (this.bufferQueue.length >= this.MAX_MSE_BUFFER_QUEUE) {
                this.bufferQueue.splice(
                  0,
                  this.bufferQueue.length - this.MAX_MSE_BUFFER_QUEUE + 1,
                )
              }
              if (this.sourceBuffer.updating) {
                this.bufferQueue.push(event.data)
              } else {
                this.sourceBuffer.appendBuffer(event.data)
              }
            }
          }

          this.mseWs.onerror = () => {
            this.cleanup()
            if (!this.settled) settle(false)
            else this.onReconnectRequest()
          }

          this.mseWs.onclose = () => {
            if (!this.settled) {
              this.cleanup()
              settle(false)
            } else if (this.status === 'playing') {
              this.onReconnectRequest()
            }
          }
        }
        this.mediaSource.addEventListener('sourceopen', this.sourceOpenHandler)
      } catch {
        settle(false)
      }
    })
  }

  private cleanup() {
    if (this.mseWs) {
      this.mseWs.onopen = null
      this.mseWs.onclose = null
      this.mseWs.onerror = null
      this.mseWs.onmessage = null
      this.mseWs.close()
      this.mseWs = null
    }
    this.bufferQueue.length = 0
    if (this.sourceBuffer && this.updateEndHandler) {
      this.sourceBuffer.removeEventListener('updateend', this.updateEndHandler)
    }
    if (this.mediaSource && this.sourceOpenHandler) {
      this.mediaSource.removeEventListener('sourceopen', this.sourceOpenHandler)
    }
    if (
      this.mediaSource
      && this.mediaSource.readyState === 'open'
      && this.sourceBuffer
    ) {
      try { this.mediaSource.removeSourceBuffer(this.sourceBuffer) } catch { /* ignore */ }
    }
    this.sourceBuffer = null
    this.updateEndHandler = null
    this.sourceOpenHandler = null
    if (this.mediaSource && this.mediaSource.readyState === 'open') {
      try { this.mediaSource.endOfStream() } catch { /* ignore */ }
    }
    this.mediaSource = null
    // A newer WebRTC/MSE player may already own this shared element. Revoke
    // our URL regardless, but detach only media that still belongs to us.
    if (
      this.objectUrl
      && this.videoEl.getAttribute('src') === this.objectUrl
      && this.videoEl.srcObject == null
    ) {
      this.videoEl.removeAttribute('src')
      if (typeof this.videoEl.load === 'function') this.videoEl.load()
    }
    if (this.objectUrl) {
      URL.revokeObjectURL(this.objectUrl)
      this.objectUrl = null
    }
  }

  destroy() {
    this.cancelConnection?.()
    this.cancelConnection = null
    this.cleanup()
  }
}

export class PlayerFactory {
  static async fetchStreamInfo(cameraId: string, signal?: AbortSignal): Promise<StreamInfo | null> {
    try {
      const path = `/api/streaming/${cameraId}`
      const resp = signal ? await axios.get(path, { signal }) : await axios.get(path)
      const payload = resp.data
      if (
        payload &&
        typeof payload === 'object' &&
        'code' in payload &&
        'data' in payload
      ) {
        if (payload.code !== 0) return null
        return isStreamInfo(payload.data) ? payload.data : null
      }
      return isStreamInfo(payload) ? payload : null
    } catch (e) {
      logger.debug(`[go2rtc] fetchStreamInfo failed for ${cameraId}:`, e)
      return null
    }
  }

  private static async connectPlayer(player: PlayerInstance, signal?: AbortSignal): Promise<boolean> {
    if (signal?.aborted) {
      player.destroy()
      return false
    }
    const abort = () => player.destroy()
    signal?.addEventListener('abort', abort, { once: true })
    try {
      return await player.connect()
    } catch (error) {
      player.destroy()
      throw error
    } finally {
      signal?.removeEventListener('abort', abort)
    }
  }

  static async create(
    cameraId: string,
    videoEl: HTMLVideoElement | null,
    isCancelled: () => boolean,
    onFailureReason?: (reason: string) => void,
    signal?: AbortSignal,
  ): Promise<PlayerInstance | null> {
    const cancelled = () => isCancelled() || Boolean(signal?.aborted)
    if (cancelled()) return null
    if (!videoEl) {
      onFailureReason?.('video_element_unavailable')
      return null
    }

    const MAX_RETRIES = 3
    const RETRY_BASE_MS = 2000

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      if (cancelled()) return null

      if (attempt > 0) {
        await new Promise(r => setTimeout(r, RETRY_BASE_MS * Math.pow(2, attempt - 1)))
        if (cancelled()) return null
      }

      const info = await this.fetchStreamInfo(cameraId, signal)
      if (cancelled()) return null

      if (!info) {
        if (attempt === MAX_RETRIES) {
          onFailureReason?.('stream_info_unavailable')
          return null
        }
        continue
      }

      if (info.go2rtc) {
        if (info.webrtc_ws && videoEl) {
          const player = new WebRTCPlayer(cameraId, info.webrtc_ws, videoEl)
          const ok = await this.connectPlayer(player, signal)
          if (cancelled()) { player.destroy(); return null }
          if (ok) return player
          onFailureReason?.('webrtc_failed')
          player.destroy()
        }
        if (info.mse_ws && videoEl) {
          const player = new MSEPlayer(cameraId, info.mse_ws, videoEl)
          const ok = await this.connectPlayer(player, signal)
          if (cancelled()) { player.destroy(); return null }
          if (ok) return player
          onFailureReason?.('mse_failed')
          player.destroy()
        }
      } else {
        onFailureReason?.('go2rtc_unavailable')
        break
      }
    }
    return null
  }
}
