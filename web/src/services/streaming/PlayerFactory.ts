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

      try {
        this.ws = new WebSocket(this.wsUrl)
        this.ws.onopen = () => {
          this.pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] })
          this.pc.ontrack = (event) => {
            if (this.videoEl && event.streams[0]) {
              this.videoEl.srcObject = event.streams[0]
              this.setStatus('playing')
              settle(true)
            }
          }
          this.pc.onconnectionstatechange = () => {
            if (this.pc && (this.pc.connectionState === 'failed' || this.pc.connectionState === 'disconnected')) {
              this.onReconnectRequest()
            }
          }
          this.pc.onicecandidate = (event) => {
            if (event.candidate && this.ws?.readyState === WebSocket.OPEN) {
              this.ws.send(JSON.stringify({ type: 'webrtc/candidate', value: event.candidate.candidate }))
            }
          }
          this.pc.addTransceiver('video', { direction: 'recvonly' })
          this.pc.addTransceiver('audio', { direction: 'recvonly' })
          this.pc.createOffer().then((offer) => {
            this.pc!.setLocalDescription(offer).then(() => {
              this.ws!.send(JSON.stringify({ type: 'webrtc/offer', value: offer.sdp }))
            })
          })
        }

        this.ws.onmessage = (event) => {
          const msg = JSON.parse(event.data)
          if (msg.type === 'webrtc/answer' && this.pc) {
            this.pc.setRemoteDescription(new RTCSessionDescription({ type: 'answer', sdp: msg.value }))
          } else if (msg.type === 'webrtc/candidate' && this.pc) {
            this.pc.addIceCandidate(new RTCIceCandidate({ candidate: msg.value, sdpMid: '0' }))
          }
        }

        this.ws.onerror = () => {
          this.cleanup()
          settle(false)
        }

        this.ws.onclose = () => {
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
  }

  destroy() {
    this.cleanup()
  }
}

export class MSEPlayer implements PlayerInstance {
  status: 'idle' | 'connecting' | 'playing' | 'error' | 'fallback' = 'connecting'
  onStatusChange: (status: 'idle' | 'connecting' | 'playing' | 'error' | 'fallback') => void = () => {}
  onReconnectRequest: () => void = () => {}

  private mseWs: WebSocket | null = null
  private mediaSource: MediaSource | null = null
  private settled = false
  private MAX_MSE_BUFFER_QUEUE = 30
  private cameraId: string
  private wsUrl: string
  private videoEl: HTMLVideoElement

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

      try {
        if (!this.videoEl) {
          settle(false)
          return
        }
        this.mediaSource = new MediaSource()
        this.videoEl.srcObject = null
        this.videoEl.src = URL.createObjectURL(this.mediaSource)

        this.mediaSource.addEventListener('sourceopen', () => {
          this.mseWs = new WebSocket(this.wsUrl)
          this.mseWs.binaryType = 'arraybuffer'

          let sourceBuffer: SourceBuffer | null = null
          const bufferQueue: ArrayBuffer[] = []

          this.mseWs.onmessage = (event) => {
            if (typeof event.data === 'string') {
              const msg = JSON.parse(event.data)
              if (msg.type === 'mse' && msg.value && this.mediaSource) {
                try {
                  sourceBuffer = this.mediaSource.addSourceBuffer(msg.value)
                  sourceBuffer.mode = 'segments'
                  sourceBuffer.addEventListener('updateend', () => {
                    if (bufferQueue.length > 0 && sourceBuffer && !sourceBuffer.updating) {
                      sourceBuffer.appendBuffer(bufferQueue.shift()!)
                    }
                  })
                } catch {
                  this.cleanup()
                  settle(false)
                }
              }
            } else if (event.data instanceof ArrayBuffer && sourceBuffer) {
              if (!this.settled) {
                this.setStatus('playing')
                settle(true)
              }
              if (bufferQueue.length >= this.MAX_MSE_BUFFER_QUEUE) {
                bufferQueue.splice(0, bufferQueue.length - this.MAX_MSE_BUFFER_QUEUE + 1)
              }
              if (sourceBuffer.updating) {
                bufferQueue.push(event.data)
              } else {
                sourceBuffer.appendBuffer(event.data)
              }
            }
          }

          this.mseWs.onerror = () => {
            this.cleanup()
            settle(false)
          }

          this.mseWs.onclose = () => {
            if (!this.settled) {
              this.cleanup()
              settle(false)
            } else if (this.status === 'playing') {
              this.onReconnectRequest()
            }
          }
        })
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
    if (this.mediaSource && this.mediaSource.readyState === 'open') {
      try { this.mediaSource.endOfStream() } catch { /* ignore */ }
    }
    this.mediaSource = null
  }

  destroy() {
    this.cleanup()
  }
}

export class PlayerFactory {
  static async fetchStreamInfo(cameraId: string): Promise<StreamInfo | null> {
    try {
      const resp = await axios.get(`/api/streaming/${cameraId}`)
      return resp.data as StreamInfo
    } catch (e) {
      logger.debug(`[go2rtc] fetchStreamInfo failed for ${cameraId}:`, e)
      return null
    }
  }

  static async create(cameraId: string, videoEl: HTMLVideoElement | null, isCancelled: () => boolean): Promise<PlayerInstance | null> {
    const MAX_RETRIES = 3
    const RETRY_BASE_MS = 2000

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      if (isCancelled()) return null

      if (attempt > 0) {
        await new Promise(r => setTimeout(r, RETRY_BASE_MS * Math.pow(2, attempt - 1)))
        if (isCancelled()) return null
      }

      const info = await this.fetchStreamInfo(cameraId)
      if (isCancelled()) return null

      if (!info) {
        if (attempt === MAX_RETRIES) return null
        continue
      }

      if (info.go2rtc) {
        if (info.webrtc_ws && videoEl) {
          const player = new WebRTCPlayer(cameraId, info.webrtc_ws, videoEl)
          const ok = await player.connect()
          if (isCancelled()) { player.destroy(); return null }
          if (ok) return player
          player.destroy()
        }
        if (info.mse_ws && videoEl) {
          const player = new MSEPlayer(cameraId, info.mse_ws, videoEl)
          const ok = await player.connect()
          if (isCancelled()) { player.destroy(); return null }
          if (ok) return player
          player.destroy()
        }
      } else {
        break
      }
    }
    return null
  }
}
