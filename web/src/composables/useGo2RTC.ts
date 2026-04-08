/**
 * go2rtc WebRTC/MSE player composable.
 *
 * Negotiates the best available streaming protocol with go2rtc:
 *   1. WebRTC (preferred — lowest latency, H.264 passthrough)
 *   2. MSE / Media Source Extensions (fallback — wider compatibility)
 *   3. MJPEG via <img> (legacy fallback when go2rtc is unavailable)
 */

import { ref, onUnmounted, type Ref } from 'vue'
import axios from 'axios'

export type StreamStatus = 'idle' | 'connecting' | 'playing' | 'error' | 'fallback'

const MAX_MSE_BUFFER_QUEUE = 30
const MAX_RETRIES = 3
const RETRY_BASE_MS = 2000
const MAX_RECONNECT = 3

interface StreamInfo {
  camera_id: string
  go2rtc: boolean
  webrtc_ws?: string
  mse_ws?: string
  fallback: string
}

export function useGo2RTC(cameraId: Ref<string> | string) {
  const videoRef = ref<HTMLVideoElement | null>(null)
  const status = ref<StreamStatus>('idle')

  let pc: RTCPeerConnection | null = null
  let ws: WebSocket | null = null
  let mseWs: WebSocket | null = null
  let mediaSource: MediaSource | null = null
  let generation = 0
  let reconnectCount = 0

  const getCameraId = () => typeof cameraId === 'string' ? cameraId : cameraId.value

  async function fetchStreamInfo(): Promise<StreamInfo | null> {
    try {
      const resp = await axios.get(`/api/streaming/${getCameraId()}`)
      return resp.data as StreamInfo
    } catch (e) {
      console.warn(`[go2rtc] fetchStreamInfo failed for ${getCameraId()}:`, e)
      return null
    }
  }

  async function connectWebRTC(wsUrl: string): Promise<boolean> {
    const cam = getCameraId()
    return new Promise((resolve) => {
      let settled = false
      const settle = (ok: boolean) => { if (!settled) { settled = true; clearTimeout(timeout); resolve(ok) } }

      const timeout = setTimeout(() => {
        console.warn(`[go2rtc] WebRTC timeout for ${cam}, pc state: ${pc?.connectionState}`)
        cleanup()
        settle(false)
      }, 15_000)

      try {
        console.debug(`[go2rtc] WebRTC connecting: ${wsUrl}`)
        ws = new WebSocket(wsUrl)

        ws.onopen = () => {
          console.debug(`[go2rtc] WebRTC ws open for ${cam}`)
          pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
          })

          pc.ontrack = (event) => {
            console.debug(`[go2rtc] WebRTC ontrack for ${cam}, videoRef=${!!videoRef.value}`)
            if (videoRef.value && event.streams[0]) {
              videoRef.value.srcObject = event.streams[0]
              status.value = 'playing'
              settle(true)
            }
          }

          pc.onconnectionstatechange = () => {
            console.debug(`[go2rtc] WebRTC state: ${pc?.connectionState} for ${cam}`)
            if (pc && (pc.connectionState === 'failed' || pc.connectionState === 'disconnected')) {
              if (reconnectCount < MAX_RECONNECT) {
                reconnectCount++
                start()
              } else {
                status.value = 'error'
              }
            }
          }

          pc.onicecandidate = (event) => {
            if (event.candidate && ws?.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({
                type: 'webrtc/candidate',
                value: event.candidate.candidate,
              }))
            }
          }

          pc.addTransceiver('video', { direction: 'recvonly' })
          pc.addTransceiver('audio', { direction: 'recvonly' })

          pc.createOffer().then((offer) => {
            pc!.setLocalDescription(offer).then(() => {
              console.debug(`[go2rtc] WebRTC offer sent for ${cam}`)
              ws!.send(JSON.stringify({
                type: 'webrtc/offer',
                value: offer.sdp,
              }))
            })
          })
        }

        ws.onmessage = (event) => {
          const msg = JSON.parse(event.data)
          console.debug(`[go2rtc] WebRTC msg: ${msg.type} for ${cam}`)
          if (msg.type === 'webrtc/answer' && pc) {
            pc.setRemoteDescription(new RTCSessionDescription({
              type: 'answer',
              sdp: msg.value,
            }))
          } else if (msg.type === 'webrtc/candidate' && pc) {
            pc.addIceCandidate(new RTCIceCandidate({ candidate: msg.value, sdpMid: '0' }))
          }
        }

        ws.onerror = (ev) => {
          console.warn(`[go2rtc] WebRTC ws error for ${cam}:`, ev)
          cleanup()
          settle(false)
        }

        ws.onclose = (ev) => {
          console.debug(`[go2rtc] WebRTC ws close for ${cam}, code=${ev.code}`)
          if (!settled) {
            cleanup()
            settle(false)
          }
        }
      } catch {
        settle(false)
      }
    })
  }

  async function connectMSE(wsUrl: string): Promise<boolean> {
    if (!('MediaSource' in window)) return false

    return new Promise((resolve) => {
      let settled = false
      const settle = (ok: boolean) => { if (!settled) { settled = true; clearTimeout(timeout); resolve(ok) } }

      const timeout = setTimeout(() => {
        console.warn(`[go2rtc] MSE timeout for ${getCameraId()}`)
        cleanupMSE()
        settle(false)
      }, 15_000)

      try {
        if (!videoRef.value) {
          console.warn(`[go2rtc] MSE: videoRef is null for ${getCameraId()}, skipping`)
          settle(false)
          return
        }
        mediaSource = new MediaSource()
        videoRef.value.srcObject = null
        videoRef.value.src = URL.createObjectURL(mediaSource)

        mediaSource.addEventListener('sourceopen', () => {
          console.debug(`[go2rtc] MSE sourceopen for ${getCameraId()}, connecting: ${wsUrl}`)
          mseWs = new WebSocket(wsUrl)
          mseWs.binaryType = 'arraybuffer'

          let sourceBuffer: SourceBuffer | null = null
          const bufferQueue: ArrayBuffer[] = []

          mseWs.onmessage = (event) => {
            if (typeof event.data === 'string') {
              const msg = JSON.parse(event.data)
              if (msg.type === 'mse' && msg.value && mediaSource) {
                try {
                  sourceBuffer = mediaSource.addSourceBuffer(msg.value)
                  sourceBuffer.mode = 'segments'
                  sourceBuffer.addEventListener('updateend', () => {
                    if (bufferQueue.length > 0 && sourceBuffer && !sourceBuffer.updating) {
                      sourceBuffer.appendBuffer(bufferQueue.shift()!)
                    }
                  })
                } catch {
                  cleanupMSE()
                  settle(false)
                }
              }
            } else if (event.data instanceof ArrayBuffer && sourceBuffer) {
              if (!settled) {
                status.value = 'playing'
                settle(true)
              }
              // Cap buffer queue to prevent unbounded memory growth
              if (bufferQueue.length >= MAX_MSE_BUFFER_QUEUE) {
                bufferQueue.splice(0, bufferQueue.length - MAX_MSE_BUFFER_QUEUE + 1)
              }
              if (sourceBuffer.updating) {
                bufferQueue.push(event.data)
              } else {
                sourceBuffer.appendBuffer(event.data)
              }
            }
          }

          mseWs.onerror = (ev) => {
            console.warn(`[go2rtc] MSE ws error for ${getCameraId()}:`, ev)
            cleanupMSE()
            settle(false)
          }

          mseWs.onclose = (ev) => {
            console.debug(`[go2rtc] MSE ws close for ${getCameraId()}, code=${ev.code}`)
            if (!settled) {
              cleanupMSE()
              settle(false)
            } else if (status.value === 'playing') {
              if (reconnectCount < MAX_RECONNECT) {
                reconnectCount++
                start()
              } else {
                status.value = 'error'
              }
            }
          }
        })
      } catch {
        settle(false)
      }
    })
  }

  async function start() {
    stop()
    const thisGen = ++generation
    status.value = 'connecting'

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      if (thisGen !== generation) return

      if (attempt > 0) {
        await new Promise(r => setTimeout(r, RETRY_BASE_MS * Math.pow(2, attempt - 1)))
        if (thisGen !== generation) return
      }

      const info = await fetchStreamInfo()
      if (thisGen !== generation) return

      if (!info) {
        if (attempt === MAX_RETRIES) { status.value = 'error'; return }
        continue
      }

      if (info.go2rtc) {
        if (info.webrtc_ws) {
          const ok = await connectWebRTC(info.webrtc_ws)
          if (thisGen !== generation) return
          if (ok) { reconnectCount = 0; return }
        }
        if (info.mse_ws) {
          const ok = await connectMSE(info.mse_ws)
          if (thisGen !== generation) return
          if (ok) { reconnectCount = 0; return }
        }
      } else {
        // go2rtc not available, no point retrying
        break
      }
    }

    status.value = 'fallback'
  }

  function cleanup() {
    if (pc) {
      pc.onconnectionstatechange = null
      pc.ontrack = null
      pc.onicecandidate = null
      pc.close()
      pc = null
    }
    if (ws) {
      ws.onopen = null
      ws.onclose = null
      ws.onerror = null
      ws.onmessage = null
      ws.close()
      ws = null
    }
  }

  function cleanupMSE() {
    if (mseWs) {
      mseWs.onopen = null
      mseWs.onclose = null
      mseWs.onerror = null
      mseWs.onmessage = null
      mseWs.close()
      mseWs = null
    }
    if (mediaSource && mediaSource.readyState === 'open') {
      try { mediaSource.endOfStream() } catch { /* ignore */ }
    }
    mediaSource = null
  }

  function stop() {
    generation++  // cancel any in-flight start()
    cleanup()
    cleanupMSE()
    if (videoRef.value) {
      videoRef.value.srcObject = null
      videoRef.value.src = ''
    }
    status.value = 'idle'
  }

  onUnmounted(stop)

  return {
    videoRef,
    status,
    start,
    stop,
  }
}
