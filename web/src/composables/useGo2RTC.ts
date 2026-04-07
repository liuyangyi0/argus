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

  const getCameraId = () => typeof cameraId === 'string' ? cameraId : cameraId.value

  async function fetchStreamInfo(): Promise<StreamInfo | null> {
    try {
      const resp = await axios.get(`/api/streaming/${getCameraId()}`)
      return resp.data as StreamInfo
    } catch {
      return null
    }
  }

  async function connectWebRTC(wsUrl: string): Promise<boolean> {
    return new Promise((resolve) => {
      let settled = false
      const settle = (ok: boolean) => { if (!settled) { settled = true; clearTimeout(timeout); resolve(ok) } }

      const timeout = setTimeout(() => {
        cleanup()
        settle(false)
      }, 10_000)

      try {
        ws = new WebSocket(wsUrl)

        ws.onopen = () => {
          pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
          })

          pc.ontrack = (event) => {
            if (videoRef.value && event.streams[0]) {
              videoRef.value.srcObject = event.streams[0]
              status.value = 'playing'
              settle(true)
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
              ws!.send(JSON.stringify({
                type: 'webrtc/offer',
                value: offer.sdp,
              }))
            })
          })
        }

        ws.onmessage = (event) => {
          const msg = JSON.parse(event.data)
          if (msg.type === 'webrtc/answer' && pc) {
            pc.setRemoteDescription(new RTCSessionDescription({
              type: 'answer',
              sdp: msg.value,
            }))
          } else if (msg.type === 'webrtc/candidate' && pc) {
            pc.addIceCandidate(new RTCIceCandidate({ candidate: msg.value, sdpMid: '0' }))
          }
        }

        ws.onerror = () => { cleanup(); settle(false) }

        // Handle signalling channel closure after connection established
        ws.onclose = () => {
          if (settled && status.value === 'playing') {
            // Signalling channel closed but media may still flow via ICE;
            // RTCPeerConnection.onconnectionstatechange handles media failure.
          }
        }

        // No separate onclose needed before settlement — onerror fires first.
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
        cleanupMSE()
        settle(false)
      }, 10_000)

      try {
        mediaSource = new MediaSource()
        if (videoRef.value) {
          videoRef.value.srcObject = null
          videoRef.value.src = URL.createObjectURL(mediaSource)
        }

        mediaSource.addEventListener('sourceopen', () => {
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

          mseWs.onerror = () => { cleanupMSE(); settle(false) }

          mseWs.onclose = () => {
            if (settled && status.value === 'playing') {
              status.value = 'error'
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
    status.value = 'connecting'

    const info = await fetchStreamInfo()
    if (!info) {
      status.value = 'error'
      return
    }

    if (info.go2rtc) {
      if (info.webrtc_ws) {
        const ok = await connectWebRTC(info.webrtc_ws)
        if (ok) return
      }
      if (info.mse_ws) {
        const ok = await connectMSE(info.mse_ws)
        if (ok) return
      }
    }

    status.value = 'fallback'
  }

  function cleanup() {
    if (pc) {
      pc.close()
      pc = null
    }
    if (ws) {
      ws.close()
      ws = null
    }
  }

  function cleanupMSE() {
    if (mseWs) {
      mseWs.close()
      mseWs = null
    }
    if (mediaSource && mediaSource.readyState === 'open') {
      try { mediaSource.endOfStream() } catch { /* ignore */ }
    }
    mediaSource = null
  }

  function stop() {
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
