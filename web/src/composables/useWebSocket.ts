import { ref, onUnmounted, type Ref } from 'vue'

type Topic = 'health' | 'cameras' | 'alerts' | 'tasks' | 'wall' | 'degradation' | 'heatmap' | 'audio_alert' | 'models' | 'model_release'

// ── Audio Alert System ──
const audioMuted = ref(false)
let audioCtx: AudioContext | null = null

function playAlertBeep(severity: string = 'medium'): void {
  if (audioMuted.value) return
  try {
    if (!audioCtx) audioCtx = new AudioContext()
    const ctx = audioCtx
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.connect(gain)
    gain.connect(ctx.destination)

    // Frequency varies by severity
    const freqMap: Record<string, number> = {
      high: 880,
      medium: 660,
      low: 440,
      info: 330,
    }
    osc.frequency.value = freqMap[severity] || 660
    osc.type = 'sine'

    // Duration varies by severity
    const durMap: Record<string, number> = { high: 0.4, medium: 0.25, low: 0.15, info: 0.1 }
    const dur = durMap[severity] || 0.25

    gain.gain.setValueAtTime(0.3, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + dur)
    osc.start(ctx.currentTime)
    osc.stop(ctx.currentTime + dur)

    // HIGH severity: double beep
    if (severity === 'high') {
      const osc2 = ctx.createOscillator()
      const gain2 = ctx.createGain()
      osc2.connect(gain2)
      gain2.connect(ctx.destination)
      osc2.frequency.value = 880
      osc2.type = 'sine'
      gain2.gain.setValueAtTime(0.3, ctx.currentTime + dur + 0.1)
      gain2.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + dur + 0.1 + dur)
      osc2.start(ctx.currentTime + dur + 0.1)
      osc2.stop(ctx.currentTime + dur + 0.1 + dur)
    }
  } catch {
    // Audio API may not be available
  }
}

function toggleAudioMute(): void {
  audioMuted.value = !audioMuted.value
}

interface WsMessage {
  topic: string
  data: any
  timestamp: number
}

interface UseWebSocketOptions {
  /** Topics to subscribe to */
  topics: Topic[]
  /** Callback when a message is received for a topic */
  onMessage?: (topic: Topic, data: any) => void
  /** Fallback polling function — called if WS fails after maxRetries */
  fallbackPoll?: () => void
  /** Fallback polling interval in ms (default 15000) */
  fallbackInterval?: number
}

function getCookie(name: string): string | null {
  const match = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`))
  return match ? decodeURIComponent(match[1]) : null
}

// ── Singleton WebSocket connection ──
// All components share a single WebSocket. Each useWebSocket() call adds
// a subscriber with its own topics and callbacks. When a subscriber unmounts,
// it is removed. The connection stays alive as long as at least one subscriber
// exists; it closes when the last one unsubscribes.

interface Subscriber {
  id: number
  topics: Set<Topic>
  onMessage?: (topic: Topic, data: any) => void
  fallbackPoll?: () => void
  fallbackInterval: number
  fallbackTimer: ReturnType<typeof setInterval> | null
}

let ws: WebSocket | null = null
let subscriberIdCounter = 0
const subscribers = new Map<number, Subscriber>()
let retryCount = 0
let retryTimer: ReturnType<typeof setTimeout> | null = null
import { DEFAULT_WS_RETRY_MAX, DEFAULT_WS_FALLBACK_INTERVAL_MS } from '../config/constants'

const MAX_RETRIES = DEFAULT_WS_RETRY_MAX
// After exhausting MAX_RETRIES we fall back to polling. Every FALLBACK_RECHECK_MS
// we make one more WS attempt — success brings us back to live updates, failure
// keeps us polling. Avoids the "must refresh the page" dead-end.
const FALLBACK_RECHECK_MS = 60_000
let fallbackRecheckTimer: ReturnType<typeof setTimeout> | null = null
const globalConnected = ref(false)
const globalReconnecting = ref(false)
const globalRetryCount = ref(0)
const globalFallbackMode = ref(false)
const globalNextRetryIn = ref(0) // seconds until next retry
let countdownTimer: ReturnType<typeof setInterval> | null = null

function allTopics(): Topic[] {
  const topics = new Set<Topic>()
  for (const sub of subscribers.values()) {
    for (const t of sub.topics) topics.add(t)
  }
  return [...topics]
}

function wsConnect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return
  if (subscribers.size === 0) return

  const token = getCookie('argus_session')
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const url = `${proto}://${location.host}/ws${token ? `?token=${token}` : ''}`

  ws = new WebSocket(url)

  ws.onopen = () => {
    globalConnected.value = true
    globalReconnecting.value = false
    globalRetryCount.value = 0
    globalFallbackMode.value = false
    globalNextRetryIn.value = 0
    if (countdownTimer) { clearInterval(countdownTimer); countdownTimer = null }
    clearFallbackRecheck()
    retryCount = 0
    // Stop all fallback polling
    for (const sub of subscribers.values()) {
      if (sub.fallbackTimer !== null) {
        clearInterval(sub.fallbackTimer)
        sub.fallbackTimer = null
      }
    }
    // Subscribe to the union of all topics
    ws!.send(JSON.stringify({ action: 'subscribe', topics: allTopics() }))
  }

  ws.onmessage = (event) => {
    try {
      const msg: WsMessage = JSON.parse(event.data)
      if (msg.topic === 'ping') {
        ws?.send(JSON.stringify({ action: 'pong' }))
        return
      }
      // Audio alert: play beep on audio_alert topic
      if (msg.topic === 'audio_alert' && msg.data?.severity) {
        playAlertBeep(msg.data.severity)
      }

      // Dispatch to subscribers interested in this topic
      for (const sub of subscribers.values()) {
        if (sub.topics.has(msg.topic as Topic)) {
          sub.onMessage?.(msg.topic as Topic, msg.data)
        }
      }
    } catch {
      // ignore malformed messages
    }
  }

  ws.onclose = () => {
    globalConnected.value = false
    globalReconnecting.value = true
    ws = null
    scheduleReconnect()
  }

  ws.onerror = () => {
    // onclose fires after onerror
  }
}

function scheduleReconnect() {
  if (subscribers.size === 0) return
  retryCount++
  globalRetryCount.value = retryCount
  if (retryCount > MAX_RETRIES) {
    // Fall back to per-subscriber polling and schedule a slow WS recheck.
    globalFallbackMode.value = true
    globalReconnecting.value = false
    for (const sub of subscribers.values()) {
      startFallbackPolling(sub)
    }
    scheduleFallbackRecheck()
    return
  }
  const delay = Math.min(1000 * Math.pow(2, retryCount - 1), 30000)
  globalNextRetryIn.value = Math.ceil(delay / 1000)
  if (countdownTimer) clearInterval(countdownTimer)
  countdownTimer = setInterval(() => {
    globalNextRetryIn.value = Math.max(0, globalNextRetryIn.value - 1)
    if (globalNextRetryIn.value <= 0 && countdownTimer) {
      clearInterval(countdownTimer)
      countdownTimer = null
    }
  }, 1000)
  retryTimer = setTimeout(wsConnect, delay)
}

// In fallback mode we periodically retry WebSocket. Success (ws.onopen) resets
// retryCount + globalFallbackMode so the normal code paths resume. Failure
// triggers onclose → scheduleReconnect → hits MAX_RETRIES again → re-arms this.
function scheduleFallbackRecheck() {
  if (fallbackRecheckTimer !== null) return
  fallbackRecheckTimer = setTimeout(() => {
    fallbackRecheckTimer = null
    if (subscribers.size === 0) return
    if (!globalFallbackMode.value) return
    // Reset retryCount so a single attempt happens; if it fails,
    // scheduleReconnect will bump back above MAX_RETRIES and re-arm us.
    retryCount = 0
    globalRetryCount.value = 0
    wsConnect()
  }, FALLBACK_RECHECK_MS)
}

function clearFallbackRecheck() {
  if (fallbackRecheckTimer !== null) {
    clearTimeout(fallbackRecheckTimer)
    fallbackRecheckTimer = null
  }
}

function startFallbackPolling(sub: Subscriber) {
  if (!sub.fallbackPoll || sub.fallbackTimer !== null) return
  sub.fallbackPoll()
  sub.fallbackTimer = setInterval(sub.fallbackPoll, sub.fallbackInterval)
}

function wsDisconnect() {
  if (retryTimer !== null) {
    clearTimeout(retryTimer)
    retryTimer = null
  }
  clearFallbackRecheck()
  if (countdownTimer) { clearInterval(countdownTimer); countdownTimer = null }
  if (ws) {
    ws.onclose = null
    ws.close()
    ws = null
  }
  globalConnected.value = false
  retryCount = 0
}

function resubscribe() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'subscribe', topics: allTopics() }))
  }
}

function addSubscriber(sub: Subscriber) {
  subscribers.set(sub.id, sub)
  if (!ws || ws.readyState === WebSocket.CLOSED) {
    retryCount = 0
    wsConnect()
  } else {
    resubscribe()
  }
}

function removeSubscriber(id: number) {
  const sub = subscribers.get(id)
  if (sub?.fallbackTimer !== null) {
    clearInterval(sub!.fallbackTimer!)
    sub!.fallbackTimer = null
  }
  subscribers.delete(id)
  if (subscribers.size === 0) {
    wsDisconnect()
  } else {
    resubscribe()
  }
}

// ── Public composable ──

export function useWebSocket(options: UseWebSocketOptions) {
  const {
    topics,
    onMessage,
    fallbackPoll,
    fallbackInterval = DEFAULT_WS_FALLBACK_INTERVAL_MS,
  } = options

  const connected = globalConnected
  const error = ref<string | null>(null)
  const topicData: Ref<Record<string, any>> = ref({})

  const subId = ++subscriberIdCounter
  const sub: Subscriber = {
    id: subId,
    topics: new Set(topics),
    onMessage: (topic, data) => {
      topicData.value = { ...topicData.value, [topic]: data }
      onMessage?.(topic, data)
    },
    fallbackPoll,
    fallbackInterval,
    fallbackTimer: null,
  }

  addSubscriber(sub)

  onUnmounted(() => {
    removeSubscriber(subId)
  })

  return {
    connected,
    reconnecting: globalReconnecting,
    retryCount: globalRetryCount,
    fallbackMode: globalFallbackMode,
    nextRetryIn: globalNextRetryIn,
    error,
    topicData,
    audioMuted,
    toggleAudioMute,
  }
}

// ── HMR cleanup ──
// Module-level state (`ws`, `subscribers`, timers) outlives a hot-module swap.
// Without this, every edit leaks a live WebSocket and stale subscribers, and
// eventually the browser tab eats several megabytes of idle sockets.
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    if (retryTimer !== null) { clearTimeout(retryTimer); retryTimer = null }
    if (countdownTimer !== null) { clearInterval(countdownTimer); countdownTimer = null }
    if (fallbackRecheckTimer !== null) { clearTimeout(fallbackRecheckTimer); fallbackRecheckTimer = null }
    for (const sub of subscribers.values()) {
      if (sub.fallbackTimer !== null) {
        clearInterval(sub.fallbackTimer)
        sub.fallbackTimer = null
      }
    }
    subscribers.clear()
    if (ws) {
      ws.onopen = null
      ws.onmessage = null
      ws.onerror = null
      ws.onclose = null
      try { ws.close() } catch { /* ignore */ }
      ws = null
    }
    globalConnected.value = false
    globalReconnecting.value = false
    globalFallbackMode.value = false
  })
}
