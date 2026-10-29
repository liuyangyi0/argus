import { ref, onUnmounted, type Ref } from 'vue'
import {
  DEFAULT_WS_FALLBACK_INTERVAL_MS,
  DEFAULT_WS_RECONCILE_INTERVAL_MS,
  DEFAULT_WS_RETRY_MAX,
} from '../config/constants'
import type { WsEvent } from '../types/api'

export type Topic = 'health' | 'cameras' | 'alerts' | 'tasks' | 'wall' | 'degradation' | 'heatmap' | 'audio_alert' | 'models' | 'model_release' | 'system_degradation' | 'system_errors' | 'task_events'

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

interface UseWebSocketOptions<TData = unknown> {
  /** Topics to subscribe to */
  topics: Topic[]
  /** Callback when a message is received for a topic */
  onMessage?: (topic: Topic, data: TData, event?: WsEvent<TData>) => void
  /** Fallback polling function — called if WS fails after maxRetries */
  fallbackPoll?: () => unknown | Promise<unknown>
  /** Fallback polling interval in ms (default 15000) */
  fallbackInterval?: number
  /** REST reconciliation interval while WS is healthy (default 60000). */
  reconcileInterval?: number
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
  onMessage?: (topic: Topic, data: unknown, event?: WsEvent<unknown>) => void
  fallbackPoll?: () => unknown | Promise<unknown>
  fallbackInterval: number
  fallbackTimer: ReturnType<typeof setInterval> | null
  reconcileInterval: number
  reconcileTimer: ReturnType<typeof setInterval> | null
  pollInFlight: boolean
}

let ws: WebSocket | null = null
let subscriberIdCounter = 0
const subscribers = new Map<number, Subscriber>()
let retryCount = 0
let retryTimer: ReturnType<typeof setTimeout> | null = null

const MAX_RETRIES = DEFAULT_WS_RETRY_MAX
// After exhausting MAX_RETRIES we fall back to polling. Every FALLBACK_RECHECK_MS
// we make one more WS attempt — success brings us back to live updates, failure
// keeps us polling. Avoids the "must refresh the page" dead-end.
const FALLBACK_RECHECK_MS = 60_000
let fallbackRecheckTimer: ReturnType<typeof setTimeout> | null = null
let networkListenersInstalled = false
const globalConnected = ref(false)
const globalReconnecting = ref(false)
const globalRetryCount = ref(0)
const globalFallbackMode = ref(false)
const globalSubscriptionAcknowledged = ref(false)
const globalNextRetryIn = ref(0) // seconds until next retry
let countdownTimer: ReturnType<typeof setInterval> | null = null
let activeServerEpoch: string | null = null
const latestResourceRevisions = new Map<Topic, Map<string, number>>()
const resourceRevisionOrder = new Map<string, { topic: Topic; resourceId: string }>()
const seenEventIds = new Set<string>()
const seenEventOrder: string[] = []
const MAX_SEEN_EVENT_IDS = 2048
const MAX_RESOURCE_REVISIONS = 4096

function allTopics(): Topic[] {
  const topics = new Set<Topic>()
  for (const sub of subscribers.values()) {
    for (const t of sub.topics) topics.add(t)
  }
  return [...topics]
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object'
}

function isTopic(value: unknown): value is Topic {
  return typeof value === 'string' && allTopics().includes(value as Topic)
}

function isV1Event(value: unknown): value is WsEvent<unknown> {
  if (!isRecord(value)) return false
  return (
    value.schema_version === 1
    && typeof value.event_id === 'string' && value.event_id.length > 0
    && typeof value.topic === 'string'
    && typeof value.type === 'string' && value.type.length > 0
    && typeof value.resource_id === 'string' && value.resource_id.length > 0
    && typeof value.revision === 'number' && Number.isInteger(value.revision) && value.revision >= 0
    && typeof value.occurred_at === 'string' && value.occurred_at.length > 0
    && 'data' in value
  )
}

function topicRevisionSnapshot(): Record<string, number> {
  const snapshot: Record<string, number> = {}
  for (const [topic, resources] of latestResourceRevisions.entries()) {
    snapshot[topic] = Math.max(0, ...resources.values())
  }
  return snapshot
}

function resourceRevisionSnapshot(): Record<string, Record<string, number>> {
  const snapshot: Record<string, Record<string, number>> = {}
  for (const [topic, resources] of latestResourceRevisions.entries()) {
    snapshot[topic] = Object.fromEntries(resources.entries())
  }
  return snapshot
}

function revisionEntryKey(topic: Topic, resourceId: string): string {
  return `${topic.length}:${topic}${resourceId}`
}

function rememberResourceRevision(topic: Topic, resourceId: string, revision: number): void {
  let resources = latestResourceRevisions.get(topic)
  if (!resources) {
    resources = new Map<string, number>()
    latestResourceRevisions.set(topic, resources)
  }
  resources.set(resourceId, revision)

  const key = revisionEntryKey(topic, resourceId)
  resourceRevisionOrder.delete(key)
  resourceRevisionOrder.set(key, { topic, resourceId })
  while (resourceRevisionOrder.size > MAX_RESOURCE_REVISIONS) {
    const oldestKey = resourceRevisionOrder.keys().next().value as string | undefined
    if (oldestKey === undefined) break
    const oldest = resourceRevisionOrder.get(oldestKey)
    resourceRevisionOrder.delete(oldestKey)
    if (!oldest) continue
    const topicResources = latestResourceRevisions.get(oldest.topic)
    topicResources?.delete(oldest.resourceId)
    if (topicResources?.size === 0) latestResourceRevisions.delete(oldest.topic)
  }
}

function rememberEventId(eventId: string): boolean {
  if (seenEventIds.has(eventId)) return false
  seenEventIds.add(eventId)
  seenEventOrder.push(eventId)
  if (seenEventOrder.length > MAX_SEEN_EVENT_IDS) {
    const expired = seenEventOrder.shift()
    if (expired) seenEventIds.delete(expired)
  }
  return true
}

async function runSubscriberPoll(sub: Subscriber): Promise<void> {
  if (!sub.fallbackPoll || sub.pollInFlight) return
  sub.pollInFlight = true
  try {
    await sub.fallbackPoll()
  } catch {
    // The page-level loader owns user-facing REST errors. Reconciliation must
    // never create an unhandled promise rejection or stop future attempts.
  } finally {
    sub.pollInFlight = false
  }
}

function pollSubscribers(topics?: Set<Topic>): void {
  for (const sub of subscribers.values()) {
    if (topics && ![...sub.topics].some(topic => topics.has(topic))) continue
    void runSubscriberPoll(sub)
  }
}

function resyncTopics(data: unknown): Set<Topic> | undefined {
  if (!isRecord(data) || !Array.isArray(data.topics)) return undefined
  const result = new Set<Topic>()
  for (const item of data.topics) {
    const topic = typeof item === 'string'
      ? item
      : isRecord(item) ? item.topic : null
    if (isTopic(topic)) result.add(topic)
  }
  return result.size ? result : undefined
}

function applySubscribeAck(data: unknown): void {
  if (!isRecord(data)) return

  const serverEpoch = typeof data.server_epoch === 'string' && data.server_epoch
    ? data.server_epoch
    : null
  const epochChanged = Boolean(
    serverEpoch
    && activeServerEpoch
    && serverEpoch !== activeServerEpoch,
  )
  if (serverEpoch) activeServerEpoch = serverEpoch
  const subscribedTopics = Array.isArray(data.subscriptions)
    ? data.subscriptions.filter(isTopic)
    : allTopics()
  // A process restart starts a new revision epoch. Within one epoch cursors
  // never move backwards, including after the server's bounded LRU evicts a
  // resource, so a repeated ACK may only advance existing client high-water.
  if (epochChanged) {
    latestResourceRevisions.clear()
    resourceRevisionOrder.clear()
  }

  if (isRecord(data.resource_revisions)) {
    for (const [topic, rawResources] of Object.entries(data.resource_revisions)) {
      if (!isTopic(topic) || !isRecord(rawResources)) continue
      for (const [resourceId, revision] of Object.entries(rawResources)) {
        if (typeof revision === 'number' && Number.isInteger(revision) && revision >= 0) {
          const previous = latestResourceRevisions.get(topic)?.get(resourceId) ?? 0
          if (epochChanged || revision > previous) {
            rememberResourceRevision(topic, resourceId, revision)
          }
        }
      }
    }
  }
  globalSubscriptionAcknowledged.value = true
  if (epochChanged) pollSubscribers(new Set(subscribedTopics))
}

function dispatchTopicMessage(
  topic: Topic,
  data: unknown,
  event?: WsEvent<unknown>,
): void {
  if (topic === 'audio_alert' && isRecord(data) && typeof data.severity === 'string') {
    playAlertBeep(data.severity)
  }
  for (const sub of subscribers.values()) {
    if (sub.topics.has(topic)) sub.onMessage?.(topic, data, event)
  }
}

function handleV1Event(event: WsEvent<unknown>, socket: WebSocket): void {
  if (!rememberEventId(event.event_id)) return

  if (event.topic === 'control') {
    if (event.type === 'subscribe_ack') applySubscribeAck(event.data)
    if (event.type === 'resync_required') pollSubscribers(resyncTopics(event.data))
    return
  }
  if (event.topic === 'ping' || event.type === 'ping') {
    if (ws === socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ action: 'pong' }))
    }
    return
  }
  if (!isTopic(event.topic)) return

  const resources = latestResourceRevisions.get(event.topic)
  const previous = resources?.get(event.resource_id) ?? 0
  if (event.revision <= previous) return
  if (previous > 0 && event.revision > previous + 1) {
    pollSubscribers(new Set([event.topic]))
  }
  rememberResourceRevision(event.topic, event.resource_id, event.revision)
  dispatchTopicMessage(event.topic, event.data, event)
}

function wsConnect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return
  if (subscribers.size === 0) return
  if (typeof navigator !== 'undefined' && navigator.onLine === false) {
    handleNetworkOffline()
    return
  }

  const token = getCookie('argus_session')
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const url = `${proto}://${location.host}/ws${token ? `?token=${token}` : ''}`

  const socket = new WebSocket(url)
  ws = socket

  socket.onopen = () => {
    if (ws !== socket) return
    globalConnected.value = true
    globalReconnecting.value = false
    globalRetryCount.value = 0
    globalFallbackMode.value = false
    globalSubscriptionAcknowledged.value = false
    globalNextRetryIn.value = 0
    if (countdownTimer) { clearInterval(countdownTimer); countdownTimer = null }
    clearFallbackRecheck()
    retryCount = 0
    // Stop high-frequency disconnected polling. The 60s REST reconciliation
    // timer intentionally remains active even while WS is healthy.
    for (const sub of subscribers.values()) {
      if (sub.fallbackTimer !== null) {
        clearInterval(sub.fallbackTimer)
        sub.fallbackTimer = null
      }
    }
    // Subscribe to the union of all topics
    socket.send(JSON.stringify({
      action: 'subscribe',
      topics: allTopics(),
      revisions: topicRevisionSnapshot(),
      resource_revisions: resourceRevisionSnapshot(),
    }))
  }

  socket.onmessage = (event) => {
    if (ws !== socket) return
    try {
      const raw: unknown = JSON.parse(event.data)
      if (isV1Event(raw)) {
        handleV1Event(raw, socket)
        return
      }
      // Business events are accepted only through the complete schema-v1
      // validator above. Unversioned legacy payloads are deliberately ignored.
    } catch {
      // ignore malformed messages
    }
  }

  socket.onclose = () => {
    if (ws !== socket) return
    detachSocket(socket)
    globalConnected.value = false
    globalSubscriptionAcknowledged.value = false
    globalReconnecting.value = true
    ws = null
    scheduleReconnect()
  }

  socket.onerror = () => {
    if (ws !== socket) return
    // onclose fires after onerror
  }
}

function detachSocket(socket: WebSocket): void {
  socket.onopen = null
  socket.onmessage = null
  socket.onerror = null
  socket.onclose = null
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

function stopFallbackPolling(): void {
  for (const sub of subscribers.values()) {
    if (sub.fallbackTimer !== null) {
      clearInterval(sub.fallbackTimer)
      sub.fallbackTimer = null
    }
  }
}

function handleNetworkOffline(): void {
  if (subscribers.size === 0) return
  if (retryTimer !== null) {
    clearTimeout(retryTimer)
    retryTimer = null
  }
  clearFallbackRecheck()
  if (countdownTimer !== null) {
    clearInterval(countdownTimer)
    countdownTimer = null
  }
  if (ws) {
    const socket = ws
    ws = null
    detachSocket(socket)
    socket.close()
  }
  stopFallbackPolling()
  globalConnected.value = false
  globalSubscriptionAcknowledged.value = false
  globalReconnecting.value = true
  globalFallbackMode.value = false
  globalRetryCount.value = Math.max(1, globalRetryCount.value)
  globalNextRetryIn.value = 0
}

function handleNetworkOnline(): void {
  if (subscribers.size === 0) return
  retryCount = 0
  globalRetryCount.value = 0
  globalFallbackMode.value = false
  globalReconnecting.value = true
  // Reconcile immediately instead of waiting for the healthy 60-second timer.
  // This repairs events missed while the browser was offline even if the
  // server's revision history has already rolled over.
  pollSubscribers()
  wsConnect()
}

function installNetworkListeners(): void {
  if (networkListenersInstalled || typeof window === 'undefined') return
  window.addEventListener('offline', handleNetworkOffline)
  window.addEventListener('online', handleNetworkOnline)
  networkListenersInstalled = true
}

function removeNetworkListeners(): void {
  if (!networkListenersInstalled || typeof window === 'undefined') return
  window.removeEventListener('offline', handleNetworkOffline)
  window.removeEventListener('online', handleNetworkOnline)
  networkListenersInstalled = false
}

function startFallbackPolling(sub: Subscriber) {
  if (!sub.fallbackPoll || sub.fallbackTimer !== null) return
  void runSubscriberPoll(sub)
  sub.fallbackTimer = setInterval(() => {
    void runSubscriberPoll(sub)
  }, sub.fallbackInterval)
}

function startReconciliation(sub: Subscriber) {
  if (!sub.fallbackPoll || sub.reconcileTimer !== null || sub.reconcileInterval <= 0) return
  sub.reconcileTimer = setInterval(() => {
    if (globalConnected.value) void runSubscriberPoll(sub)
  }, sub.reconcileInterval)
}

function wsDisconnect() {
  if (retryTimer !== null) {
    clearTimeout(retryTimer)
    retryTimer = null
  }
  clearFallbackRecheck()
  if (countdownTimer) { clearInterval(countdownTimer); countdownTimer = null }
  if (ws) {
    const socket = ws
    ws = null
    detachSocket(socket)
    socket.close()
  }
  globalConnected.value = false
  globalReconnecting.value = false
  globalFallbackMode.value = false
  globalSubscriptionAcknowledged.value = false
  globalRetryCount.value = 0
  globalNextRetryIn.value = 0
  retryCount = 0
}

function resubscribe() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    globalSubscriptionAcknowledged.value = false
    ws.send(JSON.stringify({
      action: 'subscribe',
      topics: allTopics(),
      revisions: topicRevisionSnapshot(),
      resource_revisions: resourceRevisionSnapshot(),
    }))
  }
}

function addSubscriber(sub: Subscriber) {
  subscribers.set(sub.id, sub)
  installNetworkListeners()
  startReconciliation(sub)
  if (!ws || ws.readyState === WebSocket.CLOSED) {
    retryCount = 0
    wsConnect()
  } else {
    resubscribe()
  }
}

function removeSubscriber(id: number) {
  const sub = subscribers.get(id)
  if (sub && sub.fallbackTimer !== null) {
    clearInterval(sub.fallbackTimer)
    sub.fallbackTimer = null
  }
  if (sub && sub.reconcileTimer !== null) {
    clearInterval(sub.reconcileTimer)
    sub.reconcileTimer = null
  }
  subscribers.delete(id)
  if (subscribers.size === 0) {
    wsDisconnect()
    removeNetworkListeners()
  } else {
    resubscribe()
  }
}

// ── Public composable ──

export function useWebSocket<TData = unknown>(options: UseWebSocketOptions<TData>) {
  const {
    topics,
    onMessage,
    fallbackPoll,
    fallbackInterval = DEFAULT_WS_FALLBACK_INTERVAL_MS,
    reconcileInterval = DEFAULT_WS_RECONCILE_INTERVAL_MS,
  } = options

  const connected = globalConnected
  const error = ref<string | null>(null)
  const topicData: Ref<Record<string, unknown>> = ref({})

  const subId = ++subscriberIdCounter
  const sub: Subscriber = {
    id: subId,
    topics: new Set(topics),
    onMessage: (topic, data, event) => {
      topicData.value = { ...topicData.value, [topic]: data }
      // The generic type belongs to this subscriber; the shared socket keeps
      // heterogeneous subscribers as unknown and narrows only at dispatch.
      onMessage?.(topic, data as TData, event as WsEvent<TData> | undefined)
    },
    fallbackPoll,
    fallbackInterval,
    fallbackTimer: null,
    reconcileInterval,
    reconcileTimer: null,
    pollInFlight: false,
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
    subscriptionAcknowledged: globalSubscriptionAcknowledged,
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
      if (sub.reconcileTimer !== null) {
        clearInterval(sub.reconcileTimer)
        sub.reconcileTimer = null
      }
    }
    subscribers.clear()
    removeNetworkListeners()
    if (ws) {
      const socket = ws
      ws = null
      detachSocket(socket)
      try { socket.close() } catch { /* ignore */ }
    }
    globalConnected.value = false
    globalReconnecting.value = false
    globalFallbackMode.value = false
    globalSubscriptionAcknowledged.value = false
    latestResourceRevisions.clear()
    resourceRevisionOrder.clear()
    activeServerEpoch = null
    seenEventIds.clear()
    seenEventOrder.splice(0)
  })
}
