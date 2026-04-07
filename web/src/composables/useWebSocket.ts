import { ref, onUnmounted, type Ref } from 'vue'

type Topic = 'health' | 'cameras' | 'alerts' | 'tasks' | 'wall' | 'degradation' | 'heatmap'

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
const MAX_RETRIES = 3
const globalConnected = ref(false)

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
  if (retryCount > MAX_RETRIES) {
    // Fall back to per-subscriber polling
    for (const sub of subscribers.values()) {
      startFallbackPolling(sub)
    }
    return
  }
  const delay = Math.min(1000 * Math.pow(2, retryCount - 1), 30000)
  retryTimer = setTimeout(wsConnect, delay)
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
    fallbackInterval = 15000,
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
    error,
    topicData,
  }
}
