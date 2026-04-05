import { ref, onMounted, onUnmounted, type Ref } from 'vue'

type Topic = 'health' | 'cameras' | 'alerts' | 'tasks'

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
  /** Max reconnect retries before falling back to polling (default 3) */
  maxRetries?: number
}

function getCookie(name: string): string | null {
  const match = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`))
  return match ? decodeURIComponent(match[1]) : null
}

/**
 * Composable for subscribing to real-time WebSocket updates.
 *
 * Connects on mount, disconnects on unmount. Auto-reconnects with
 * exponential backoff. Falls back to polling after maxRetries failures.
 */
export function useWebSocket(options: UseWebSocketOptions) {
  const {
    topics,
    onMessage,
    fallbackPoll,
    fallbackInterval = 15000,
    maxRetries = 3,
  } = options

  const connected = ref(false)
  const error = ref<string | null>(null)

  // Reactive data per topic — consumers can read topicData.value[topic]
  const topicData: Ref<Record<string, any>> = ref({})

  let ws: WebSocket | null = null
  let retryCount = 0
  let retryTimer: ReturnType<typeof setTimeout> | null = null
  let fallbackTimer: ReturnType<typeof setInterval> | null = null
  let unmounted = false

  function connect() {
    if (unmounted) return

    const token = getCookie('argus_session')
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${proto}://${location.host}/ws${token ? `?token=${token}` : ''}`

    ws = new WebSocket(url)

    ws.onopen = () => {
      connected.value = true
      error.value = null
      retryCount = 0

      // Stop fallback polling if active
      if (fallbackTimer !== null) {
        clearInterval(fallbackTimer)
        fallbackTimer = null
      }

      // Subscribe to requested topics
      ws!.send(JSON.stringify({ action: 'subscribe', topics }))
    }

    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data)

        // Respond to heartbeat ping
        if (msg.topic === 'ping') {
          ws?.send(JSON.stringify({ action: 'pong' }))
          return
        }

        // Update reactive data
        if (topics.includes(msg.topic as Topic)) {
          topicData.value = { ...topicData.value, [msg.topic]: msg.data }
          onMessage?.(msg.topic as Topic, msg.data)
        }
      } catch {
        // ignore malformed messages
      }
    }

    ws.onclose = () => {
      connected.value = false
      ws = null
      scheduleReconnect()
    }

    ws.onerror = () => {
      error.value = 'WebSocket error'
      // onclose will fire after onerror, which triggers reconnect
    }
  }

  function scheduleReconnect() {
    if (unmounted) return

    retryCount++
    if (retryCount > maxRetries) {
      // Fall back to polling
      error.value = 'WebSocket unavailable, using polling'
      startFallbackPolling()
      return
    }

    // Exponential backoff: 1s, 2s, 4s, ... capped at 30s
    const delay = Math.min(1000 * Math.pow(2, retryCount - 1), 30000)
    retryTimer = setTimeout(connect, delay)
  }

  function startFallbackPolling() {
    if (!fallbackPoll || fallbackTimer !== null) return
    fallbackPoll() // immediate first poll
    fallbackTimer = setInterval(fallbackPoll, fallbackInterval)
  }

  function disconnect() {
    unmounted = true
    if (retryTimer !== null) {
      clearTimeout(retryTimer)
      retryTimer = null
    }
    if (fallbackTimer !== null) {
      clearInterval(fallbackTimer)
      fallbackTimer = null
    }
    if (ws) {
      ws.onclose = null // prevent reconnect on intentional close
      ws.close()
      ws = null
    }
    connected.value = false
  }

  onMounted(connect)
  onUnmounted(disconnect)

  return {
    /** Whether the WebSocket is currently connected */
    connected,
    /** Last error message, or null */
    error,
    /** Reactive object keyed by topic with latest data */
    topicData,
  }
}
