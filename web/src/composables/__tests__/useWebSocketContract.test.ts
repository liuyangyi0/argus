import { mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

class MockWebSocket {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3
  static instances: MockWebSocket[] = []

  readonly url: string
  readyState = MockWebSocket.CONNECTING
  onopen: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  send = vi.fn()

  constructor(url: string | URL) {
    this.url = String(url)
    MockWebSocket.instances.push(this)
  }

  open(): void {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.(new Event('open'))
  }

  receive(payload: unknown): void {
    this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(payload) }))
  }

  close(): void {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.(new CloseEvent('close'))
  }
}

function event(
  eventId: string,
  topic: string,
  type: string,
  revision: number,
  data: unknown,
  resourceId = `${topic}:test`,
) {
  return {
    schema_version: 1,
    event_id: eventId,
    topic,
    type,
    resource_id: resourceId,
    revision,
    occurred_at: '2026-07-14T00:00:00Z',
    data,
  }
}

async function mountSocket(options: Record<string, unknown>) {
  const { useWebSocket } = await import('../useWebSocket')
  let socketState: ReturnType<typeof useWebSocket> | null = null
  const wrapper = mount(defineComponent({
    setup() {
      socketState = useWebSocket(
        options as unknown as Parameters<typeof useWebSocket>[0],
      )
      return () => h('div')
    },
  }))
  return { wrapper, state: () => socketState! }
}

describe('useWebSocket v1 contract', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    MockWebSocket.instances = []
    vi.stubGlobal('WebSocket', MockWebSocket)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('acknowledges subscriptions and drops duplicate event ids and revisions', async () => {
    const onMessage = vi.fn()
    const { wrapper, state } = await mountSocket({ topics: ['cameras'], onMessage })
    const socket = MockWebSocket.instances[0]!

    socket.open()
    expect(JSON.parse(String(socket.send.mock.calls[0]?.[0]))).toMatchObject({
      action: 'subscribe',
      topics: ['cameras'],
      revisions: {},
      resource_revisions: {},
    })

    socket.receive(event('ack-1', 'control', 'subscribe_ack', 0, {
      subscriptions: ['cameras'],
      revisions: { cameras: 0 },
      resource_revisions: { cameras: {} },
    }))
    expect(state().subscriptionAcknowledged.value).toBe(true)

    socket.receive(event('camera-1', 'cameras', 'camera.updated', 1, { camera_id: 'cam-01' }))
    socket.receive(event('camera-1', 'cameras', 'camera.updated', 1, { camera_id: 'cam-01' }))
    socket.receive(event('camera-stale', 'cameras', 'camera.updated', 1, { camera_id: 'cam-01' }))

    expect(onMessage).toHaveBeenCalledTimes(1)
    expect(onMessage).toHaveBeenCalledWith(
      'cameras',
      { camera_id: 'cam-01' },
      expect.objectContaining({ event_id: 'camera-1', revision: 1 }),
    )
    wrapper.unmount()
  })

  it('keeps 60 second REST reconciliation online and resyncs immediately on demand', async () => {
    const fallbackPoll = vi.fn()
    const { wrapper } = await mountSocket({ topics: ['alerts'], fallbackPoll })
    const socket = MockWebSocket.instances[0]!
    socket.open()
    socket.receive(event('ack-2', 'control', 'subscribe_ack', 0, {
      subscriptions: ['alerts'],
      revisions: { alerts: 0 },
      resource_revisions: { alerts: {} },
    }))

    await vi.advanceTimersByTimeAsync(60_000)
    expect(fallbackPoll).toHaveBeenCalledTimes(1)

    socket.receive(event('resync-1', 'control', 'resync_required', 0, {
      reason: 'revision_gap',
      topics: [{ topic: 'alerts', client_revision: 1, server_revision: 3 }],
    }))
    await Promise.resolve()
    expect(fallbackPoll).toHaveBeenCalledTimes(2)
    wrapper.unmount()
  })

  it('surfaces browser offline immediately and reconciles before reconnecting', async () => {
    const fallbackPoll = vi.fn()
    const { wrapper, state } = await mountSocket({ topics: ['cameras'], fallbackPoll })
    const firstSocket = MockWebSocket.instances[0]!
    firstSocket.open()

    window.dispatchEvent(new Event('offline'))

    expect(firstSocket.readyState).toBe(MockWebSocket.CLOSED)
    expect(state().connected.value).toBe(false)
    expect(state().reconnecting.value).toBe(true)

    window.dispatchEvent(new Event('online'))
    await Promise.resolve()

    expect(fallbackPoll).toHaveBeenCalledTimes(1)
    expect(MockWebSocket.instances).toHaveLength(2)
    MockWebSocket.instances[1]!.open()
    expect(state().connected.value).toBe(true)
    expect(state().reconnecting.value).toBe(false)
    wrapper.unmount()
  })

  it('ignores callbacks queued by a socket replaced during offline recovery', async () => {
    const onMessage = vi.fn()
    const { wrapper, state } = await mountSocket({ topics: ['cameras'], onMessage })
    const firstSocket = MockWebSocket.instances[0]!
    const staleOpen = firstSocket.onopen
    const staleMessage = firstSocket.onmessage
    const staleClose = firstSocket.onclose

    window.dispatchEvent(new Event('offline'))
    window.dispatchEvent(new Event('online'))
    const secondSocket = MockWebSocket.instances[1]!

    staleOpen?.(new Event('open'))
    staleMessage?.(new MessageEvent('message', {
      data: JSON.stringify(event(
        'stale-camera-event',
        'cameras',
        'camera.updated',
        1,
        { camera_id: 'stale' },
      )),
    }))
    staleClose?.(new CloseEvent('close'))

    expect(state().connected.value).toBe(false)
    expect(state().reconnecting.value).toBe(true)
    expect(secondSocket.send).not.toHaveBeenCalled()
    expect(onMessage).not.toHaveBeenCalled()

    secondSocket.open()
    expect(state().connected.value).toBe(true)
    expect(secondSocket.send).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })

  it('accepts a new low revision after subscribe ack starts a server restart epoch', async () => {
    const onMessage = vi.fn()
    const { wrapper } = await mountSocket({ topics: ['models'], onMessage })
    const socket = MockWebSocket.instances[0]!
    socket.open()
    socket.receive(event('ack-before-restart', 'control', 'subscribe_ack', 0, {
      server_epoch: 'epoch-before-restart',
      subscriptions: ['models'],
      revisions: { models: 0 },
      resource_revisions: { models: {} },
    }))
    socket.receive(event('model-old-epoch', 'models', 'model.updated', 5, { version: 'v5' }))

    socket.receive(event('ack-after-restart', 'control', 'subscribe_ack', 0, {
      server_epoch: 'epoch-after-restart',
      subscriptions: ['models'],
      revisions: { models: 0 },
      resource_revisions: { models: {} },
    }))
    socket.receive(event('model-new-epoch', 'models', 'model.updated', 1, { version: 'v1' }))

    expect(onMessage.mock.calls.map(call => call[1])).toEqual([
      { version: 'v5' },
      { version: 'v1' },
    ])
    wrapper.unmount()
  })

  it('clears old-epoch cursors for topics not subscribed during restart', async () => {
    const first = await mountSocket({ topics: ['alerts'] })
    const firstSocket = MockWebSocket.instances[0]!
    firstSocket.open()
    firstSocket.receive(event('old-alert-ack', 'control', 'subscribe_ack', 0, {
      server_epoch: 'epoch-old-hidden-topic',
      subscriptions: ['alerts'],
      resource_revisions: { alerts: { 'alert-hidden': 100 } },
    }))
    first.wrapper.unmount()

    const second = await mountSocket({ topics: ['cameras'] })
    const secondSocket = MockWebSocket.instances[1]!
    secondSocket.open()
    secondSocket.receive(event('new-camera-ack', 'control', 'subscribe_ack', 0, {
      server_epoch: 'epoch-new-hidden-topic',
      subscriptions: ['cameras'],
      resource_revisions: { cameras: {} },
    }))
    second.wrapper.unmount()

    const onMessage = vi.fn()
    const third = await mountSocket({ topics: ['alerts'], onMessage })
    const thirdSocket = MockWebSocket.instances[2]!
    thirdSocket.open()
    thirdSocket.receive(event('new-alert-ack', 'control', 'subscribe_ack', 0, {
      server_epoch: 'epoch-new-hidden-topic',
      subscriptions: ['alerts'],
      resource_revisions: { alerts: {} },
    }))
    thirdSocket.receive(event(
      'new-alert-r1',
      'alerts',
      'alert.updated',
      1,
      { alert_id: 'alert-hidden' },
      'alert-hidden',
    ))

    expect(onMessage).toHaveBeenCalledOnce()
    third.wrapper.unmount()
  })

  it('does not lower a resource cursor on a same-epoch ack after server eviction', async () => {
    const onMessage = vi.fn()
    const { wrapper } = await mountSocket({ topics: ['models'], onMessage })
    const socket = MockWebSocket.instances[0]!
    socket.open()
    socket.receive(event('same-epoch-ack-1', 'control', 'subscribe_ack', 0, {
      server_epoch: 'stable-epoch',
      subscriptions: ['models'],
      resource_revisions: { models: { 'model-1': 4 } },
    }))
    socket.receive(event(
      'same-epoch-model-r5',
      'models',
      'model.updated',
      5,
      { version: 'v5' },
      'model-1',
    ))

    socket.receive(event('same-epoch-ack-2', 'control', 'subscribe_ack', 0, {
      server_epoch: 'stable-epoch',
      subscriptions: ['models'],
      resource_revisions: { models: {} },
    }))
    socket.receive(event(
      'same-epoch-model-reset-r1',
      'models',
      'model.updated',
      1,
      { version: 'reset-v1' },
      'model-1',
    ))

    expect(onMessage.mock.calls.map(call => call[1])).toEqual([{ version: 'v5' }])
    wrapper.unmount()
  })

  it('orders revisions independently for different resources in one topic', async () => {
    const onMessage = vi.fn()
    const { wrapper } = await mountSocket({ topics: ['cameras'], onMessage })
    const socket = MockWebSocket.instances[0]!
    socket.open()
    socket.receive(event('ack-resources', 'control', 'subscribe_ack', 0, {
      subscriptions: ['cameras'],
      revisions: { cameras: 0 },
      resource_revisions: { cameras: {} },
    }))

    socket.receive(event('cam-01-r1', 'cameras', 'camera.updated', 1, { camera_id: 'cam-01' }, 'cam-01'))
    socket.receive(event('cam-02-r1', 'cameras', 'camera.updated', 1, { camera_id: 'cam-02' }, 'cam-02'))
    socket.receive(event('cam-01-stale', 'cameras', 'camera.updated', 1, { camera_id: 'cam-01' }, 'cam-01'))

    expect(onMessage.mock.calls.map(call => call[1])).toEqual([
      { camera_id: 'cam-01' },
      { camera_id: 'cam-02' },
    ])
    wrapper.unmount()
  })

  it('ignores unversioned and incomplete payloads instead of bypassing v1 validation', async () => {
    const onMessage = vi.fn()
    const { wrapper } = await mountSocket({ topics: ['alerts'], onMessage })
    const socket = MockWebSocket.instances[0]!
    socket.open()

    socket.receive({ topic: 'alerts', data: { alert_id: 'legacy' } })
    socket.receive({
      schema_version: 1,
      event_id: 'incomplete',
      topic: 'alerts',
      type: 'alert.created',
      data: { alert_id: 'missing-resource-fields' },
    })

    expect(onMessage).not.toHaveBeenCalled()
    wrapper.unmount()
  })
})
