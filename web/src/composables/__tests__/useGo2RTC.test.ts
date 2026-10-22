import { defineComponent } from 'vue'
import { mount } from '@vue/test-utils'
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { useGo2RTC } from '../useGo2RTC'
import { StreamManager } from '../../services/streaming/StreamManager'
import { PlayerFactory } from '../../services/streaming/PlayerFactory'
import type { PlayerInstance } from '../../services/streaming/PlayerFactory'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: Error) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

function player(): PlayerInstance {
  return {
    status: 'playing', onStatusChange: vi.fn(), onReconnectRequest: vi.fn(),
    connect: vi.fn().mockResolvedValue(true), destroy: vi.fn(),
  }
}

function mountStream() {
  let stream!: ReturnType<typeof useGo2RTC>
  const wrapper = mount(defineComponent({
    setup() { stream = useGo2RTC('cam_01'); return stream },
    template: '<div>{{ status }}:{{ fallbackReason }}</div>',
  }))
  return { stream, wrapper }
}

describe('useGo2RTC', () => {
  const originalMaxStreams = StreamManager.MAX_STREAMS

  beforeEach(() => {
    StreamManager.activeCount = 0
    StreamManager.MAX_STREAMS = originalMaxStreams
  })

  afterEach(() => {
    vi.restoreAllMocks()
    StreamManager.activeCount = 0
    StreamManager.MAX_STREAMS = originalMaxStreams
  })

  it('falls back to MJPEG when the go2rtc connection budget is exhausted', async () => {
    StreamManager.MAX_STREAMS = 0
    const Harness = defineComponent({
      setup() {
        const stream = useGo2RTC('cam_01')
        return {
          start: stream.start,
          status: stream.status,
          reason: stream.fallbackReason,
        }
      },
      template: '<div>{{ status }}:{{ reason }}</div>',
    })

    const wrapper = mount(Harness)
    await (wrapper.vm as unknown as { start: () => Promise<void> }).start()

    expect(wrapper.text()).toBe('fallback:connection_budget_exhausted')
    expect(StreamManager.activeCount).toBe(0)

    wrapper.unmount()
  })

  it('falls back to MJPEG and releases budget when player creation fails', async () => {
    vi.spyOn(PlayerFactory, 'create').mockRejectedValue(new Error('go2rtc offline'))
    const Harness = defineComponent({
      setup() {
        const stream = useGo2RTC('cam_01')
        return {
          start: stream.start,
          status: stream.status,
          reason: stream.fallbackReason,
        }
      },
      template: '<div>{{ status }}:{{ reason }}</div>',
    })

    const wrapper = mount(Harness)
    await expect((wrapper.vm as unknown as { start: () => Promise<void> }).start()).resolves.toBeUndefined()

    expect(wrapper.text()).toBe('fallback:player_creation_exception')
    expect(StreamManager.activeCount).toBe(0)

    wrapper.unmount()
  })

  it.each(['null', 'candidate', 'failure'])('does not let a late old %s overwrite the new player or budget', async (outcome) => {
    const older = deferred<PlayerInstance | null>()
    const oldPlayer = player()
    const currentPlayer = player()
    const create = vi.spyOn(PlayerFactory, 'create')
      .mockReturnValueOnce(older.promise)
      .mockResolvedValueOnce(currentPlayer)
    const { stream, wrapper } = mountStream()

    const first = stream.start()
    expect(StreamManager.activeCount).toBe(1)
    const firstSignal = create.mock.calls[0]?.[4]
    await stream.start()
    expect(firstSignal?.aborted).toBe(true)
    expect(StreamManager.activeCount).toBe(1)
    create.mock.calls[0]?.[3]?.('webrtc_failed')

    if (outcome === 'failure') older.reject(new Error('old startup failed'))
    else older.resolve(outcome === 'candidate' ? oldPlayer : null)
    await first

    expect(stream.status.value).toBe('playing')
    expect(stream.fallbackReason.value).toBeNull()
    expect(StreamManager.activeCount).toBe(1)
    expect(currentPlayer.destroy).not.toHaveBeenCalled()
    expect(oldPlayer.destroy).toHaveBeenCalledTimes(outcome === 'candidate' ? 1 : 0)

    stream.stop()
    expect(currentPlayer.destroy).toHaveBeenCalledTimes(1)
    expect(StreamManager.activeCount).toBe(0)
    wrapper.unmount()
  })

  it('cancels pending creation before returning its budget and cleans a late candidate', async () => {
    const older = deferred<PlayerInstance | null>()
    const late = player()
    const order: string[] = []
    const release = StreamManager.releaseSlot.bind(StreamManager)
    vi.spyOn(StreamManager, 'releaseSlot').mockImplementation(() => { order.push('release'); release() })
    vi.spyOn(PlayerFactory, 'create').mockImplementation((_id, _video, _cancelled, _reason, signal) => {
      signal?.addEventListener('abort', () => order.push('abort'), { once: true })
      return older.promise
    })
    const { stream, wrapper } = mountStream()
    const pending = stream.start()
    stream.stop()

    expect(order).toEqual(['abort', 'release'])
    expect(StreamManager.activeCount).toBe(0)
    older.resolve(late)
    await pending
    expect(late.destroy).toHaveBeenCalledOnce()
    expect(stream.status.value).toBe('idle')
    expect(StreamManager.activeCount).toBe(0)
    wrapper.unmount()
  })
})
