import { defineComponent } from 'vue'
import { mount } from '@vue/test-utils'
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { useGo2RTC } from '../useGo2RTC'
import { StreamManager } from '../../services/streaming/StreamManager'
import { PlayerFactory } from '../../services/streaming/PlayerFactory'

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
        }
      },
      template: '<div>{{ status }}</div>',
    })

    const wrapper = mount(Harness)
    await (wrapper.vm as unknown as { start: () => Promise<void> }).start()

    expect(wrapper.text()).toBe('fallback')
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
        }
      },
      template: '<div>{{ status }}</div>',
    })

    const wrapper = mount(Harness)
    await expect((wrapper.vm as unknown as { start: () => Promise<void> }).start()).resolves.toBeUndefined()

    expect(wrapper.text()).toBe('fallback')
    expect(StreamManager.activeCount).toBe(0)

    wrapper.unmount()
  })
})
