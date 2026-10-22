import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import ReplaySignalTracks from '../ReplaySignalTracks.vue'

function mountTracks(overrides: Record<string, any> = {}) {
  const seekTo = vi.fn()
  const ctrl = {
    metadata: ref({ severity: 'medium' }),
    signals: ref({
      timestamps: [1, 2, 3, 4],
      anomaly_scores: [0.1, 0.2, 0.9, 0.3],
      simplex_scores: [null, null, null, null],
      cusum_evidence: {},
      yolo_persons: [],
      candidate_events: [
        {
          frame_index: 2,
          candidate_id: 'cam:2:1',
          candidate_type: 'persistent_change',
          decision: 'candidate',
          suppression_reason: null,
          evidence: {
            score: 0.87,
          },
        },
        {
          frame_index: 3,
          candidate_id: 'cam:3:1',
          candidate_type: 'frame_anomaly',
          decision: 'info',
          suppression_reason: 'grader_no_alert',
          evidence: {
            anomaly_score: 0.91,
          },
        },
      ],
      ...overrides.signals,
    }),
    currentIndex: ref(overrides.currentIndex ?? 2),
    seekTo,
    handlePinFrame: vi.fn(),
  }

  const wrapper = mount(ReplaySignalTracks, {
    global: {
      provide: { replayCtrl: ctrl },
      stubs: {
        SignalTrack: {
          props: ['label'],
          template: '<div data-test="signal-track">{{ label }}</div>',
        },
      },
    },
  })
  return { wrapper, seekTo }
}

describe('ReplaySignalTracks candidate evidence', () => {
  it('renders nearby candidate evidence and seeks to the selected frame', async () => {
    const { wrapper, seekTo } = mountTracks()

    const panel = wrapper.find('[data-test="replay-candidates"]')
    expect(panel.exists()).toBe(true)
    expect(panel.text()).toContain('候选证据')
    expect(panel.text()).toContain('固定场景变化')
    expect(panel.text()).toContain('候选')
    expect(panel.text()).toContain('0.870')
    expect(panel.text()).toContain('帧异常')
    expect(panel.text()).toContain('grader_no_alert')

    await panel.findAll('button')[0].trigger('click')
    expect(seekTo).toHaveBeenCalledWith(2)
  })

  it('falls back to per-frame candidate records when flat events are absent', () => {
    const { wrapper } = mountTracks({
      signals: {
        candidate_events: [],
        candidate_records: [
          [],
          [
            {
              candidate_id: 'cam:1:1',
              candidate_type: 'fast_motion',
              decision: 'alert',
              evidence: { score: 0.96 },
            },
          ],
        ],
      },
      currentIndex: 1,
    })

    const panel = wrapper.find('[data-test="replay-candidates"]')
    expect(panel.text()).toContain('高速飞溅')
    expect(panel.text()).toContain('告警')
  })
})
