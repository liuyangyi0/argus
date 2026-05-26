import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import Overview from '../Overview.vue'
import { getWallStatus, getHealth } from '../../api'
import { getDailyTrend } from '../../api/reports'
import { getTrainingJobs } from '../../api/training'
import { getAnomalyDegradation } from '../../api/system'
import { useAuthStore } from '../../stores/useAuthStore'

const routerPush = vi.hoisted(() => vi.fn())

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPush }),
}))

vi.mock('../../composables/useWebSocket', () => ({
  useWebSocket: vi.fn(),
}))

vi.mock('../../api', () => ({
  getWallStatus: vi.fn(),
  getHealth: vi.fn(),
}))

vi.mock('../../api/reports', () => ({
  getDailyTrend: vi.fn(),
}))

vi.mock('../../api/training', () => ({
  getTrainingJobs: vi.fn(),
}))

vi.mock('../../api/system', () => ({
  getAnomalyDegradation: vi.fn(),
}))

const alertStub = {
  template: '<div><slot name="message" /><slot name="description" /></div>',
}

const buttonStub = {
  props: ['type', 'size'],
  emits: ['click'],
  template: '<button type="button" @click="$emit(\'click\', $event)"><slot /></button>',
}

let pinia: ReturnType<typeof createPinia>

describe('Overview active alert links', () => {
  beforeEach(() => {
    pinia = createPinia()
    setActivePinia(pinia)
    useAuthStore().currentUser = { username: 'engineer', role: 'engineer' }
    routerPush.mockReset()
    vi.mocked(getWallStatus).mockReset()
    vi.mocked(getHealth).mockReset()
    vi.mocked(getDailyTrend).mockReset()
    vi.mocked(getTrainingJobs).mockReset()
    vi.mocked(getAnomalyDegradation).mockReset()
    vi.mocked(getWallStatus).mockResolvedValue({
      cameras: [
        {
          camera_id: 'cam_01',
          name: 'Camera 01',
          status: 'online',
          current_score: 0.82,
          score_sparkline: [0.1, 0.82],
          alert_count_today: 1,
          active_alert: {
            alert_id: 'ALT-active-1',
            severity: 'high',
            anomaly_score: 0.82,
            timestamp: Date.now() / 1000,
          },
          degradation: null,
        },
      ],
    } as any)
    vi.mocked(getHealth).mockResolvedValue({
      status: 'healthy',
      uptime_seconds: 120,
      cameras: [
        {
          camera_id: 'cam_01',
          connected: true,
          frames_captured: 123,
          avg_latency_ms: 20,
        },
      ],
    } as any)
    vi.mocked(getDailyTrend).mockResolvedValue({
      labels: ['05-25'],
      high: [1],
      medium: [0],
      low: [0],
      info: [0],
    })
    vi.mocked(getTrainingJobs).mockResolvedValue({ pending_count: 0 } as any)
    vi.mocked(getAnomalyDegradation).mockResolvedValue({
      anomaly: { degraded: false, reason: null, since: null, cameras: [] },
    } as any)
  })

  it('opens the concrete alert detail from the right-side active alert card', async () => {
    const wrapper = mount(Overview, {
      global: {
        plugins: [pinia],
        mocks: {
          $router: { push: routerPush },
        },
        stubs: {
          'a-alert': alertStub,
          'a-button': buttonStub,
          ContentSkeleton: true,
        },
      },
    })

    await flushPromises()
    await wrapper.find('.a-btn').trigger('click')

    expect(routerPush).toHaveBeenCalledWith('/alerts?id=ALT-active-1')

    wrapper.unmount()
  })

  it('shows pending training jobs only to roles that can manage training', async () => {
    vi.mocked(getTrainingJobs).mockResolvedValue({ pending_count: 2 } as any)

    const wrapper = mount(Overview, {
      global: {
        plugins: [pinia],
        mocks: {
          $router: { push: routerPush },
        },
        stubs: {
          'a-alert': alertStub,
          'a-button': buttonStub,
          ContentSkeleton: true,
        },
      },
    })

    await flushPromises()

    expect(wrapper.text()).toContain('2 个训练任务待确认')
    const trainingButton = wrapper.findAll('button').find(button => button.text().includes('训练与评估'))
    expect(trainingButton).toBeTruthy()
    await trainingButton!.trigger('click')

    expect(routerPush).toHaveBeenCalledWith({
      path: '/models/training',
      query: { tab: 'pending' },
    })

    wrapper.unmount()
  })

  it('does not surface training-management links to operators', async () => {
    pinia = createPinia()
    setActivePinia(pinia)
    useAuthStore().currentUser = { username: 'operator', role: 'operator' }
    vi.mocked(getTrainingJobs).mockResolvedValue({ pending_count: 2 } as any)

    const wrapper = mount(Overview, {
      global: {
        plugins: [pinia],
        mocks: {
          $router: { push: routerPush },
        },
        stubs: {
          'a-alert': alertStub,
          'a-button': buttonStub,
          ContentSkeleton: true,
        },
      },
    })

    await flushPromises()

    expect(getTrainingJobs).not.toHaveBeenCalled()
    expect(wrapper.text()).not.toContain('训练任务待确认')
    expect(wrapper.text()).not.toContain('训练与评估')

    wrapper.unmount()
  })
})
