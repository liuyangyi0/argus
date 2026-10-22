import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import SceneWorkflowPanel from '../SceneWorkflowPanel.vue'
import {
  activateSceneProfile,
  approveSceneProfile,
  createSceneTrainingJob,
  createSceneWorkflow,
  getSceneWorkflow,
  startSceneBaselineJob,
} from '../../../api/sceneWorkflow'
import type { SceneWorkflow, SceneWorkflowSnapshot } from '../../../api/sceneWorkflow'
import { ApiError, type CameraSummary } from '../../../types/api'
import { useWebSocket } from '../../../composables/useWebSocket'

const controls = vi.hoisted(() => ({
  canManage: true,
  confirm: vi.fn(),
  fetchCurrentUser: vi.fn().mockResolvedValue(null),
}))

vi.mock('ant-design-vue', async importOriginal => ({
  ...await importOriginal<typeof import('ant-design-vue')>(),
  Button: {
    props: ['disabled', 'loading'],
    emits: ['click'],
    template: '<button type="button" :disabled="disabled" @click="$emit(\'click\')"><slot /></button>',
  },
  Card: {
    props: ['title'],
    template: '<section><h2>{{ title }}</h2><slot /></section>',
  },
  Modal: { confirm: controls.confirm },
}))

vi.mock('../../../composables/useWebSocket', () => ({ useWebSocket: vi.fn() }))

vi.mock('../../../stores/useAuthStore', () => ({
  useAuthStore: () => ({
    hasRole: () => controls.canManage,
    fetchCurrentUser: controls.fetchCurrentUser,
  }),
}))

vi.mock('../../../api/sceneWorkflow', () => ({
  getSceneWorkflow: vi.fn(),
  createSceneWorkflow: vi.fn(),
  startSceneBaselineJob: vi.fn(),
  createSceneTrainingJob: vi.fn(),
  approveSceneProfile: vi.fn(),
  activateSceneProfile: vi.fn(),
}))

const camera = (id: string): CameraSummary => ({
  camera_id: id, name: id, connected: true, running: true, stats: null,
})

function workflow(overrides: Partial<SceneWorkflow> = {}): SceneWorkflow {
  return {
    workflow_id: 'scene-a', profile_id: 'scene-a', state: 'baseline_ready',
    created_by: 'engineer', created_at: '2026-10-22T00:00:00Z',
    reference_sha256: 'a'.repeat(64), baseline_schema: 'argus.scene-profile-baseline.v3',
    training_ready: true, train_count: 100, calibration_count: 50,
    baseline_task_id: 'baseline-1', training_job_id: null, training_status: null,
    model_version_id: null, approved: false, active: false, error: null,
    blockers: ['独立回归和可信 72 小时 soak 尚未完成'],
    ...overrides,
  }
}

function snapshot(overrides: Partial<SceneWorkflowSnapshot> = {}): SceneWorkflowSnapshot {
  return {
    camera_id: 'cam1', workflows: [workflow()], profiles: [],
    references: [{ path: 'cam1/default/v001/reference.png', sha256: 'a'.repeat(64) }],
    source_root_label: 'data/regression',
    active_profile_revision: null,
    requirements: {
      baseline_schema: 'argus.scene-profile-baseline.v3', min_sources: 2,
      min_train_images: 100, min_calibration_images: 50, production_gate_unchanged: true,
    },
    ...overrides,
  }
}

function mountPanel(cameras = [camera('cam1')]) {
  return mount(SceneWorkflowPanel, {
    props: { cameras },
    global: {
      stubs: {
        RouterLink: { props: ['to'], template: '<a :href="to"><slot /></a>' },
        AsyncState: {
          props: ['loading', 'error', 'forbidden', 'notFound', 'empty'],
          emits: ['retry'],
          template: '<div><p v-if="loading">加载中</p><p v-else-if="forbidden">无权访问</p><p v-else-if="notFound">不存在</p><p v-else-if="error">{{ error }}</p><slot v-else-if="!empty" /></div>',
        },
        StatusBanner: {
          props: ['title', 'description'],
          template: '<aside>{{ title }} {{ description }}<slot name="action" /></aside>',
        },
      },
    },
  })
}

describe('SceneWorkflowPanel controlled workflow', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-10-22T00:00:00Z'))
    vi.resetAllMocks()
    controls.canManage = true
    controls.fetchCurrentUser.mockResolvedValue(null)
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot())
    vi.mocked(createSceneWorkflow).mockResolvedValue(workflow({ workflow_id: 'scene-b', state: 'draft' }))
    vi.mocked(startSceneBaselineJob).mockResolvedValue({ task_id: 'task-2' })
    vi.mocked(createSceneTrainingJob).mockResolvedValue({ job_id: 'training-2', status: 'pending_confirmation' })
    vi.mocked(approveSceneProfile).mockResolvedValue({ profile_id: 'scene-a' })
    vi.mocked(activateSceneProfile).mockResolvedValue({ profile_id: 'scene-a' })
  })

  afterEach(() => { vi.useRealTimers() })

  it('requires an explicit normal-reference confirmation and never approves on creation', async () => {
    const wrapper = mountPanel()
    await flushPromises()
    await wrapper.get('[data-test="scene-draft-id"]').setValue('scene-b')
    expect(wrapper.get('[data-test="scene-create"]').attributes('disabled')).toBeDefined()
    await wrapper.get('[data-test="scene-reference-confirmed"]').setValue(true)
    await wrapper.get('[data-test="scene-create"]').trigger('click')
    await flushPromises()
    expect(createSceneWorkflow).toHaveBeenCalledWith('cam1', {
      workflow_id: 'scene-b', reference_path: 'cam1/default/v001/reference.png',
      reference_sha256: 'a'.repeat(64), normal_reference_confirmed: true,
    })
    expect(approveSceneProfile).not.toHaveBeenCalled()
    expect(activateSceneProfile).not.toHaveBeenCalled()
    expect(createSceneTrainingJob).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('尚未批准、激活或发布')
    wrapper.unmount()
  })

  it('does not clear another camera draft when a successful create is still reconciling the old camera', async () => {
    let resolveFollowup!: (value: SceneWorkflowSnapshot) => void
    const oldFollowup = new Promise<SceneWorkflowSnapshot>(resolve => { resolveFollowup = resolve })
    vi.mocked(getSceneWorkflow)
      .mockResolvedValueOnce(snapshot())
      .mockReturnValueOnce(oldFollowup)
      .mockImplementation(async id => snapshot({
        camera_id: id,
        workflows: [workflow({ workflow_id: 'camera-two-workflow' })],
        references: [{ path: 'camera-two-reference.png', sha256: 'b'.repeat(64) }],
      }))
    const wrapper = mountPanel([camera('cam1'), camera('cam2')])
    await flushPromises()
    await wrapper.get('[data-test="scene-draft-id"]').setValue('camera-one-created')
    await wrapper.get('[data-test="scene-reference-confirmed"]').setValue(true)
    await wrapper.get('[data-test="scene-create"]').trigger('click')
    await flushPromises()
    expect(getSceneWorkflow).toHaveBeenCalledTimes(2)

    await wrapper.get('[data-test="scene-camera"]').setValue('cam2')
    await flushPromises()
    await wrapper.get('[data-test="scene-draft-id"]').setValue('camera-two-unsaved')
    await wrapper.get('[data-test="scene-reference-confirmed"]').setValue(true)
    resolveFollowup(snapshot({ workflows: [workflow({ workflow_id: 'camera-one-created' })] }))
    await flushPromises()

    expect((wrapper.get('[data-test="scene-draft-id"]').element as HTMLInputElement).value).toBe('camera-two-unsaved')
    expect((wrapper.get('[data-test="scene-reference-confirmed"]').element as HTMLInputElement).checked).toBe(true)
    expect((wrapper.get('[data-test="scene-workflow-select"]').element as HTMLSelectElement).value).toBe('camera-two-workflow')
    wrapper.unmount()
  })

  it('submits only explicit reviewed source paths without decoder or gate overrides', async () => {
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({ workflows: [workflow({ state: 'draft' })] }))
    const wrapper = mountPanel()
    await flushPromises()
    expect(wrapper.get('[data-test="scene-build"]').attributes('disabled')).toBeDefined()
    for (const index of [0, 1]) {
      await wrapper.get(`[data-test="scene-source-video-${index}"]`).setValue(`normal/session-${index}.mp4`)
      await wrapper.get(`[data-test="scene-source-annotations-${index}"]`).setValue(`normal/session-${index}.json`)
    }
    await wrapper.get('[data-test="scene-build"]').trigger('click')
    await flushPromises()
    expect(startSceneBaselineJob).toHaveBeenCalledWith('cam1', 'scene-a', {
      sources: [
        { video: 'normal/session-0.mp4', annotations: 'normal/session-0.json' },
        { video: 'normal/session-1.mp4', annotations: 'normal/session-1.json' },
      ], sample_every: 30, max_images: 2000,
    })
    expect(approveSceneProfile).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('requires a new manual confirmation when the reference bytes change at the same path', async () => {
    const wrapper = mountPanel()
    await flushPromises()
    await wrapper.get('[data-test="scene-draft-id"]').setValue('scene-b')
    await wrapper.get('[data-test="scene-reference-confirmed"]').setValue(true)
    expect(wrapper.get('[data-test="scene-create"]').attributes('disabled')).toBeUndefined()
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({
      references: [{ path: 'cam1/default/v001/reference.png', sha256: 'b'.repeat(64) }],
    }))
    await vi.advanceTimersByTimeAsync(60_000)
    await flushPromises()
    expect(wrapper.get('[data-test="scene-create"]').attributes('disabled')).toBeDefined()
    expect((wrapper.get('[data-test="scene-reference-confirmed"]').element as HTMLInputElement).checked).toBe(false)
    expect(createSceneWorkflow).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it.each([
    { baseline_schema: 'argus.scene-profile-baseline.v2' },
    { calibration_count: 49 },
    { training_ready: false },
  ])('keeps training disabled for a non-compliant baseline: %j', async overrides => {
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({ workflows: [workflow(overrides)] }))
    const wrapper = mountPanel()
    await flushPromises()
    expect(wrapper.get('[data-test="scene-train"]').attributes('disabled')).toBeDefined()
    expect(createSceneTrainingJob).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('creates a pending training job without confirming, approving, or publishing it', async () => {
    const wrapper = mountPanel()
    await flushPromises()
    await wrapper.get('[data-test="scene-train"]').trigger('click')
    await flushPromises()
    expect(createSceneTrainingJob).toHaveBeenCalledWith('cam1', 'scene-a', {
      model_type: 'patchcore', image_size: 256,
    })
    expect(approveSceneProfile).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('待人工确认')
    wrapper.unmount()
  })

  it.each(['failed', 'rejected'])('allows only an explicit training retry after a known %s result', async status => {
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({
      workflows: [workflow({ state: 'training_submitted', training_job_id: 'previous-job', training_status: status })],
    }))
    const wrapper = mountPanel()
    await flushPromises()
    expect(createSceneTrainingJob).not.toHaveBeenCalled()
    expect(wrapper.get('[data-test="scene-train"]').text()).toContain('显式重试')
    expect(wrapper.text()).toContain('同一冻结基线')
    await wrapper.get('[data-test="scene-train"]').trigger('click')
    await flushPromises()
    expect(createSceneTrainingJob).toHaveBeenCalledWith('cam1', 'scene-a', {
      model_type: 'patchcore', image_size: 256,
    })
    expect(approveSceneProfile).not.toHaveBeenCalled()
    expect(activateSceneProfile).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it.each([null, 'complete', 'pending_confirmation', 'queued', 'running'])('never duplicates a training job whose current result is %s', async status => {
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({
      workflows: [workflow({ state: 'training_submitted', training_job_id: 'prior-job', training_status: status })],
    }))
    const wrapper = mountPanel()
    await flushPromises()
    expect(wrapper.get('[data-test="scene-train"]').attributes('disabled')).toBeDefined()
    expect(createSceneTrainingJob).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('isolates damaged workflow notices without granting them authority or blocking valid draft creation', async () => {
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({
      workflows: [],
      issues: [{ workflow_id: 'orphan-draft', message: '缺少记录，必须人工核查' }],
    }))
    const wrapper = mountPanel()
    await flushPromises()
    expect(wrapper.text()).toContain('部分工作流记录缺失或损坏')
    expect(wrapper.text()).toContain('orphan-draft')
    expect(wrapper.get('[data-test="scene-workflow-select"]').findAll('option')).toHaveLength(0)
    await wrapper.get('[data-test="scene-draft-id"]').setValue('new-valid-draft')
    await wrapper.get('[data-test="scene-reference-confirmed"]').setValue(true)
    expect(wrapper.get('[data-test="scene-create"]').attributes('disabled')).toBeUndefined()
    expect(createSceneWorkflow).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('requires a second human confirmation for a completed model binding and never activates it', async () => {
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({
      workflows: [workflow({ training_status: 'complete', training_job_id: 'training-1', model_version_id: 'model-1' })],
    }))
    const wrapper = mountPanel()
    await flushPromises()
    await wrapper.get('[data-test="scene-approve"]').trigger('click')
    expect(approveSceneProfile).not.toHaveBeenCalled()
    const options = controls.confirm.mock.calls[0]?.[0] as { onOk: () => Promise<void>; content: string }
    expect(options.content).toContain('不会激活场景')
    await options.onOk()
    await flushPromises()
    expect(approveSceneProfile).toHaveBeenCalledWith('cam1', 'scene-a')
    expect(activateSceneProfile).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('活动场景和旧模型保持不变')
    wrapper.unmount()
  })

  it('selects an approved scene only after a separate confirmation and does not claim runtime recovery', async () => {
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({
      workflows: [workflow({ approved: true, training_status: 'complete', model_version_id: 'model-1' })],
    }))
    const wrapper = mountPanel()
    await flushPromises()
    await wrapper.get('[data-test="scene-select-profile"]').trigger('click')
    expect(activateSceneProfile).not.toHaveBeenCalled()
    const options = controls.confirm.mock.calls[0]?.[0] as { onOk: () => Promise<void>; content: string }
    expect(options.content).toContain('不会发布模型或恢复评分')
    expect(options.content).toContain('身份不匹配会停检')
    await options.onOk()
    await flushPromises()
    expect(activateSceneProfile).toHaveBeenCalledWith('cam1', 'scene-a', null, null)
    expect(wrapper.text()).toContain('未切换运行模型、恢复评分或发布')
    wrapper.unmount()
  })

  it('keeps scene selection disabled until approval and when the profile is already selected', async () => {
    const wrapper = mountPanel()
    await flushPromises()
    expect(wrapper.get('[data-test="scene-select-profile"]').attributes('disabled')).toBeDefined()
    wrapper.unmount()
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({ workflows: [workflow({ approved: true, active: true })] }))
    const selected = mountPanel()
    await flushPromises()
    expect(selected.get('[data-test="scene-select-profile"]').attributes('disabled')).toBeDefined()
    expect(activateSceneProfile).not.toHaveBeenCalled()
    selected.unmount()
  })

  it('does not apply a delayed approval confirmation to another camera', async () => {
    vi.mocked(getSceneWorkflow).mockImplementation(async id => snapshot({
      camera_id: id,
      workflows: [workflow({ training_status: 'complete', training_job_id: 'training-1', model_version_id: 'model-1' })],
    }))
    const wrapper = mountPanel([camera('cam1'), camera('cam2')])
    await flushPromises()
    await wrapper.get('[data-test="scene-approve"]').trigger('click')
    const options = controls.confirm.mock.calls[0]?.[0] as { onOk: () => Promise<void> }
    await wrapper.get('[data-test="scene-camera"]').setValue('cam2')
    await flushPromises()
    await options.onOk()
    expect(approveSceneProfile).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('rejects an approval confirmation after the known model identity changes', async () => {
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({
      workflows: [workflow({ training_status: 'complete', training_job_id: 'training-1', model_version_id: 'model-1' })],
    }))
    const wrapper = mountPanel()
    await flushPromises()
    await wrapper.get('[data-test="scene-approve"]').trigger('click')
    const options = controls.confirm.mock.calls[0]?.[0] as { onOk: () => Promise<void> }
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({
      workflows: [workflow({ training_status: 'complete', training_job_id: 'training-2', model_version_id: 'model-2' })],
    }))
    vi.mocked(useWebSocket).mock.calls[0]?.[0]?.onMessage?.('models', { camera_id: 'cam1' })
    await flushPromises()
    await options.onOk()
    expect(approveSceneProfile).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('审批目标或状态已变化')
    wrapper.unmount()
  })

  it('rejects an old scene-selection confirmation after another active profile is observed', async () => {
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({
      workflows: [workflow({ approved: true, training_status: 'complete', model_version_id: 'model-1' })],
      profiles: [{ profile_id: 'previous-active', approved: true, active: true, model_version_id: 'old-model' }],
      active_profile_revision: 'a'.repeat(64),
    }))
    const wrapper = mountPanel()
    await flushPromises()
    await wrapper.get('[data-test="scene-select-profile"]').trigger('click')
    const options = controls.confirm.mock.calls[0]?.[0] as { onOk: () => Promise<void> }
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({
      workflows: [workflow({ approved: true, training_status: 'complete', model_version_id: 'model-1' })],
      profiles: [{ profile_id: 'newly-selected', approved: true, active: true, model_version_id: 'other-model' }],
      active_profile_revision: 'b'.repeat(64),
    }))
    vi.mocked(useWebSocket).mock.calls[0]?.[0]?.onMessage?.('models', { camera_id: 'cam1' })
    await flushPromises()
    await options.onOk()
    expect(activateSceneProfile).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('所选场景或当前活动配置已变化')
    wrapper.unmount()
  })

  it('rejects an old scene-selection confirmation after an observed A to B to A revision change', async () => {
    const data = snapshot({
      workflows: [workflow({ approved: true, training_status: 'complete', model_version_id: 'model-1' })],
      profiles: [{ profile_id: 'active-a', approved: true, active: true, model_version_id: 'old-model' }],
      active_profile_revision: 'a'.repeat(64),
    })
    vi.mocked(getSceneWorkflow).mockResolvedValue(data)
    const wrapper = mountPanel()
    await flushPromises()
    await wrapper.get('[data-test="scene-select-profile"]').trigger('click')
    const options = controls.confirm.mock.calls[0]?.[0] as { onOk: () => Promise<void> }
    vi.mocked(getSceneWorkflow).mockResolvedValue({ ...data, active_profile_revision: 'c'.repeat(64) })
    vi.mocked(useWebSocket).mock.calls[0]?.[0]?.onMessage?.('models', { camera_id: 'cam1' })
    await flushPromises()
    await options.onOk()
    expect(activateSceneProfile).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('所选场景或当前活动配置已变化')
    wrapper.unmount()
  })

  it('binds scene selection to the exact observed legacy active profile ID and revision', async () => {
    const legacyId = 'a'.repeat(64)
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({
      workflows: [workflow({ approved: true, training_status: 'complete', model_version_id: 'model-1' })],
      profiles: [{ profile_id: legacyId, approved: true, active: true, model_version_id: 'old-model' }],
      active_profile_revision: 'b'.repeat(64),
    }))
    const wrapper = mountPanel()
    await flushPromises()
    await wrapper.get('[data-test="scene-select-profile"]').trigger('click')
    const options = controls.confirm.mock.calls[0]?.[0] as { onOk: () => Promise<void> }
    await options.onOk()
    expect(activateSceneProfile).toHaveBeenCalledWith('cam1', 'scene-a', legacyId, 'b'.repeat(64))
    wrapper.unmount()
  })

  it('keeps selection disabled when an old server omits the active pointer revision', async () => {
    const legacy = snapshot({ workflows: [workflow({ approved: true, training_status: 'complete', model_version_id: 'model-1' })] })
    delete (legacy as Partial<SceneWorkflowSnapshot>).active_profile_revision
    vi.mocked(getSceneWorkflow).mockResolvedValue(legacy)
    const wrapper = mountPanel()
    await flushPromises()
    expect(wrapper.get('[data-test="scene-select-profile"]').attributes('disabled')).toBeDefined()
    expect(wrapper.text()).toContain('活动场景版本未知，选择操作保持禁用')
    expect(activateSceneProfile).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('preserves last known data but blocks writes after a reconciliation failure', async () => {
    const wrapper = mountPanel()
    await flushPromises()
    vi.mocked(getSceneWorkflow).mockRejectedValue(new Error('network unavailable'))
    await vi.advanceTimersByTimeAsync(60_000)
    await flushPromises()
    expect(wrapper.text()).toContain('当前为上次数据，所有写入操作已禁用')
    expect(wrapper.text()).toContain('scene-a')
    expect(wrapper.get('[data-test="scene-train"]').attributes('disabled')).toBeDefined()
    expect(createSceneTrainingJob).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('polls active work every five seconds and clears its timer when unmounted', async () => {
    vi.mocked(getSceneWorkflow).mockResolvedValue(snapshot({ workflows: [workflow({ state: 'building_baseline' })] }))
    const wrapper = mountPanel()
    await flushPromises()
    await vi.advanceTimersByTimeAsync(5000)
    await flushPromises()
    expect(getSceneWorkflow).toHaveBeenCalledTimes(2)
    wrapper.unmount()
    await vi.advanceTimersByTimeAsync(60_000)
    expect(getSceneWorkflow).toHaveBeenCalledTimes(2)
  })

  it('reconciles workflow mutation events and retains REST fallback while the socket is unavailable', async () => {
    const wrapper = mountPanel()
    await flushPromises()
    const options = vi.mocked(useWebSocket).mock.calls[0]?.[0]
    expect(options?.topics).toEqual(['models', 'model_release', 'tasks'])
    expect(options?.fallbackInterval).toBe(60_000)
    expect(options?.reconcileInterval).toBe(60_000)
    options?.onMessage?.('models', { camera_id: 'other-camera' })
    expect(getSceneWorkflow).toHaveBeenCalledTimes(1)
    options?.onMessage?.('models', { camera_id: 'cam1' })
    await flushPromises()
    expect(getSceneWorkflow).toHaveBeenCalledTimes(2)
    await options?.fallbackPoll?.()
    expect(getSceneWorkflow).toHaveBeenCalledTimes(3)
    wrapper.unmount()
  })

  it('does not replace another camera snapshot with a late response', async () => {
    let resolveOld: ((value: SceneWorkflowSnapshot) => void) | undefined
    vi.mocked(getSceneWorkflow).mockImplementation(id => {
      if (id === 'cam1') return new Promise(resolve => { resolveOld = resolve })
      return Promise.resolve(snapshot({ camera_id: id, workflows: [workflow({ workflow_id: 'new-camera-workflow' })] }))
    })
    const wrapper = mountPanel([camera('cam1'), camera('cam2')])
    await wrapper.get('[data-test="scene-camera"]').setValue('cam2')
    await flushPromises()
    resolveOld?.(snapshot({ workflows: [workflow({ workflow_id: 'old-camera-workflow' })] }))
    await flushPromises()
    expect(wrapper.text()).toContain('new-camera-workflow')
    expect(wrapper.text()).not.toContain('old-camera-workflow')
    wrapper.unmount()
  })

  it('fails closed for a viewer and a forbidden response', async () => {
    controls.canManage = false
    const viewer = mountPanel()
    await flushPromises()
    expect(viewer.text()).toContain('当前账号只可查看')
    expect(viewer.get('[data-test="scene-train"]').attributes('disabled')).toBeDefined()
    viewer.unmount()

    vi.mocked(getSceneWorkflow).mockRejectedValue(new ApiError({ httpStatus: 403, message: 'forbidden' }))
    const forbidden = mountPanel()
    await flushPromises()
    expect(forbidden.text()).toContain('无权访问')
    expect(forbidden.find('[data-test="scene-create"]').exists()).toBe(false)
    expect(createSceneWorkflow).not.toHaveBeenCalled()
    forbidden.unmount()
  })
})
