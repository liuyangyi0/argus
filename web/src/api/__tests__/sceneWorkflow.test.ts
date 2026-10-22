import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '../client'
import {
  activateSceneProfile,
  approveSceneProfile,
  createSceneTrainingJob,
  createSceneWorkflow,
  getSceneWorkflow,
  startSceneBaselineJob,
} from '../sceneWorkflow'

vi.mock('../client', () => ({
  api: { get: vi.fn(), post: vi.fn() },
  unwrap: (response: { data: { data: unknown } }) => response.data.data,
}))

describe('scene workflow API contract', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    vi.mocked(api.get).mockResolvedValue({ data: { code: 0, msg: 'ok', data: {} } })
    vi.mocked(api.post).mockResolvedValue({ data: { code: 0, msg: 'ok', data: {} } })
  })

  it('encodes both identities instead of treating them as URL paths', async () => {
    await getSceneWorkflow('camera/one')
    expect(api.get).toHaveBeenCalledWith('/scene-workflow/camera%2Fone')
    await createSceneTrainingJob('camera/one', 'scene/one', { model_type: 'patchcore', image_size: 256 })
    expect(api.post).toHaveBeenCalledWith('/scene-workflow/camera%2Fone/workflows/scene%2Fone/training-jobs', {
      model_type: 'patchcore', image_size: 256,
    })
  })

  it('keeps reference confirmation, approval, and scene selection in separate requests', async () => {
    const draft = {
      workflow_id: 'new-scene', reference_path: 'cam/reference.png',
      reference_sha256: 'a'.repeat(64), normal_reference_confirmed: true as const,
    }
    await createSceneWorkflow('cam', draft)
    expect(api.post).toHaveBeenLastCalledWith('/scene-workflow/cam/workflows', draft)
    await approveSceneProfile('cam', 'new-scene')
    expect(api.post).toHaveBeenLastCalledWith('/scene-workflow/cam/workflows/new-scene/approve-profile', {
      operator_approved: true,
    })
    await activateSceneProfile('cam', 'new-scene', null, null)
    expect(api.post).toHaveBeenLastCalledWith('/scene-workflow/cam/workflows/new-scene/activate-profile', {
      acknowledge_runtime_unchanged: true,
      expected_active_profile_id: null,
      expected_active_profile_revision: null,
    })
    expect(api.post).toHaveBeenCalledTimes(3)
  })

  it('does not introduce decoder, approval, activation, or production overrides to a baseline job', async () => {
    const baseline = {
      sources: [
        { video: 'a.mp4', annotations: 'a.json' },
        { video: 'b.mp4', annotations: 'b.json' },
      ],
      sample_every: 30,
      max_images: 2000,
    }
    await startSceneBaselineJob('cam', 'scene', baseline)
    expect(api.post).toHaveBeenCalledOnce()
    expect(api.post).toHaveBeenCalledWith('/scene-workflow/cam/workflows/scene/baseline-jobs', baseline)
  })

  it('sends both active pointer preconditions, including a 64-character legacy profile ID', async () => {
    await activateSceneProfile('cam', 'new-scene', 'a'.repeat(64), 'b'.repeat(64))
    expect(api.post).toHaveBeenCalledWith('/scene-workflow/cam/workflows/new-scene/activate-profile', {
      acknowledge_runtime_unchanged: true,
      expected_active_profile_id: 'a'.repeat(64),
      expected_active_profile_revision: 'b'.repeat(64),
    })
  })
})
