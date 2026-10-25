import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../../../api/analysisTasks'
import TaskWorkbench from '../TaskWorkbench.vue'
import type { AnalysisTask } from '../../../api/analysisTasks'

vi.mock('../../../api/analysisTasks', async (load) => ({
  ...await load<typeof import('../../../api/analysisTasks')>(),
  getAnalysisTaskTemplates: vi.fn(), createAnalysisTask: vi.fn(), updateAnalysisTask: vi.fn(),
  getTaskModelCompatibility: vi.fn(), getTaskBundles: vi.fn(), freezeTaskBundle: vi.fn(), startAnalysisTask: vi.fn(), stopAnalysisTask: vi.fn(),
}))

const task = (id = 'task-1'): AnalysisTask => ({
  task_id: id, camera_id: 'cam1', template_id: 'object_detection.v1', name: 'Object task',
  authority: 'managed', lifecycle: 'active', current_revision: 2, config_sha256: 'a'.repeat(64),
  config: {}, runtime: { state: 'stopped' }, source: { state: 'running' },
})
const templates = () => ({ templates: [{
  template_id: 'object_detection.v1', name: 'Object detection', supports_creation: true,
  availability: 'shadow' as const, deployment_authority: 'managed' as const,
  safety_contract: 'shadow_only', default_config: {}, model_roles: { detector: 'onnx_detection.v1' },
}] })
const model = (id: string, role = 'detector', compatible = true) => ({
  model_id: id, name: id, compatible_roles: compatible ? [role] : [],
  role_compatibility: { [role]: { compatible, reasons: compatible ? [] : ['REQUIRED_LABELS_MISSING'] } },
})

describe('TaskWorkbench', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getAnalysisTaskTemplates).mockResolvedValue(templates())
    vi.mocked(api.getTaskModelCompatibility).mockResolvedValue({ documents: [] })
    vi.mocked(api.getTaskBundles).mockResolvedValue({ documents: [] })
    vi.mocked(api.freezeTaskBundle).mockResolvedValue({ document: {} })
  })

  it('reuses a command key after an uncertain create and leaves production disabled', async () => {
    vi.mocked(api.createAnalysisTask).mockRejectedValueOnce(new Error('network failed'))
      .mockResolvedValueOnce(task())
    const wrapper = mount(TaskWorkbench, { props: { cameraId: 'cam1', tasks: [] }, global: { stubs: { TaskDatasetPanel: true, TaskDeploymentPanel: true } } })
    await flushPromises()
    await wrapper.get('[data-test="task-name"]').setValue('New task')
    await wrapper.get('[data-test="create-task"]').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('network failed')
    await wrapper.get('[data-test="create-task"]').trigger('click')
    await flushPromises()
    expect(api.createAnalysisTask).toHaveBeenCalledTimes(2)
    const calls = vi.mocked(api.createAnalysisTask).mock.calls
    expect(calls[0]?.[1]).toEqual(calls[1]?.[1])
    expect(calls[0]?.[0].camera_id).toBe('cam1')
    expect(wrapper.emitted('refresh')).toHaveLength(1)
    wrapper.unmount()
  })

  it('delegates explicit bundle identity to the durable deployment panel instead of the old runtime endpoint', async () => {
    vi.mocked(api.getTaskBundles).mockResolvedValue({ documents: [{ bundle_id: 'b'.repeat(64), task_revision: 1 }] })
    const wrapper = mount(TaskWorkbench, { props: { cameraId: 'cam1', tasks: [task()] }, global: { stubs: { TaskDatasetPanel: true, TaskDeploymentPanel: true } } })
    await flushPromises()
    await wrapper.get('[data-test="task-select"]').setValue('task-1')
    await flushPromises()
    const selectors = wrapper.findAll('select')
    await selectors[selectors.length - 1]!.setValue('b'.repeat(64))
    const panel = wrapper.findComponent({ name: 'TaskDeploymentPanel' })
    expect(panel.props('task').task_id).toBe('task-1')
    expect(panel.props('bundleId')).toBe('b'.repeat(64))
    expect(wrapper.find('[data-test="start-task"]').exists()).toBe(false)
    expect(wrapper.findAll('button').find(button => button.text().includes('生产发布'))?.attributes('disabled')).toBeDefined()
    expect(api.startAnalysisTask).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('ignores model and bundle responses after switching tasks', async () => {
    let finish!: (value: { documents: Record<string, unknown>[] }) => void
    vi.mocked(api.getTaskModelCompatibility).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
      .mockResolvedValueOnce({ documents: [model('New model')] })
    const wrapper = mount(TaskWorkbench, { props: { cameraId: 'cam1', tasks: [task(), task('task-2')] }, global: { stubs: { TaskDatasetPanel: true, TaskDeploymentPanel: true } } })
    await flushPromises()
    await wrapper.get('[data-test="task-select"]').setValue('task-1')
    await wrapper.get('[data-test="task-select"]').setValue('task-2')
    await flushPromises()
    finish({ documents: [model('Old model')] })
    await flushPromises()
    expect(wrapper.text()).toContain('New model')
    expect(wrapper.text()).not.toContain('Old model')
    wrapper.unmount()
  })

  it('does not silently rebase an old configuration form onto a newer task revision', async () => {
    const wrapper = mount(TaskWorkbench, { props: { cameraId: 'cam1', tasks: [task()] }, global: { stubs: { TaskDatasetPanel: true, TaskDeploymentPanel: true } } })
    await flushPromises()
    await wrapper.get('[data-test="task-select"]').setValue('task-1')
    await flushPromises()
    await wrapper.get('textarea[aria-label="任务配置 JSON"]').setValue('{"rules":{"labels":["draft"]}}')
    await wrapper.setProps({ tasks: [{ ...task(), current_revision: 3, config: { rules: { labels: ['new'] } } }] })
    expect(wrapper.get('[data-test="save-task"]').attributes('disabled')).toBeDefined()
    expect((wrapper.get('textarea[aria-label="任务配置 JSON"]').element as HTMLTextAreaElement).value).toContain('draft')
    expect(api.updateAnalysisTask).not.toHaveBeenCalled()
    await wrapper.findAll('button').find(button => button.text() === '重新载入当前配置')!.trigger('click')
    expect(wrapper.get('[data-test="save-task"]').attributes('disabled')).toBeUndefined()
    expect((wrapper.get('textarea[aria-label="任务配置 JSON"]').element as HTMLTextAreaElement).value).toContain('new')
    wrapper.unmount()
  })

  it('fences old responses when selecting A then B then A again', async () => {
    let finish!: (value: { documents: Record<string, unknown>[] }) => void
    vi.mocked(api.getTaskModelCompatibility).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
      .mockResolvedValueOnce({ documents: [] })
      .mockResolvedValueOnce({ documents: [model('Fresh A model')] })
    const wrapper = mount(TaskWorkbench, { props: { cameraId: 'cam1', tasks: [task(), task('task-2')] }, global: { stubs: { TaskDatasetPanel: true, TaskDeploymentPanel: true } } })
    await flushPromises()
    const select = wrapper.get('[data-test="task-select"]')
    await select.setValue('task-1')
    await select.setValue('task-2')
    await select.setValue('task-1')
    await flushPromises()
    finish({ documents: [model('Outdated A model')] })
    await flushPromises()
    expect(wrapper.text()).toContain('Fresh A model')
    expect(wrapper.text()).not.toContain('Outdated A model')
    wrapper.unmount()
  })

  it('clears camera-scoped candidate drafts when changing cameras', async () => {
    const wrapper = mount(TaskWorkbench, { props: { cameraId: 'cam1', tasks: [] }, global: { stubs: { TaskDatasetPanel: true, TaskDeploymentPanel: true } } })
    await flushPromises()
    const path = () => wrapper.get('input[placeholder="models 目录内的相对路径"]')
    await path().setValue('camera-a/detector.onnx')
    await wrapper.setProps({ cameraId: 'cam2', tasks: [] })
    await flushPromises()
    expect((path().element as HTMLInputElement).value).toBe('')
    wrapper.unmount()
  })

  it('requires and freezes all model roles, then clears their bindings on task switch', async () => {
    const dual = { ...templates().templates[0]!, template_id: 'ppe_compliance_dual.v1', model_roles: { person_detector: 'onnx_detection.v1', head_detector: 'onnx_detection.v1' } }
    vi.mocked(api.getAnalysisTaskTemplates).mockResolvedValue({ templates: [dual] })
    vi.mocked(api.getTaskModelCompatibility).mockResolvedValue({ documents: [model('person-model', 'person_detector'), model('head-model', 'head_detector')] })
    const wrapper = mount(TaskWorkbench, { props: { cameraId: 'cam1', tasks: [{ ...task(), template_id: dual.template_id }, { ...task('task-2'), template_id: dual.template_id }] }, global: { stubs: { TaskDatasetPanel: true, TaskDeploymentPanel: true } } })
    await flushPromises()
    await wrapper.get('[data-test="task-select"]').setValue('task-1'); await flushPromises()
    await wrapper.get('[data-test="model-role-person_detector"]').setValue('person-model')
    await wrapper.get('[data-test="freeze-bundle"]').trigger('click'); await flushPromises()
    expect(api.freezeTaskBundle).not.toHaveBeenCalled()
    await wrapper.get('[data-test="model-role-head_detector"]').setValue('head-model')
    await wrapper.get('[data-test="freeze-bundle"]').trigger('click'); await flushPromises()
    expect(api.freezeTaskBundle).toHaveBeenCalledWith('task-1', { expected_revision: 2, model_ids: { person_detector: 'person-model', head_detector: 'head-model' } }, expect.any(String))
    await wrapper.get('[data-test="task-select"]').setValue('task-2'); await flushPromises()
    expect((wrapper.get('[data-test="model-role-person_detector"]').element as HTMLSelectElement).value).toBe('')
    wrapper.unmount()
  })

  it('explains incompatible models and rejects programmatically selecting a disabled option', async () => {
    vi.mocked(api.getTaskModelCompatibility).mockResolvedValue({ documents: [model('bad-model', 'detector', false)] })
    const wrapper = mount(TaskWorkbench, { props: { cameraId: 'cam1', tasks: [task()] }, global: { stubs: { TaskDatasetPanel: true, TaskDeploymentPanel: true } } })
    await flushPromises()
    await wrapper.get('[data-test="task-select"]').setValue('task-1'); await flushPromises()
    const option = wrapper.get('option[value="bad-model"]')
    expect(option.attributes('disabled')).toBeDefined()
    expect(option.text()).toContain('缺少角色必需类别')
    await wrapper.get('[data-test="model-role-detector"]').setValue('bad-model')
    await wrapper.get('[data-test="freeze-bundle"]').trigger('click'); await flushPromises()
    expect(api.freezeTaskBundle).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('reloads bundles after a deployment/configuration event without automatically selecting a new bundle', async () => {
    vi.mocked(api.getTaskBundles).mockResolvedValueOnce({ documents: [] })
      .mockResolvedValueOnce({ documents: [{ bundle_id: 'restored-bundle', task_id: 'task-1', task_revision: 3 }] })
    const wrapper = mount(TaskWorkbench, { props: { cameraId: 'cam1', tasks: [task()] }, global: { stubs: { TaskDatasetPanel: true, TaskDeploymentPanel: true } } })
    await flushPromises()
    await wrapper.get('[data-test="task-select"]').setValue('task-1'); await flushPromises()
    wrapper.findComponent({ name: 'TaskDeploymentPanel' }).vm.$emit('refresh')
    await flushPromises()
    expect(api.getTaskBundles).toHaveBeenCalledTimes(2)
    expect(wrapper.find('option[value="restored-bundle"]').exists()).toBe(true)
    expect((wrapper.get('[data-test="bundle-select"]').element as HTMLSelectElement).value).toBe('')
    expect(wrapper.emitted('refresh')).toHaveLength(1)
    wrapper.unmount()
  })

  it.each(['resolve', 'reject'] as const)('fences an older same-task bundle refresh that later %s', async outcome => {
    let finish!: (value: { documents: Record<string, unknown>[] }) => void
    let fail!: (cause: Error) => void
    vi.mocked(api.getTaskBundles).mockResolvedValueOnce({ documents: [] })
      .mockReturnValueOnce(new Promise((resolve, reject) => { finish = resolve; fail = reject }))
      .mockResolvedValueOnce({ documents: [{ bundle_id: 'fresh-bundle', task_id: 'task-1', task_revision: 3 }] })
    const wrapper = mount(TaskWorkbench, { props: { cameraId: 'cam1', tasks: [task()] }, global: { stubs: { TaskDatasetPanel: true, TaskDeploymentPanel: true } } })
    await flushPromises()
    await wrapper.get('[data-test="task-select"]').setValue('task-1'); await flushPromises()
    const panel = wrapper.findComponent({ name: 'TaskDeploymentPanel' })
    panel.vm.$emit('refresh'); await flushPromises()
    panel.vm.$emit('refresh'); await flushPromises()
    expect(wrapper.find('option[value="fresh-bundle"]').exists()).toBe(true)
    if (outcome === 'resolve') finish({ documents: [{ bundle_id: 'outdated-bundle', task_id: 'task-1', task_revision: 2 }] })
    else fail(new Error('outdated refresh failure'))
    await flushPromises()
    expect(wrapper.find('option[value="fresh-bundle"]').exists()).toBe(true)
    expect(wrapper.find('option[value="outdated-bundle"]').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('outdated refresh failure')
    wrapper.unmount()
  })
})
