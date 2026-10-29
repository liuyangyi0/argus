import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { createMemoryHistory, createRouter } from 'vue-router'
import { useAuthStore } from '../../stores/useAuthStore'
import { getAnalysisTasks, type AnalysisTask } from '../../api/analysisTasks'
import AnalysisTasks from '../AnalysisTasks.vue'

const sockets = vi.hoisted(() => ({ refresh: undefined as undefined | (() => void | Promise<void>) }))
vi.mock('../../api/analysisTasks', () => ({ getAnalysisTasks: vi.fn() }))
vi.mock('../../composables/useWebSocket', () => ({ useWebSocket: (options: { fallbackPoll: () => Promise<void> }) => { sockets.refresh = options.fallbackPoll } }))

function task(id = 'task-one', camera = 'camera-one'): AnalysisTask {
  return { task_id: id, camera_id: camera, authority: 'managed', template_id: 'object_detection.v1',
    current_revision: 1, config_sha256: 'a'.repeat(64), name: `Task ${id}`, lifecycle: 'active',
    guidance: { schema_version: 'argus.task-guidance.v1', workflow_kind: 'model_task', training_requirement: 'optional',
      is_authorization: false, production_scope: 'unsupported', state: 'attention', reason_code: 'APPLICATION_UNVERIFIED',
      title: '实际应用版本尚未核实', detail: '先对账，不重复训练。', target: 'deployment', action_label: '核对实际应用版本' } }
}
async function setup(role: 'engineer' | 'viewer' = 'engineer') {
  const pinia = createPinia(); setActivePinia(pinia)
  const auth = useAuthStore(pinia)
  auth.currentUser = { username: 'reader', role }
  const router = createRouter({ history: createMemoryHistory(), routes: [{ path: '/:pathMatch(.*)*', component: { template: '<div />' } }] })
  await router.push('/analysis-tasks'); await router.isReady()
  const wrapper = mount(AnalysisTasks, { global: { plugins: [pinia, router] } })
  await flushPromises()
  return { wrapper, router, auth }
}
describe('AnalysisTasks', () => {
  beforeEach(() => { vi.clearAllMocks(); sockets.refresh = undefined })
  it('opens the exact task and requested section without executing it', async () => {
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [task(), task('two', 'camera-two')] })
    const { wrapper, router } = await setup()
    expect(wrapper.findAll('[data-test="task-center-card"]')).toHaveLength(2)
    await wrapper.get('[data-test="task-center-card"] [data-test="task-guidance"] button').trigger('click')
    await flushPromises()
    expect(router.currentRoute.value.path).toBe('/cameras/camera-one')
    expect(router.currentRoute.value.query).toEqual({ analysis_task: 'task-one', task_section: 'deployment' })
    expect(getAnalysisTasks).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })
  it('filters cameras and archived tasks without interpreting missing guidance as success', async () => {
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [task(), { ...task('two', 'camera-two'), guidance: null }, { ...task('old'), lifecycle: 'archived' }] })
    const { wrapper } = await setup()
    expect(wrapper.findAll('[data-test="task-center-card"]')).toHaveLength(2)
    await wrapper.get('select').setValue('camera-two')
    expect(wrapper.findAll('[data-test="task-center-card"]')).toHaveLength(1)
    expect(wrapper.text()).toContain('流程指引尚未核实')
    await wrapper.get('select').setValue('')
    await wrapper.get('input[type="checkbox"]').setValue(true)
    expect(wrapper.findAll('[data-test="task-center-card"]')).toHaveLength(3)
    wrapper.unmount()
  })
  it('retains stale results with next actions disabled after reconciliation fails', async () => {
    vi.mocked(getAnalysisTasks).mockResolvedValueOnce({ tasks: [task()] }).mockRejectedValueOnce(new Error('暂时不可用'))
    const { wrapper } = await setup()
    await sockets.refresh?.(); await flushPromises()
    expect(wrapper.text()).toContain('保留上次读取结果')
    expect(wrapper.get('[data-test="task-guidance"] button').attributes('disabled')).toBeDefined()
    expect(wrapper.findAll('[data-test="task-center-card"]')).toHaveLength(1)
    wrapper.unmount()
  })
  it('clears prior data on permission loss and does not present an empty success', async () => {
    vi.mocked(getAnalysisTasks).mockResolvedValueOnce({ tasks: [task()] }).mockRejectedValueOnce({ response: { status: 403, data: { code: 40300, msg: '权限不足' } } })
    const { wrapper } = await setup()
    await sockets.refresh?.(); await flushPromises()
    expect(wrapper.text()).toContain('无权访问')
    expect(wrapper.findAll('[data-test="task-center-card"]')).toHaveLength(0)
    expect(wrapper.text()).not.toContain('没有符合条件的任务')
    wrapper.unmount()
  })
  it('does not show management actions to viewers', async () => {
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [task()] })
    const { wrapper } = await setup('viewer')
    expect(wrapper.find('[data-test="task-guidance"] button').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('选择摄像头并创建任务')
    expect(wrapper.text()).toContain('查看相机与任务状态')
    wrapper.unmount()
  })
  it('discards late snapshots after a newer reconciliation', async () => {
    let finish!: (response: { tasks: AnalysisTask[] }) => void
    vi.mocked(getAnalysisTasks).mockReturnValueOnce(new Promise(resolve => { finish = resolve })).mockResolvedValueOnce({ tasks: [task('fresh')] })
    const { wrapper } = await setup()
    await sockets.refresh?.(); await flushPromises()
    finish({ tasks: [task('obsolete')] }); await flushPromises()
    expect(wrapper.text()).toContain('Task fresh')
    expect(wrapper.text()).not.toContain('Task obsolete')
    wrapper.unmount()
  })
  it('rejects duplicate task identity rather than rendering an ambiguous action', async () => {
    vi.mocked(getAnalysisTasks).mockResolvedValue({ tasks: [task(), task()] })
    const { wrapper } = await setup()
    expect(wrapper.text()).toContain('任务列表身份无效')
    expect(wrapper.findAll('[data-test="task-center-card"]')).toHaveLength(0)
    wrapper.unmount()
  })

  it('does not restore a late list after the account identity is cleared', async () => {
    let finish!: (response: { tasks: AnalysisTask[] }) => void
    vi.mocked(getAnalysisTasks).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const { wrapper, auth } = await setup()
    auth.clear(); await flushPromises()
    finish({ tasks: [task('private-old-account')] }); await flushPromises()
    expect(wrapper.findAll('[data-test="task-center-card"]')).toHaveLength(0)
    expect(wrapper.text()).not.toContain('private-old-account')
    await sockets.refresh?.(); await flushPromises()
    expect(getAnalysisTasks).toHaveBeenCalledTimes(1)
    wrapper.unmount()
  })

  it('fences pending reads across clear and same-identity reauthentication in one tick', async () => {
    let finish!: (response: { tasks: AnalysisTask[] }) => void
    vi.mocked(getAnalysisTasks).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
      .mockResolvedValueOnce({ tasks: [task('fresh-session')] })
    const { wrapper, auth } = await setup()
    auth.clear()
    auth.currentUser = { username: 'reader', role: 'engineer' }
    await flushPromises()
    finish({ tasks: [task('old-session')] }); await flushPromises()
    expect(wrapper.text()).toContain('Task fresh-session')
    expect(wrapper.text()).not.toContain('Task old-session')
    wrapper.unmount()
  })

  it('clears data for a permission business error even when a proxy returned HTTP 200', async () => {
    vi.mocked(getAnalysisTasks).mockResolvedValueOnce({ tasks: [task()] })
      .mockRejectedValueOnce({ response: { status: 200, data: { code: 40300, msg: '权限不足' } } })
    const { wrapper } = await setup()
    await sockets.refresh?.(); await flushPromises()
    expect(wrapper.findAll('[data-test="task-center-card"]')).toHaveLength(0)
    expect(wrapper.text()).toContain('无权访问')
    wrapper.unmount()
  })
})
