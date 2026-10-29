import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import TaskGuidanceCard from '../TaskGuidanceCard.vue'
import type { AnalysisTask } from '../../../api/analysisTasks'

const task = (): AnalysisTask => ({
  task_id: 'task-one', camera_id: 'camera-one', authority: 'managed', template_id: 'object_detection.v1',
  current_revision: 1, config_sha256: 'a'.repeat(64),
  guidance: { schema_version: 'argus.task-guidance.v1', workflow_kind: 'model_task', training_requirement: 'optional',
    is_authorization: false, production_scope: 'unsupported', state: 'attention', reason_code: 'APPLICATION_UNVERIFIED',
    title: '实际应用版本尚未核实', detail: '先对账，不重复训练。', target: 'deployment', action_label: '核对实际应用版本' },
})

describe('TaskGuidanceCard', () => {
  it('uses server guidance and emits navigation only, not a deployment command', async () => {
    const wrapper = mount(TaskGuidanceCard, { props: { task: task(), actionable: true } })
    expect(wrapper.text()).toContain('实际应用版本尚未核实')
    expect(wrapper.text()).toContain('不是必经步骤')
    expect(wrapper.text()).toContain('不是启动许可')
    await wrapper.get('button').trigger('click')
    expect(wrapper.emitted()).toHaveProperty('open', [['deployment']])
    wrapper.unmount()
  })
  it('disables stale next actions and keeps explanation', async () => {
    const wrapper = mount(TaskGuidanceCard, { props: { task: task(), actionable: true, stale: true } })
    expect(wrapper.get('button').attributes('disabled')).toBeDefined()
    await wrapper.get('button').trigger('click')
    expect(wrapper.emitted('open')).toBeUndefined()
    expect(wrapper.text()).toContain('状态已过期')
    wrapper.unmount()
  })
  it('does not give a viewer management navigation', () => {
    const wrapper = mount(TaskGuidanceCard, { props: { task: task() } })
    expect(wrapper.find('button').exists()).toBe(false)
    wrapper.unmount()
  })
  it('treats missing guidance as unknown, not healthy', () => {
    const wrapper = mount(TaskGuidanceCard, { props: { task: { ...task(), guidance: null }, actionable: true } })
    expect(wrapper.text()).toContain('尚未核实')
    expect(wrapper.find('button').exists()).toBe(false)
    wrapper.unmount()
  })
  it('explains algorithm and legacy paths without inferring approval', () => {
    const fixture = task()
    fixture.guidance!.training_requirement = 'not_applicable'
    const wrapper = mount(TaskGuidanceCard, { props: { task: fixture } })
    expect(wrapper.text()).toContain('不需要模型权重或训练')
    wrapper.unmount()
  })
})
