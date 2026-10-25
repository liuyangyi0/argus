import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../../../api/analysisTasks'
import type { AnalysisTask } from '../../../api/analysisTasks'
import TaskDatasetPanel from '../TaskDatasetPanel.vue'

vi.mock('../../../api/analysisTasks', async load => ({
  ...await load<typeof import('../../../api/analysisTasks')>(),
  getTaskWorkflows: vi.fn(), getTaskTrainingRecipes: vi.fn(),
  reviewTaskSample: vi.fn(), createTaskWorkflow: vi.fn(),
}))
enableAutoUnmount(afterEach)
const task: AnalysisTask = {
  task_id: 'task-1', camera_id: 'cam1', template_id: 'object_detection.v1',
  authority: 'managed', current_revision: 1, config_sha256: 'a'.repeat(64),
  runtime: { state: 'stopped' }, source: { state: 'running' },
}
const annotation = { schema: 'argus.task-annotation.v1', coordinate_space: 'normalized', kind: 'detection', objects: [] }
const row = (revision = 1, notes = 'Original review') => ({
  workflow_id: 'workflow-1', task_revision: 1, revision, state: 'review',
  manifest: {
    samples: [{ sample_id: 'sample-1', image_sha256: 'b'.repeat(64), identity: { frame_index: 7 } }],
    reviews: { 'sample-1': { decision: 'needs_review', notes, annotation } },
  },
})

describe('TaskDatasetPanel', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [row()] })
    vi.mocked(api.getTaskTrainingRecipes).mockResolvedValue({ documents: [] })
  })

  it('keeps drafts and requires explicit reload after review revision changes', async () => {
    const wrapper = mount(TaskDatasetPanel, { props: { task } })
    await flushPromises()
    await wrapper.get('[data-test="sample-select"]').setValue('sample-1')
    await wrapper.get('img').trigger('load')
    const notes = wrapper.findAll('textarea')[0]!
    await notes.setValue('Unsaved draft')
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [row(2, 'New human review')] })
    await wrapper.findAll('button').find(button => button.text() === '刷新流程')!.trigger('click')
    await flushPromises()
    expect((notes.element as HTMLTextAreaElement).value).toBe('Unsaved draft')
    const approve = () => wrapper.findAll('button').find(button => button.text() === '批准此标注')!
    expect(approve().attributes('disabled')).toBeDefined()
    expect(api.reviewTaskSample).not.toHaveBeenCalled()
    await wrapper.findAll('button').find(button => button.text() === '重新载入当前审核')!.trigger('click')
    expect((notes.element as HTMLTextAreaElement).value).toBe('New human review')
    vi.mocked(api.reviewTaskSample).mockResolvedValue({ document: row(3) })
    await approve().trigger('click')
    await flushPromises()
    expect(api.reviewTaskSample).toHaveBeenCalledWith('task-1', 'workflow-1', expect.objectContaining({
      expected_revision: 2, expected_image_sha256: 'b'.repeat(64), notes: 'New human review',
    }))
  })

  it('ignores an older workflow fetch that completes after a command refresh', async () => {
    let finish!: (value: { documents: Record<string, unknown>[] }) => void
    vi.mocked(api.getTaskWorkflows).mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
      .mockResolvedValueOnce({ documents: [row(3)] })
    vi.mocked(api.createTaskWorkflow).mockResolvedValue({ document: row(3) })
    const wrapper = mount(TaskDatasetPanel, { props: { task } })
    await wrapper.findAll('button').find(button => button.text() === '新建数据流程')!.trigger('click')
    await flushPromises()
    expect(wrapper.text()).toContain('流程版本 3')
    finish({ documents: [row(1)] })
    await flushPromises()
    expect(wrapper.text()).toContain('流程版本 3')
    expect(wrapper.text()).not.toContain('流程版本 1')
  })

  it('never unlocks approval for a failed image or a late image from another sample', async () => {
    const workflow = row()
    workflow.manifest.samples.push({ sample_id: 'sample-2', image_sha256: 'c'.repeat(64), identity: { frame_index: 8 } })
    vi.mocked(api.getTaskWorkflows).mockResolvedValue({ documents: [workflow] })
    const wrapper = mount(TaskDatasetPanel, { props: { task } })
    await flushPromises()
    const select = wrapper.get('[data-test="sample-select"]')
    await select.setValue('sample-1')
    const oldImage = wrapper.get('img')
    const approve = () => wrapper.findAll('button').find(button => button.text() === '批准此标注')!
    expect(approve().attributes('disabled')).toBeDefined()
    await oldImage.trigger('error')
    expect(wrapper.text()).toContain('审核图片加载失败')
    expect(approve().attributes('disabled')).toBeDefined()
    await select.setValue('sample-2')
    await oldImage.trigger('load')
    expect(approve().attributes('disabled')).toBeDefined()
    expect(api.reviewTaskSample).not.toHaveBeenCalled()
    await wrapper.get('img').trigger('load')
    expect(approve().attributes('disabled')).toBeUndefined()
    expect(wrapper.text()).toContain('待人工复核')
  })
})
