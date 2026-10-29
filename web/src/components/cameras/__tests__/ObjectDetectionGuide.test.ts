import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import ObjectDetectionGuide from '../ObjectDetectionGuide.vue'

describe('ObjectDetectionGuide', () => {
  it('applies guided fields while preserving region, collection and resource budgets', async () => {
    const wrapper = mount(ObjectDetectionGuide, { props: { modelValue: JSON.stringify({
      rules: { region_polygon: [[0, 0], [1, 0], [1, 1]] }, collection: { enabled: false },
      resources: { memory_limit_mb: 1024 }, sampling: { capacity: 16, mode: 'ordered_window' },
    }) } })
    await wrapper.get('[data-test="guide-labels"]').setValue('person, package')
    await wrapper.get('[data-test="guide-confidence"]').setValue('0.8')
    await wrapper.get('[data-test="guide-device"]').setValue('cuda')
    await wrapper.get('[data-test="apply-object-guide"]').trigger('click')
    const updated = JSON.parse(wrapper.emitted('update:modelValue')![0]![0] as string)
    expect(updated.rules.labels).toEqual(['person', 'package'])
    expect(updated.rules.region_polygon).toEqual([[0, 0], [1, 0], [1, 1]])
    expect(updated.rules.confidence_threshold).toBe(0.8)
    expect(updated.resources).toEqual({ memory_limit_mb: 1024, device: 'cuda' })
    expect(updated.collection).toEqual({ enabled: false })
    expect(updated.sampling.capacity).toBe(16)
    expect(updated.sampling.mode).toBe('ordered_window')
    wrapper.unmount()
  })

  it('does not overwrite invalid advanced JSON or silently drop the draft', async () => {
    const wrapper = mount(ObjectDetectionGuide, { props: { modelValue: '{invalid' } })
    expect(wrapper.get('[role="alert"]').text()).toContain('不会覆盖')
    expect(wrapper.get('fieldset').attributes('disabled')).toBeDefined()
    await wrapper.get('[data-test="apply-object-guide"]').trigger('click')
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
    wrapper.unmount()
  })

  it('rejects duplicate labels and invalid numeric inputs without rewriting JSON', async () => {
    const wrapper = mount(ObjectDetectionGuide, { props: { modelValue: '{}' } })
    await wrapper.get('[data-test="guide-labels"]').setValue('person, person')
    await wrapper.get('[data-test="apply-object-guide"]').trigger('click')
    expect(wrapper.text()).toContain('类别不能重复')
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
    await wrapper.get('[data-test="guide-labels"]').setValue('person')
    await wrapper.get('[data-test="guide-confidence"]').setValue('2')
    await wrapper.get('[data-test="apply-object-guide"]').trigger('click')
    expect(wrapper.text()).toContain('请检查阈值')
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
    wrapper.unmount()
  })

  it('keeps a stale or running task form read-only', async () => {
    const wrapper = mount(ObjectDetectionGuide, { props: { modelValue: '{}', disabled: true } })
    expect(wrapper.get('fieldset').attributes('disabled')).toBeDefined()
    await wrapper.get('[data-test="apply-object-guide"]').trigger('click')
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
    wrapper.unmount()
  })

  it('writes an explicit event collection draft without changing model or scene fields', async () => {
    const wrapper = mount(ObjectDetectionGuide, { props: { modelValue: JSON.stringify({
      scene: { policy: 'require_profile', profile_id: 'approved-profile' }, rules: { labels: ['person'] },
      collection: { enabled: false, policy: 'periodic', max_storage_mb: 32 },
    }) } })
    await wrapper.get('[data-test="guide-collection-enabled"]').setValue(true)
    await wrapper.get('[data-test="guide-collection-policy"]').setValue('events')
    await wrapper.get('[data-test="guide-collection-interval"]').setValue('0.5')
    await wrapper.get('[data-test="apply-collection-guide"]').trigger('click')
    const updated = JSON.parse(wrapper.emitted('update:modelValue')![0]![0] as string)
    expect(updated.collection).toEqual({ enabled: true, policy: 'events', interval_seconds: 0.5,
      max_samples: 1000, max_storage_mb: 32 })
    expect(updated.scene).toEqual({ policy: 'require_profile', profile_id: 'approved-profile' })
    expect(updated.rules).toEqual({ labels: ['person'] })
    expect(wrapper.text()).toContain('没有样本不代表现场正常')
    wrapper.unmount()
  })

  it('preserves unsupported policy until explicitly changed and rejects invalid budgets', async () => {
    const wrapper = mount(ObjectDetectionGuide, { props: { modelValue: '{"collection":{"policy":"disagreement"}}' } })
    await wrapper.get('[data-test="apply-collection-guide"]').trigger('click')
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
    expect(wrapper.text()).toContain('不会自动替换')
    await wrapper.get('[data-test="guide-collection-policy"]').setValue('events')
    await wrapper.get('[data-test="guide-collection-limit"]').setValue('1.5')
    await wrapper.get('[data-test="apply-collection-guide"]').trigger('click')
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
    expect(wrapper.text()).toContain('请检查采集间隔')
    wrapper.unmount()
  })
})
