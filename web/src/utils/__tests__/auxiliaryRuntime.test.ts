import { describe, expect, it } from 'vitest'
import type { AuxiliaryRuntimeSummary } from '../../api/system'
import { describeAuxiliaryRuntime } from '../auxiliaryRuntime'

const base: AuxiliaryRuntimeSummary = {
  total_pipelines: 1, pipelines_attached: 1, pipelines_loaded: 1,
}

describe('auxiliary runtime readiness', () => {
  it('does not equate a legacy loaded flag with model readiness', () => {
    const result = describeAuxiliaryRuntime('分割器', true, base)
    expect(result.type).toBe('info')
    expect(result.message).toContain('无法确认模型可用')
    expect(result.message).not.toContain('已在全部')
  })

  it('does not equate a disabled configuration with a stopped runtime', () => {
    const result = describeAuxiliaryRuntime('分类器', false, base)
    expect(result.type).toBe('warning')
    expect(result.message).toContain('尚未确认全部停止')
  })

  it('keeps unknown cleanup failures visible even after detachment', () => {
    const result = describeAuxiliaryRuntime('分类器', false, {
      total_pipelines: 1, pipelines_attached: 0, pipelines_loaded: 0, pipelines_failed: 1,
    })
    expect(result.type).toBe('warning')
    expect(result.message).toContain('尚未确认全部停止')
  })

  it('keeps a detached but draining SDK task visible after disabling the configuration', () => {
    const result = describeAuxiliaryRuntime('分割器', false, {
      total_pipelines: 1, pipelines_attached: 0, pipelines_loaded: 0,
      details: [{ camera_id: 'camera-1', state: 'stopping', backend: null, loaded: false, ready: false, degraded: false, busy: true }],
    })
    expect(result.type).toBe('warning')
    expect(result.message).toContain('模块尚未完全停止')
  })

  it('shows real model readiness without claiming recognition acceptance', () => {
    const result = describeAuxiliaryRuntime('分类器', true, { ...base, pipelines_ready: 1 })
    expect(result.type).toBe('success')
    expect(result.message).toContain('不代表识别效果已验收')
  })

  it('never reports a contour fallback or failed runtime as ready', () => {
    const result = describeAuxiliaryRuntime('分割器', true, {
      ...base, pipelines_ready: 1,
      details: [{ camera_id: 'camera-1', state: 'degraded', backend: 'contour', loaded: true, ready: false, degraded: true }],
    })
    expect(result.type).toBe('warning')
    expect(result.message).toContain('降级')
    const failed = describeAuxiliaryRuntime('分类器', true, { ...base, pipelines_failed: 1, pipelines_ready: 1 })
    expect(failed.type).toBe('error')
  })

  it('does not trust a ready aggregate when per-camera details disagree', () => {
    const result = describeAuxiliaryRuntime('分类器', true, {
      ...base, pipelines_ready: 1,
      details: [{ camera_id: 'camera-1', state: 'loading', backend: null, loaded: false, ready: false, degraded: false }],
    })
    expect(result.type).toBe('warning')
    expect(result.message).toContain('仅 0 / 1 路就绪')
  })

  it('distinguishes absent, partially attached, and still busy runtimes', () => {
    expect(describeAuxiliaryRuntime('分类器', true, { total_pipelines: 0, pipelines_attached: 0, pipelines_loaded: 0 }).message)
      .toContain('尚未验证模型加载')
    expect(describeAuxiliaryRuntime('分类器', true, { ...base, pipelines_attached: 0 }).message)
      .toContain('未全部生效')
    expect(describeAuxiliaryRuntime('分割器', true, {
      ...base, details: [{ camera_id: 'camera-1', state: 'busy', backend: 'sam2', loaded: true, ready: false, degraded: false, busy: true }],
    }).message).toContain('不应重复提交')
  })
})
