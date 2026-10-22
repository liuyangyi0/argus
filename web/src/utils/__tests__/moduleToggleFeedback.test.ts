import { beforeEach, describe, expect, it, vi } from 'vitest'
import { message } from 'ant-design-vue'
import { notifyModuleToggleResult } from '../moduleToggleFeedback'

vi.mock('ant-design-vue', () => ({
  message: {
    success: vi.fn(),
    warning: vi.fn(),
  },
}))

describe('notifyModuleToggleResult', () => {
  beforeEach(() => {
    vi.mocked(message.success).mockReset()
    vi.mocked(message.warning).mockReset()
  })

  it('warns when a hot-reloadable module only applies to part of the runtime', () => {
    notifyModuleToggleResult('分类器', true, {
      key: 'classifier.enabled',
      value: true,
      restart_required: true,
      hot_reloaded: 1,
      hot_reloadable: true,
      hot_reload_failed: 1,
      pipelines_seen: 2,
      persisted: true,
    })

    expect(message.warning).toHaveBeenCalledWith(
      '分类器 配置已启用，但运行时未全部生效（1/2 个管线已应用，1 个失败）；请检查失败原因后重试，必要时重启摄像头；旧任务排空前不要反复重启',
    )
    expect(message.success).not.toHaveBeenCalled()
  })

  it('does not report success when failures are present but restart_required is false', () => {
    notifyModuleToggleResult('分割器', true, {
      key: 'segmenter.enabled', value: true, restart_required: false,
      hot_reloaded: 1, hot_reloadable: true, hot_reload_failed: 1,
      pipelines_seen: 2, persisted: true, all_applied: false,
    })
    expect(message.warning).toHaveBeenCalledWith(expect.stringContaining('运行时未全部生效'))
    expect(message.success).not.toHaveBeenCalled()
  })

  it('does not treat an unknown result or an empty runtime as confirmed readiness', () => {
    notifyModuleToggleResult('分类器', true, undefined)
    notifyModuleToggleResult('分割器', true, {
      key: 'segmenter.enabled', value: true, restart_required: false,
      hot_reloaded: 0, hot_reloadable: true, hot_reload_failed: 0,
      pipelines_seen: 0, persisted: true,
    })
    expect(message.warning).toHaveBeenCalledWith(expect.stringContaining('应用结果未知'))
    expect(message.warning).toHaveBeenCalledWith(expect.stringContaining('当前无运行管线'))
    expect(message.success).not.toHaveBeenCalled()
  })

  it('warns when the runtime change was not persisted to YAML', () => {
    notifyModuleToggleResult('多模态成像', false, {
      key: 'imaging.enabled',
      value: false,
      restart_required: false,
      hot_reloaded: 0,
      hot_reloadable: false,
      hot_reload_failed: 0,
      pipelines_seen: 0,
      persisted: false,
    })

    expect(message.warning).toHaveBeenCalledWith(
      '多模态成像 已关闭（未写入 YAML，重启后可能丢失）',
    )
  })

  it('keeps a persistence warning even when every runtime accepted the change', () => {
    notifyModuleToggleResult('分类器', true, {
      key: 'classifier.enabled', value: true, restart_required: false,
      hot_reloaded: 1, hot_reloadable: true, hot_reload_failed: 0,
      pipelines_seen: 1, persisted: false, all_applied: true,
    })
    expect(message.warning).toHaveBeenCalledWith(expect.stringContaining('未写入 YAML'))
    expect(message.success).not.toHaveBeenCalled()
  })
})
