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
      '分类器 已启用（1/2 个管线已热加载）',
    )
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
})
