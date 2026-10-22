import { message } from 'ant-design-vue'
import type { ModuleToggleResult } from '../api/system'

export interface ModuleToggleFeedback {
  type: 'success' | 'warning'
  message: string
}

export function describeModuleToggleResult(
  label: string,
  enabled: boolean,
  result: ModuleToggleResult | null | undefined,
): ModuleToggleFeedback {
  const action = enabled ? '已启用' : '已关闭'
  const persistSuffix = result?.persisted === false
    ? '；未写入 YAML，重启后可能丢失'
    : ''
  if (!result) return { type: 'warning', message: `${label} 配置请求已完成，但运行时应用结果未知，请刷新核验` }

  const incomplete = result.hot_reloadable && result.pipelines_seen > result.hot_reloaded
  if (result.hot_reload_failed > 0 || result.all_applied === false || incomplete) {
    const recovery = result.restart_required
      ? '请检查失败原因后重试，必要时重启摄像头；旧任务排空前不要反复重启'
      : '请检查运行时状态后重试'
    return {
      type: 'warning',
      message: `${label} 配置${action}，但运行时未全部生效（${result.hot_reloaded}/${result.pipelines_seen} 个管线已应用，${result.hot_reload_failed} 个失败${persistSuffix}）；${recovery}`,
    }
  }
  if (result.restart_required) {
    return { type: 'warning', message: `${label} 配置${action}（已运行的摄像头管线需要重启才会生效${persistSuffix}）` }
  }
  if (result.hot_reloadable && result.pipelines_seen === 0) {
    return { type: 'warning', message: `${label} 配置${action}（当前无运行管线，启动后仍需核验模型状态${persistSuffix}）` }
  }
  if (result.hot_reloaded > 0) {
    if (result.persisted === false) {
      return { type: 'warning', message: `${label} 已在 ${result.hot_reloaded} 个管线应用，但未写入 YAML，重启后可能丢失` }
    }
    return { type: 'success', message: `${label} ${action}（${result.hot_reloaded} 个管线已热加载${persistSuffix}）` }
  }
  if (result.persisted === false) {
    return { type: 'warning', message: `${label} ${action}（未写入 YAML，重启后可能丢失）` }
  }
  return { type: 'success', message: `${label} 配置${action}` }
}

export function notifyModuleToggleResult(
  label: string,
  enabled: boolean,
  result: ModuleToggleResult | null | undefined,
) {
  const feedback = describeModuleToggleResult(label, enabled, result)
  message[feedback.type](feedback.message)
  return feedback
}
