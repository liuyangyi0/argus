import { message } from 'ant-design-vue'
import type { ModuleToggleResult } from '../api/system'

export function notifyModuleToggleResult(
  label: string,
  enabled: boolean,
  result: ModuleToggleResult | null | undefined,
) {
  const action = enabled ? '已启用' : '已关闭'
  const persistSuffix = result?.persisted === false
    ? '；未写入 YAML，重启后可能丢失'
    : ''

  if (result?.restart_required) {
    const detail = result.hot_reloadable && result.pipelines_seen > 0
      ? `（${result.hot_reloaded}/${result.pipelines_seen} 个管线已热加载${persistSuffix}）`
      : `（已运行的摄像头管线需要重启才会生效${persistSuffix}）`
    message.warning(`${label} ${action}${detail}`)
    return
  }

  if (result?.hot_reloaded && result.hot_reloaded > 0) {
    message.success(`${label} ${action}（${result.hot_reloaded} 个管线已热加载${persistSuffix}）`)
    return
  }

  if (result?.persisted === false) {
    message.warning(`${label} ${action}（未写入 YAML，重启后可能丢失）`)
    return
  }

  message.success(`${label} ${action}`)
}
