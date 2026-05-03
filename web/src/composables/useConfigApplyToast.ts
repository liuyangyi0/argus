import { message } from 'ant-design-vue'
import type { ConfigApplyResult } from '../types/api'

/** Render a per-section toast given the three-state response from
 *  POST /config/detection-params (痛点 9). */
export function useConfigApplyToast() {
  function notify(label: string, result: ConfigApplyResult | undefined | null) {
    if (!result || !result.changed) return
    if (result.hot_reloaded && result.applied === result.total && result.total > 0) {
      message.success(`${label}：✓ 已实时生效 (${result.applied}/${result.total})`)
    } else if (result.hot_reloaded && result.applied < result.total) {
      message.warning(
        `${label}：已应用到 ${result.applied}/${result.total} 个流水线`,
      )
    } else {
      message.warning(`${label}：⚠ 需要重启进程才生效`)
    }
  }

  function notifyAll(payload: {
    anomaly_threshold?: ConfigApplyResult
    severity?: ConfigApplyResult
    temporal?: ConfigApplyResult
    suppression?: ConfigApplyResult
  }) {
    notify('异常阈值', payload.anomaly_threshold)
    notify('严重度阈值', payload.severity)
    notify('时间约束', payload.temporal)
    notify('抑制窗口', payload.suppression)
  }

  return { notify, notifyAll }
}
