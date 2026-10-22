import type { AuxiliaryRuntimeSummary } from '../api/system'

export interface AuxiliaryRuntimePresentation {
  type: 'success' | 'info' | 'warning' | 'error'
  message: string
}

export function describeAuxiliaryRuntime(
  label: string,
  enabled: boolean,
  runtime: AuxiliaryRuntimeSummary,
): AuxiliaryRuntimePresentation {
  const { total_pipelines: total, pipelines_attached: attached, pipelines_loaded: loaded } = runtime
  const details = runtime.details ?? []
  const failed = Math.max(runtime.pipelines_failed ?? 0, details.filter(item => item.state === 'failed').length)
  const degraded = Math.max(runtime.pipelines_degraded ?? 0, details.filter(item => item.degraded).length)
  const detailReady = details.filter(item => item.ready && !item.degraded && !item.busy && item.state === 'ready').length
  const ready = details.length > 0
    ? Math.min(runtime.pipelines_ready ?? detailReady, detailReady)
    : runtime.pipelines_ready
  const busy = details.filter(item => item.busy).length

  if (!enabled) {
    if (busy > 0) {
      return { type: 'warning', message: `${label}配置已关闭，但仍有 ${busy} 路旧任务尚未结束；模块尚未完全停止，不能重复启用` }
    }
    if (failed > 0 || degraded > 0) {
      return { type: 'warning', message: `${label}配置已关闭，但仍有运行时清理故障或未知状态；尚未确认全部停止，请检查逐相机原因并重试停用` }
    }
    if (attached > 0 || loaded > 0) {
      return { type: 'warning', message: `${label}配置已关闭，但仍有 ${attached} 路管线挂载模块；尚未确认全部停止，请检查运行时状态` }
    }
    return { type: 'info', message: `${label}配置已关闭，当前运行管线未挂载模块` }
  }
  if (total === 0) return { type: 'info', message: `${label}配置已启用，当前无运行中的摄像头管线，尚未验证模型加载` }
  if (failed > 0) return { type: 'error', message: `${label}有 ${failed} / ${total} 路管线故障，配置启用不代表运行时可用` }
  if (degraded > 0) return { type: 'warning', message: `${label}有 ${degraded} / ${total} 路管线降级，不能视为目标模型就绪` }
  if (busy > 0) return { type: 'warning', message: `${label}有 ${busy} 路管线任务仍在运行，等待结束期间不应重复提交` }
  if (attached < total) return { type: 'warning', message: `${label}仅 ${attached} / ${total} 路管线已挂载，未全部生效，请检查启用失败原因或重试` }
  if (ready === total) return { type: 'success', message: `${label}模型已在全部 ${total} 路管线就绪（不代表识别效果已验收）` }
  if (ready !== undefined) return { type: 'warning', message: `${label}模型仅 ${ready} / ${total} 路就绪，${loaded} 路已加载；请检查逐相机状态` }
  return { type: 'info', message: `${label}已挂载 ${attached} / ${total} 路、已加载 ${loaded} 路；服务端未提供真实就绪状态，无法确认模型可用` }
}

export function auxiliaryStateLabel(state: string): string {
  const labels: Record<string, string> = {
    ready: '就绪', loaded: '已加载，待核验', idle: '待加载', unloaded: '未加载',
    loading: '加载中', failed: '故障', degraded: '降级', disabled: '已停用',
    busy: '任务运行中', draining: '旧任务排空中', stopping: '停止中', stopped: '已停止', timeout: '超时',
  }
  return labels[state] ?? state
}
