/**
 * Shared time-formatting utilities.
 */

/**
 * Format a timestamp as a human-readable relative time string.
 *
 * @param ts  ISO string, epoch-seconds number, or undefined
 * @param short  When true returns compact labels (5s, 3m, 2h, 1d);
 *               when false returns full Chinese labels (5秒前, 3分钟前).
 */
export function formatRelativeTime(
  ts: string | number | undefined,
  short = false,
): string {
  if (!ts) return short ? '' : '--'
  const date = typeof ts === 'string' ? new Date(ts) : new Date(ts * 1000)
  const diff = Math.floor((Date.now() - date.getTime()) / 1000)
  if (diff < 10) return '刚刚'
  if (short) {
    if (diff < 60) return `${diff}s`
    if (diff < 3600) return `${Math.floor(diff / 60)}m`
    if (diff < 86400) return `${Math.floor(diff / 3600)}h`
    return `${Math.floor(diff / 86400)}d`
  }
  if (diff < 60) return `${diff}秒前`
  if (diff < 3600) return `${Math.floor(diff / 60)}分钟前`
  if (diff < 86400) return `${Math.floor(diff / 3600)}小时前`
  return `${Math.floor(diff / 86400)}天前`
}

/**
 * Format a timestamp as absolute local time: "HH:mm:ss MM-DD".
 *
 * @param ts  ISO string, epoch-seconds number, or undefined
 */
export function formatTimestamp(ts: string | number | undefined): string {
  if (!ts) return '--'
  const date = typeof ts === 'string' ? new Date(ts) : new Date(ts * 1000)
  const HH = String(date.getHours()).padStart(2, '0')
  const mm = String(date.getMinutes()).padStart(2, '0')
  const ss = String(date.getSeconds()).padStart(2, '0')
  const MM = String(date.getMonth() + 1).padStart(2, '0')
  const DD = String(date.getDate()).padStart(2, '0')
  return `${HH}:${mm}:${ss} ${MM}-${DD}`
}

/**
 * Format a seconds count as "mm:ss" for video playback display.
 */
export function formatPlaybackTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/**
 * Format an epoch-seconds start time as a duration string (e.g. "3分钟前开始").
 */
export function formatDuration(startedAt: number): string {
  const seconds = Math.floor(Date.now() / 1000 - startedAt)
  if (seconds < 0) return '刚刚'
  if (seconds < 60) return `${seconds}秒前开始`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}分钟前开始`
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  return m > 0 ? `${h}h${m}m前开始` : `${h}小时前开始`
}
