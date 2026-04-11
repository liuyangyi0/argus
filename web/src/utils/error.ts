/**
 * Extract a user-friendly error message from an axios error or generic exception.
 */
export function extractErrorMessage(e: unknown, fallback = '操作失败'): string {
  if (e && typeof e === 'object' && 'response' in e) {
    const data = (e as any).response?.data
    if (data) {
      return data.error || data.detail || data.message || fallback
    }
  }
  if (e instanceof Error) return e.message
  return fallback
}
