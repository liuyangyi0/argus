/**
 * Shared color utilities for consistent anomaly score visualization.
 */

/** Map anomaly score (0-1) to a severity color. */
export function scoreColor(score: number): string {
  if (score >= 0.95) return '#ef4444'  // red — critical
  if (score >= 0.85) return '#f97316'  // orange — high
  if (score >= 0.7) return '#f59e0b'   // amber — medium
  return '#3b82f6'                      // blue — normal
}

/** Severity level to color mapping. */
export const SEVERITY_COLORS: Record<string, string> = {
  high: '#ef4444',
  medium: '#f97316',
  low: '#f59e0b',
  info: '#3b82f6',
}
