/**
 * Shared color utilities for consistent anomaly score visualization.
 *
 * Values match the CSS design tokens in style.css:
 *   --red: #e5484d, --amber: #d97706, --blue: #2563eb, --green: #15a34a
 */

/** Map anomaly score (0-1) to a severity color. */
export function scoreColor(score: number): string {
  if (score >= 0.95) return '#e5484d'  // red — critical  (matches --red)
  if (score >= 0.85) return '#d97706'  // amber — high     (matches --amber)
  if (score >= 0.7) return '#d97706'   // amber — medium   (matches --amber)
  return '#2563eb'                      // blue — normal   (matches --blue)
}

/** Severity level to color mapping. */
export const SEVERITY_COLORS: Record<string, string> = {
  high: '#e5484d',
  medium: '#d97706',
  low: '#d97706',
  info: '#2563eb',
}
