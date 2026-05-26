import type { ApiResponse, ReportStats } from '../types/api'
import { api, u, unwrap } from './client'

// ── Report statistics ──
export const getReportStats = (days?: number) =>
  api
    .get<ApiResponse<ReportStats>>('/reports/json', {
      params: days ? { days } : undefined,
    })
    .then(unwrap)

// ── Chart data ──
export const getDailyTrend = (days = 30) =>
  api.get('/reports/daily-trend/json', { params: { days } }).then(u)

export const getSeverityDist = (days?: number) =>
  api
    .get('/reports/severity-dist/json', { params: days ? { days } : undefined })
    .then(u)

export const getCameraDist = (days?: number) =>
  api
    .get('/reports/camera-dist/json', { params: days ? { days } : undefined })
    .then(u)

export const getFPTrend = (days = 30) =>
  api.get('/reports/fp-trend/json', { params: { days } }).then(u)

// ── Compliance report download ──

/**
 * Download a compliance report file (CSV or PDF).
 * Uses blob download instead of the normal JSON API client.
 */
export async function downloadComplianceReport(days = 30, format: 'csv' | 'pdf' = 'csv'): Promise<void> {
  const res = await api.get('/reports/compliance', {
    params: { days, format },
    responseType: 'blob',
  })

  // Check if response is a JSON error (api_error returns JSON, not a file)
  const blob: Blob = res.data
  if (blob.type.includes('application/json')) {
    const text = await blob.text()
    const body = JSON.parse(text)
    throw new Error(body.msg || '生成报告失败')
  }

  // Extract filename from Content-Disposition or fall back to default
  const disposition = res.headers['content-disposition'] || ''
  const match = disposition.match(/filename="?([^";\s]+)"?/)
  const filename = match?.[1] || `compliance_report.${format}`

  // Trigger browser download via hidden anchor
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
