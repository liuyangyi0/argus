import { api, u } from './client'

// ── Report statistics ──
export const getReportStats = () => api.get('/reports/json').then(u)

// ── Chart data ──
export const getDailyTrend = (days = 30) =>
  api.get('/reports/daily-trend/json', { params: { days } }).then(u)

export const getSeverityDist = () => api.get('/reports/severity-dist/json').then(u)

export const getCameraDist = () => api.get('/reports/camera-dist/json').then(u)

export const getFPTrend = (days = 30) =>
  api.get('/reports/fp-trend/json', { params: { days } }).then(u)
