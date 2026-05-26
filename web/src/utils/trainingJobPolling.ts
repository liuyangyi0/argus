const LIVE_TRAINING_STATUSES = new Set([
  'pending_confirmation',
  'queued',
  'running',
])

export function hasLiveTrainingJobs(
  jobs: Array<{ status?: string | null }>,
): boolean {
  return jobs.some((job) => LIVE_TRAINING_STATUSES.has(job.status || ''))
}
