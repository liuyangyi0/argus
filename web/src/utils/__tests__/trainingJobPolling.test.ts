import { describe, expect, it } from 'vitest'
import { hasLiveTrainingJobs } from '../trainingJobPolling'

describe('hasLiveTrainingJobs', () => {
  it('returns true for jobs still moving through the queue', () => {
    expect(hasLiveTrainingJobs([{ status: 'pending_confirmation' }])).toBe(true)
    expect(hasLiveTrainingJobs([{ status: 'queued' }])).toBe(true)
    expect(hasLiveTrainingJobs([{ status: 'running' }])).toBe(true)
  })

  it('returns false once all jobs are terminal', () => {
    expect(hasLiveTrainingJobs([
      { status: 'complete' },
      { status: 'failed' },
      { status: 'rejected' },
      { status: null },
    ])).toBe(false)
  })
})
