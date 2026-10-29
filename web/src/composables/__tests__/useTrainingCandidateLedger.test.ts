import { effectScope, reactive } from 'vue'
import { describe, expect, it } from 'vitest'
import { MAX_TRAINING_CANDIDATE_DRAFTS, useTrainingCandidateLedger, type TrainingCandidateDraft } from '../useTrainingCandidateLedger'

const identity = () => reactive({ currentUser: { username: 'engineer', role: 'engineer' } as { username: string; role: string } | null })
const input = (overrides: Partial<TrainingCandidateDraft> = {}): TrainingCandidateDraft => ({
  origin: 'training',
  taskId: 'task-a', cameraId: 'camera-a', actor: 'engineer', jobId: 'job-a', workflowId: 'workflow-a',
  jobIdentity: 'frozen-job', binding: 'frozen-binding', manifestSha: 'd'.repeat(64),
  body: { expected_revision: 3, expected_package_sha256: 'c'.repeat(64),
    candidate: { relative_path: 'job-a/model.onnx', artifact_sha256: 'e'.repeat(64), descriptor: { labels: ['box'] } } },
  phase: 'prepared', error: '', artifactId: '', ...overrides,
})

describe('training candidate session ledger', () => {
  it('outlives a component effect scope without carrying a confirmation bit', () => {
    const auth = identity(), scope = effectScope()
    const first = scope.run(() => useTrainingCandidateLedger(auth))!
    const draft = first.prepare(input())!
    draft.phase = 'unknown'; scope.stop()
    const remounted = useTrainingCandidateLedger(auth)
    expect(remounted.get('camera-a', 'task-a')).toBe(draft)
    expect(remounted.get('camera-a', 'task-a')?.body.expected_revision).toBe(3)
    expect('confirmed' in draft).toBe(false)
    auth.currentUser = null
    expect(remounted.get('camera-a', 'task-a')).toBeNull()
  })

  it('does not overwrite a request or allow two views to acquire the same task', () => {
    const auth = identity(), first = useTrainingCandidateLedger(auth), second = useTrainingCandidateLedger(auth)
    const draft = first.prepare(input())!
    expect(second.prepare(input())).toBe(draft)
    expect(second.prepare(input({ jobIdentity: 'another-result' }))).toBeNull()
    const lease = first.acquire(draft)!
    draft.phase = 'unknown'
    expect(second.submitting('camera-a', 'task-a')).toBe(true)
    expect(second.acquire(draft)).toBeNull()
    expect(second.clear(draft)).toBe(false)
    lease.release()
    expect(second.get('camera-a', 'task-a')?.phase).toBe('unknown')
    const retry = second.acquire(draft)!
    lease.release()
    expect(retry.current()).toBe(true)
    expect(second.submitting('camera-a', 'task-a')).toBe(true)
    retry.release()
  })

  it('detaches and deeply freezes the candidate request while preserving mutable outcome state', () => {
    const ledger = useTrainingCandidateLedger(identity()), source = input(), draft = ledger.prepare(source)!
    source.body.expected_revision = 99
    expect(draft.body.expected_revision).toBe(3)
    expect(Object.isFrozen(draft.body)).toBe(true)
    expect(Object.isFrozen(draft.body.candidate)).toBe(true)
    expect(Object.isFrozen(draft.body.candidate.descriptor)).toBe(true)
    draft.phase = 'unknown'; draft.error = 'response lost'
    expect(ledger.get('camera-a', 'task-a')?.error).toBe('response lost')
  })

  it('isolates tasks, cameras and separate application/auth owners', () => {
    const auth = identity(), ledger = useTrainingCandidateLedger(auth)
    const draft = ledger.prepare(input())!
    expect(ledger.get('camera-a', 'task-b')).toBeNull()
    expect(ledger.get('camera-b', 'task-a')).toBeNull()
    expect(useTrainingCandidateLedger(identity()).get('camera-a', 'task-a')).toBeNull()
    expect(ledger.prepare(input({ cameraId: 'camera-b' }))).not.toBeNull()
    const lease = ledger.acquire(draft)!
    expect(lease).not.toBeNull()
    expect(ledger.submitting('camera-b', 'task-a')).toBe(false)
    lease.release()
  })

  it('clears visible drafts on logout but retains the old transport claim until its own release', () => {
    const auth = identity(), ledger = useTrainingCandidateLedger(auth), draft = ledger.prepare(input())!
    const old = ledger.acquire(draft)!
    draft.phase = 'unknown'; auth.currentUser = null
    expect(old.current()).toBe(false)
    auth.currentUser = { username: 'engineer', role: 'engineer' }
    expect(ledger.get('camera-a', 'task-a')).toBeNull()
    expect(ledger.prepare(input())).toBeNull()
    old.release()
    const fresh = ledger.prepare(input())!
    expect(fresh).not.toBe(draft)
    expect(fresh.phase).toBe('prepared')
  })

  it('invalidates an old lease on role changes and never shares a previous actor’s draft', () => {
    const auth = identity(), ledger = useTrainingCandidateLedger(auth), draft = ledger.prepare(input())!
    const lease = ledger.acquire(draft)!
    auth.currentUser = { username: 'engineer', role: 'operator' }
    expect(ledger.get('camera-a', 'task-a')).toBeNull()
    auth.currentUser = { username: 'engineer', role: 'engineer' }
    expect(lease.current()).toBe(false)
    lease.release()
    expect(ledger.get('camera-a', 'task-a')).toBe(draft)
    auth.currentUser = { username: 'another-engineer', role: 'engineer' }
    expect(ledger.get('camera-a', 'task-a')).toBeNull()
    expect(ledger.prepare(input())).toBeNull()
    auth.currentUser = { username: 'engineer', role: 'engineer' }
    expect(ledger.get('camera-a', 'task-a')).toBeNull()
  })

  it('bounds retained drafts without evicting unknown requests and permits explicit cleanup of known drafts', () => {
    const ledger = useTrainingCandidateLedger(identity()), first = ledger.prepare(input())!
    first.phase = 'unknown'
    for (let index = 1; index < MAX_TRAINING_CANDIDATE_DRAFTS; index++) {
      expect(ledger.prepare(input({ taskId: `task-${index}` }))).not.toBeNull()
    }
    expect(ledger.full()).toBe(true)
    expect(ledger.prepare(input({ taskId: 'overflow-task' }))).toBeNull()
    expect(ledger.get('camera-a', 'task-a')).toBe(first)
    expect(ledger.clear(first)).toBe(false)
    expect(ledger.clear(ledger.get('camera-a', 'task-1')!)).toBe(true)
    expect(ledger.prepare(input({ taskId: 'overflow-task' }))).not.toBeNull()
  })

  it('keeps an unknown request and its lease through ordinary identical personal identity refreshes', () => {
    const auth = identity(), ledger = useTrainingCandidateLedger(auth), draft = ledger.prepare(input())!
    const lease = ledger.acquire(draft)!
    draft.phase = 'unknown'
    auth.currentUser = { username: 'engineer', role: 'engineer' }
    expect(ledger.get('camera-a', 'task-a')).toBe(draft)
    expect(lease.current()).toBe(true)
    lease.release()
  })

  it('shares one claim and preserves origin between manual and trained candidate entrances', () => {
    const ledger = useTrainingCandidateLedger(identity()), trained = input()
    const manual = ledger.prepare({ origin: 'manual', taskId: trained.taskId, cameraId: trained.cameraId,
      actor: trained.actor, workflowId: trained.workflowId, binding: trained.binding, manifestSha: trained.manifestSha,
      body: trained.body, phase: 'prepared', error: '', artifactId: '' })!
    expect(manual.origin).toBe('manual')
    expect(ledger.prepare(trained)).toBeNull()
    const lease = ledger.acquire(manual)!
    manual.phase = 'unknown'
    expect(ledger.submitting('camera-a', 'task-a')).toBe(true)
    expect(ledger.acquire(manual)).toBeNull()
    expect(ledger.clear(manual)).toBe(false)
    lease.release()
    expect(ledger.get('camera-a', 'task-a')).toBe(manual)
    expect(ledger.prepare(trained)).toBeNull()
  })
})
