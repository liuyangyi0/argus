import { effectScope, reactive, shallowReactive, watch } from 'vue'
import type { TaskCandidateImportRequest } from '../types/generated-openapi'

type Identity = { currentUser: { username: string; role: string } | null }
type CandidateDraftBase = {
  readonly taskId: string; readonly cameraId: string; readonly actor: string
  readonly workflowId: string; readonly binding: string; readonly manifestSha: string
  readonly body: TaskCandidateImportRequest
  phase: 'prepared' | 'unknown' | 'registered'; error: string; artifactId: string
}
export type TrainingCandidateDraft = CandidateDraftBase & {
  readonly origin: 'training'; readonly jobId: string; readonly jobIdentity: string
}
export type ManualCandidateDraft = CandidateDraftBase & { readonly origin: 'manual' }
export type TaskCandidateDraft = TrainingCandidateDraft | ManualCandidateDraft
type CandidateLease = { current: () => boolean; release: () => void }
export const MAX_TRAINING_CANDIDATE_DRAFTS = 128

function freezeRequest(body: TaskCandidateImportRequest): TaskCandidateImportRequest {
  const copy = JSON.parse(JSON.stringify(body)) as TaskCandidateImportRequest
  const freeze = (value: unknown) => {
    if (!value || typeof value !== 'object' || Object.isFrozen(value)) return
    Object.values(value).forEach(freeze); Object.freeze(value)
  }
  freeze(copy)
  return copy
}

function createLedger(auth: Identity) {
  // This owner deliberately outlives any camera/workbench component. It is
  // scoped to the AuthStore (therefore its Pinia app), never a process-wide user.
  const drafts = shallowReactive(new Map<string, TaskCandidateDraft>())
  const claims = shallowReactive(new Map<string, symbol>())
  let actor = auth.currentUser?.username ?? '', epoch = 0
  const permitted = () => {
    const user = auth.currentUser
    return !!actor && user?.username === actor && ['admin', 'engineer'].includes(user.role)
  }
  const key = (cameraId: string, taskId: string) => JSON.stringify([actor, cameraId, taskId])
  const resourceKey = (cameraId: string, taskId: string) => JSON.stringify([cameraId, taskId])
  const scope = effectScope(true)
  scope.run(() => watch([() => auth.currentUser?.username ?? '', () => auth.currentUser?.role ?? ''], ([nextActor]) => {
    epoch += 1
    if (nextActor !== actor) {
      actor = nextActor ?? ''; drafts.clear()
    }
    // Outstanding transport claims survive logout until their own finally.
    // They reveal no previous request, and a new session cannot duplicate it.
  }, { flush: 'sync' }))
  const get = (cameraId: string, taskId: string) => permitted() ? drafts.get(key(cameraId, taskId)) ?? null : null
  function prepare(input: TrainingCandidateDraft): TrainingCandidateDraft | null
  function prepare(input: ManualCandidateDraft): ManualCandidateDraft | null
  function prepare(input: TaskCandidateDraft): TaskCandidateDraft | null {
    if (!permitted() || input.actor !== actor) return null
    const id = key(input.cameraId, input.taskId), prior = drafts.get(id)
    if (prior) {
      // A second view can reuse, but never replace, another view's request.
      const identity = (draft: TaskCandidateDraft) => JSON.stringify([draft.origin, draft.taskId, draft.cameraId, draft.actor,
        draft.origin === 'training' ? draft.jobId : null, draft.workflowId,
        draft.origin === 'training' ? draft.jobIdentity : null, draft.binding, draft.manifestSha, draft.body])
      return identity(prior) === identity(input) ? prior : null
    }
    if (drafts.size >= MAX_TRAINING_CANDIDATE_DRAFTS || claims.size >= MAX_TRAINING_CANDIDATE_DRAFTS
      || claims.has(resourceKey(input.cameraId, input.taskId))) return null
    const draft = reactive({ ...input, body: freezeRequest(input.body) }) as TaskCandidateDraft
    drafts.set(id, draft)
    return draft
  }
  return {
    get,
    submitting: (cameraId: string, taskId: string) => claims.has(resourceKey(cameraId, taskId)),
    full: () => drafts.size >= MAX_TRAINING_CANDIDATE_DRAFTS || claims.size >= MAX_TRAINING_CANDIDATE_DRAFTS,
    prepare,
    acquire(draft: TaskCandidateDraft): CandidateLease | null {
      if (!permitted() || get(draft.cameraId, draft.taskId) !== draft || draft.phase === 'registered') return null
      const id = resourceKey(draft.cameraId, draft.taskId)
      if (claims.has(id)) return null
      const token = Symbol('candidate-post'), acquiredEpoch = epoch
      claims.set(id, token)
      return {
        current: () => acquiredEpoch === epoch && permitted() && claims.get(id) === token
          && get(draft.cameraId, draft.taskId) === draft,
        release: () => { if (claims.get(id) === token) claims.delete(id) },
      }
    },
    clear(draft: TaskCandidateDraft): boolean {
      if (get(draft.cameraId, draft.taskId) !== draft || draft.phase === 'unknown'
        || claims.has(resourceKey(draft.cameraId, draft.taskId))) return false
      return drafts.delete(key(draft.cameraId, draft.taskId))
    },
  }
}

const owners = new WeakMap<Identity, ReturnType<typeof createLedger>>()
export function useTrainingCandidateLedger(auth: Identity | null) {
  if (!auth) return createLedger({ currentUser: null })
  let owner = owners.get(auth)
  if (!owner) { owner = createLedger(auth); owners.set(auth, owner) }
  return owner
}
