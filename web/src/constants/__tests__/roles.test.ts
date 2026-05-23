import { describe, expect, it } from 'vitest'
import { USER_ROLE_OPTIONS } from '../roles'

describe('role constants', () => {
  it('exposes the canonical four dashboard roles', () => {
    expect(USER_ROLE_OPTIONS.map((role) => role.value)).toEqual([
      'admin',
      'engineer',
      'operator',
      'viewer',
    ])
  })
})
