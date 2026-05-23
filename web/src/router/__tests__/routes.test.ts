import { describe, expect, it } from 'vitest'
import router from '../index'

function routeMeta(name: string) {
  const route = router.getRoutes().find((item) => item.name === name)
  if (!route) throw new Error(`Route not found: ${name}`)
  return route.meta
}

describe('router RBAC metadata', () => {
  it('requires authentication for backend-backed top-level pages', () => {
    for (const name of ['overview', 'cameras', 'alerts', 'reports']) {
      expect(routeMeta(name).requiresAuth).toBe(true)
    }
  })

  it('allows model management pages for admin and engineer roles only', () => {
    for (const name of ['models-baseline', 'models-training', 'models-registry']) {
      expect(routeMeta(name).requiresRole).toEqual(['admin', 'engineer'])
    }
  })

  it('allows config management for admin and engineer roles', () => {
    expect(routeMeta('system-config').requiresRole).toEqual(['admin', 'engineer'])
  })
})
