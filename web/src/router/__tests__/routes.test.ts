import { describe, expect, it } from 'vitest'
import router from '../index'
import type { NavigationGuard, RouteRecordNormalized } from 'vue-router'
import { createPinia, setActivePinia } from 'pinia'
import { useAuthStore } from '../../stores/useAuthStore'

function routeMeta(name: string) {
  const route = router.getRoutes().find((item) => item.name === name)
  if (!route) throw new Error(`Route not found: ${name}`)
  return route.meta
}

function routeByPath(path: string) {
  const route = router.getRoutes().find((item) => item.path === path)
  if (!route) throw new Error(`Route not found: ${path}`)
  return route
}

function routeByName(name: string) {
  const route = router.getRoutes().find((item) => item.name === name)
  if (!route) throw new Error(`Route not found: ${name}`)
  return route
}

function callBeforeEnter(route: RouteRecordNormalized, to: Record<string, unknown>) {
  const guard = route.beforeEnter
  expect(typeof guard).toBe('function')
  return (guard as NavigationGuard)(
    {
      fullPath: to.path,
      hash: '',
      matched: [{}],
      meta: route.meta,
      name: route.name,
      params: {},
      path: to.path,
      query: {},
      redirectedFrom: undefined,
      ...to,
    } as never,
    {} as never,
    (() => undefined) as never,
  )
}

function setRole(role: 'admin' | 'engineer' | 'operator' | 'viewer') {
  setActivePinia(createPinia())
  const auth = useAuthStore()
  auth.currentUser = { username: role, role }
}

describe('router RBAC metadata', () => {
  it('requires authentication for the core business loop pages', () => {
    for (const name of [
      'overview',
      'cameras',
      'camera-detail',
      'alerts',
      'replay',
      'models-training',
      'models-registry',
      'system-overview',
      'system-config',
      'system-degradation',
      'reports',
    ]) {
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

describe('router core business-loop contract', () => {
  it('keeps the camera-alert-replay-report path available at stable URLs', () => {
    expect(routeByName('cameras').path).toBe('/cameras')
    expect(routeByName('camera-detail').path).toBe('/cameras/:id')
    expect(routeByName('alerts').path).toBe('/alerts')
    expect(routeByName('replay').path).toBe('/replay/:alertId')
    expect(routeByName('reports').path).toBe('/reports')
  })

  it('keeps training, registry, config, and degradation under their product sections', () => {
    expect(routeByName('models-training').path).toBe('/models/training')
    expect(routeByName('models-registry').path).toBe('/models/registry')
    expect(routeByName('system-config').path).toBe('/system/config')
    expect(routeByName('system-degradation').path).toBe('/system/degradation')
  })

  it('preserves legacy training and tab links into the current model/system sections', () => {
    setRole('engineer')
    expect(routeByPath('/training').redirect).toBe('/models/training')

    expect(
      callBeforeEnter(routeByPath('/models'), {
        path: '/models',
        query: { tab: 'registry', camera: 'cam-01' },
        hash: '#release',
      }),
    ).toEqual({
      path: '/models/registry',
      query: { camera: 'cam-01' },
      hash: '#release',
    })

    expect(
      callBeforeEnter(routeByPath('/system'), {
        path: '/system',
        query: { tab: 'config', section: 'detection' },
        hash: '#thresholds',
      }),
    ).toEqual({
      path: '/system/config',
      query: { section: 'detection' },
      hash: '#thresholds',
    })
  })

  it('sends the /models parent route to a role-accessible default child', () => {
    const modelsRoute = routeByPath('/models')

    setRole('engineer')
    expect(
      callBeforeEnter(modelsRoute, {
        path: '/models',
        query: {},
      }),
    ).toEqual({
      path: '/models/baseline',
      query: {},
      hash: '',
    })

    setRole('operator')
    expect(
      callBeforeEnter(modelsRoute, {
        path: '/models',
        query: {},
      }),
    ).toEqual({
      path: '/models/collections',
      query: {},
      hash: '',
    })
  })
})
