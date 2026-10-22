/** Uses only run_local_functional_server.py and its disposable database. */
import { expect, test } from '@playwright/test'
import { authenticateIfRequired, readApiData } from './support/auth'
import './error-states.spec'
import './rbac.spec'

test('ordinary navigation renders without a model, GPU, or connected camera', async ({ page }, testInfo) => {
  test.setTimeout(90_000)
  const pageErrors: string[] = []
  page.on('pageerror', error => pageErrors.push(error.message))
  await authenticateIfRequired(page)
  for (const path of [
    '/overview', '/cameras', '/cameras/local-camera', '/alerts', '/reports',
    '/models/baseline', '/models/training', '/models/registry', '/models/comparison',
    '/models/labeling', '/models/threshold', '/models/collections',
    '/system/overview', '/system/config', '/system/degradation', '/system/model-status',
    '/system/audit', '/system/modules', '/system/classifier', '/system/segmenter',
    '/system/imaging', '/system/cross-camera', '/system/users', '/system/regions',
  ]) {
    await page.goto(path)
    await expect(page.locator('main')).toHaveCount(1)
    await expect(page.locator('main')).toBeVisible()
    await expect(page.locator('main')).not.toBeEmpty()
    await expect(page.locator('vite-error-overlay')).toHaveCount(0)
    await page.waitForLoadState('networkidle')
    expect(pageErrors, `${path}: JavaScript exceptions`).toEqual([])
  }
  await page.screenshot({ path: testInfo.outputPath('local-functional-shell.png') })
})

test('two independent sessions see the same isolated camera and alert state', async ({ browser, baseURL }) => {
  const contexts = await Promise.all([browser.newContext({ baseURL }), browser.newContext({ baseURL })])
  try {
    const pages = await Promise.all(contexts.map(context => context.newPage()))
    await Promise.all(pages.map(page => authenticateIfRequired(page)))
    const cameras = await Promise.all(pages.map(page => (
      readApiData<{ cameras: { camera_id: string }[] }>(page, '/api/cameras/json')
    )))
    expect(cameras[0]?.cameras.map(camera => camera.camera_id)).toEqual(['local-camera'])
    expect(cameras[1]?.cameras).toEqual(cameras[0]?.cameras)
    const response = await pages[0]!.request.post('/api/alerts/local-functional-alert/workflow', {
      data: { status: 'acknowledged', confirmed: true, from_detail_view: true },
    })
    expect(response.ok()).toBe(true)
    await expect.poll(async () => (
      await readApiData<{ workflow_status: string }>(pages[1]!, '/api/alerts/local-functional-alert/detail')
    ).workflow_status).toBe('acknowledged')
  } finally {
    await Promise.all(contexts.map(context => context.close()))
  }
})

test('viewer API permissions are enforced independently of navigation', async ({ page }) => {
  await page.context().clearCookies()
  await authenticateIfRequired(page, { username: 'local-viewer', password: 'Argus-local-only!42' })
  const denied = await page.request.get('/api/users/json')
  expect(denied.status()).toBe(403)
  expect(await denied.json()).toMatchObject({ msg: '权限不足' })
})

test('websocket connection recovers after ordinary network disconnection', async ({ page, context }) => {
  await authenticateIfRequired(page)
  await expect(page.locator('.connection-banner')).toBeHidden()
  await context.setOffline(true)
  await expect(page.locator('.connection-banner')).toBeVisible({ timeout: 10_000 })
  await context.setOffline(false)
  await expect(page.locator('.connection-banner')).toBeHidden({ timeout: 20_000 })
})

test('missing replay evidence shows an explicit unavailable state', async ({ page }) => {
  await authenticateIfRequired(page)
  await page.goto('/replay/local-functional-alert')
  await expect(page.locator('main')).toBeVisible()
  await expect(page.locator('main')).toContainText(/不存在|不可用|失败|未找到|暂无/)
})
