import { expect, test } from '@playwright/test'
import { authenticateIfRequired, readApiData } from './support/auth'

test('core monitoring shell renders and keyboard navigation responds', async ({ page }, testInfo) => {
  const consoleErrors: string[] = []
  page.on('console', entry => {
    if (entry.type() === 'error') consoleErrors.push(entry.text())
  })

  await authenticateIfRequired(page)
  await expect(page).toHaveTitle(/Argus/)
  await expect(page.locator('body')).toContainText('Argus')
  await expect(page.locator('vite-error-overlay')).toHaveCount(0)

  await page.keyboard.press('2')
  await expect(page).toHaveURL(/\/cameras(?:\?.*)?$/)
  await expect(page.locator('main')).toBeVisible()
  const bodyOverflow = await page.evaluate(() => (
    document.documentElement.scrollWidth - window.innerWidth
  ))
  expect(bodyOverflow, 'the page shell must not create body-level horizontal overflow').toBeLessThanOrEqual(1)
  await page.screenshot({
    path: testInfo.outputPath('cameras-shell.png'),
    fullPage: false,
  })
  expect(consoleErrors, `browser console errors:\n${consoleErrors.join('\n')}`).toEqual([])
})

test('two independent browser sessions reach the same authoritative overview', async ({ browser, baseURL }) => {
  const firstContext = await browser.newContext({ baseURL })
  const secondContext = await browser.newContext({ baseURL })
  try {
    const [first, second] = await Promise.all([
      firstContext.newPage(),
      secondContext.newPage(),
    ])
    await Promise.all([authenticateIfRequired(first), authenticateIfRequired(second)])
    await expect(first.locator('body')).toContainText('Argus')
    await expect(second.locator('body')).toContainText('Argus')
    await expect(first).toHaveURL(/\/overview(?:\?.*)?$/)
    await expect(second).toHaveURL(/\/overview(?:\?.*)?$/)
    const [firstState, secondState] = await Promise.all([
      readApiData<{ cameras: Array<{ camera_id: string }> }>(first, '/api/cameras/json'),
      readApiData<{ cameras: Array<{ camera_id: string }> }>(second, '/api/cameras/json'),
    ])
    expect(firstState.cameras.map(item => item.camera_id).sort()).toEqual(
      secondState.cameras.map(item => item.camera_id).sort(),
    )
  } finally {
    await Promise.all([firstContext.close(), secondContext.close()])
  }
})

test('websocket disconnect is visible and reconnects after network recovery', async ({ page, context }) => {
  await authenticateIfRequired(page)
  await expect(page.locator('.connection-banner')).toBeHidden()

  await context.setOffline(true)
  await expect(page.locator('.connection-banner')).toBeVisible({ timeout: 10_000 })
  await expect(page.locator('.connection-banner')).toContainText(/重新连接|受限模式/)

  await context.setOffline(false)
  await expect(page.locator('.connection-banner')).toBeHidden({ timeout: 20_000 })
})

test('a missed camera event is reconciled across two sessions within two seconds', async ({
  browser,
  page,
  context,
  baseURL,
}) => {
  const cameraId = process.env.ARGUS_E2E_CAMERA_ID || 'validation_camera_01'
  const controllerContext = await browser.newContext({ baseURL })
  const controller = await controllerContext.newPage()
  let originalMode = 'active'
  try {
    await Promise.all([authenticateIfRequired(page), authenticateIfRequired(controller)])
    await page.goto('/cameras')
    const row = page.locator('tr.camera-row').filter({ hasText: cameraId })
    await expect(row).toBeVisible()

    const detail = await readApiData<{ runtime?: { pipeline_mode?: string } }>(
      controller,
      `/api/cameras/${encodeURIComponent(cameraId)}/detail/json`,
    )
    originalMode = detail.runtime?.pipeline_mode || 'active'
    const targetMode = originalMode === 'maintenance' ? 'learning' : 'maintenance'
    const targetLabel = targetMode === 'maintenance' ? '维护' : '学习中'

    await context.setOffline(true)
    await expect(page.locator('.connection-banner')).toBeVisible({ timeout: 10_000 })
    await controller.evaluate(async ({ id, mode }) => {
      const response = await fetch(`/api/cameras/${encodeURIComponent(id)}/mode`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      })
      const body = await response.json() as { code: number; msg: string }
      if (!response.ok || body.code !== 0) {
        throw new Error(`mode mutation failed: HTTP ${response.status} ${body.msg}`)
      }
    }, { id: cameraId, mode: targetMode })

    const recoveryStarted = Date.now()
    await context.setOffline(false)
    await expect(row).toContainText(targetLabel, { timeout: 5_000 })
    expect(
      Date.now() - recoveryStarted,
      'revision-gap resync must repair the missed event within two seconds',
    ).toBeLessThanOrEqual(2_000)
  } finally {
    if (!controller.isClosed()) {
      await controller.evaluate(async ({ id, mode }) => {
        await fetch(`/api/cameras/${encodeURIComponent(id)}/mode`, {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode }),
        })
      }, { id: cameraId, mode: originalMode }).catch(() => undefined)
    }
    await controllerContext.close()
  }
})
