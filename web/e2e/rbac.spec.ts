import { expect, test } from '@playwright/test'
import { authenticateIfRequired, readApiData } from './support/auth'

const viewer = {
  username: process.env.ARGUS_E2E_VIEWER_USERNAME,
  password: process.env.ARGUS_E2E_VIEWER_PASSWORD,
}

test('viewer can monitor but cannot enter production model management', async ({ browser, baseURL }, testInfo) => {
  test.skip(
    !viewer.username || !viewer.password,
    'Configure ARGUS_E2E_VIEWER_USERNAME/PASSWORD for the RBAC browser gate',
  )
  const context = await browser.newContext({
    baseURL,
    viewport: testInfo.project.use.viewport ?? undefined,
  })
  try {
    const page = await context.newPage()
    await authenticateIfRequired(page, viewer)
    await page.goto('/cameras')
    await expect(page.locator('main')).toBeVisible()

    await page.goto('/models/registry')
    await expect(page).not.toHaveURL(/\/models\/registry/)
    await expect(page.locator('body')).toContainText(/权限不足|模型|总览/)
  } finally {
    await context.close()
  }
})

test('admin identity envelope permits production model management', async ({ page }) => {
  await authenticateIfRequired(page)
  const identity = await readApiData<{ username: string; role: string }>(page, '/api/me')
  expect(['admin', 'engineer']).toContain(identity.role)

  await page.goto('/models/registry')
  await expect(page).toHaveURL(/\/models\/registry(?:\?.*)?$/)
  await expect(page.locator('main')).toContainText(/模型|发布/)
})

test('logout clears the HttpOnly session before returning to login', async ({ page }) => {
  await authenticateIfRequired(page)
  const context = page.context()
  const sessionBefore = (await context.cookies()).find(cookie => cookie.name === 'argus_session')
  expect(sessionBefore).toMatchObject({ httpOnly: true, sameSite: 'Lax' })

  await page.getByRole('button', { name: '打开用户菜单' }).click()
  await page.getByText('登出', { exact: true }).click()
  await expect(page).toHaveURL(/\/login(?:\?.*)?$/)
  expect((await context.cookies()).some(cookie => cookie.name === 'argus_session')).toBe(false)

  await page.goto('/cameras')
  await expect(page).toHaveURL(/\/login\?next=/)
})
