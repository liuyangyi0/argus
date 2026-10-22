import { expect, type Page } from '@playwright/test'

export type Credentials = {
  username?: string
  password?: string
}

export const defaultCredentials: Credentials = {
  username: process.env.ARGUS_E2E_USERNAME,
  password: process.env.ARGUS_E2E_PASSWORD,
}

export async function authenticateIfRequired(
  page: Page,
  credentials: Credentials = defaultCredentials,
): Promise<void> {
  await page.goto('/overview')
  if (!new URL(page.url()).pathname.startsWith('/login')) return
  if (!credentials.username || !credentials.password) {
    throw new Error(
      'Argus authentication is enabled; configure the matching E2E credentials',
    )
  }
  await page.locator('input[name="username"]').fill(credentials.username)
  await page.locator('input[name="password"]').fill(credentials.password)
  await Promise.all([
    page.waitForURL(url => !url.pathname.startsWith('/login')),
    page.getByRole('button', { name: '登录' }).click(),
  ])
  await expect(page).toHaveURL(/\/overview(?:\?.*)?$/)
}

export async function readApiData<T>(page: Page, path: string): Promise<T> {
  return page.evaluate(async (apiPath) => {
    const response = await fetch(apiPath, { credentials: 'include' })
    const body = await response.json() as { code: number; msg: string; data: T }
    if (!response.ok || body.code !== 0) {
      throw new Error(`API ${apiPath} failed: HTTP ${response.status} ${body.msg}`)
    }
    return body.data
  }, path)
}
