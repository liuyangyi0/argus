import { expect, test } from '@playwright/test'
import { authenticateIfRequired } from './support/auth'

const errorCases = [
  { status: 403, code: 40301, message: 'E2E 权限拒绝' },
  { status: 404, code: 40401, message: 'E2E 摄像头接口不存在' },
  { status: 409, code: 40901, message: 'E2E 状态冲突' },
  { status: 500, code: 50001, message: 'E2E 服务内部错误' },
]

test.beforeEach(async ({ page }) => {
  await authenticateIfRequired(page)
})

for (const errorCase of errorCases) {
  test(`camera query surfaces HTTP ${errorCase.status} as failure, not empty data`, async ({ page }) => {
    await page.route('**/api/cameras/json', route => route.fulfill({
      status: errorCase.status,
      contentType: 'application/json',
      body: JSON.stringify({
        code: errorCase.code,
        msg: errorCase.message,
        data: { request_id: `e2e-${errorCase.status}` },
      }),
    }))

    await page.goto('/cameras')
    await expect(page.locator('main')).toContainText(errorCase.message)
    await expect(page.locator('main')).not.toContainText('暂无摄像头')
    await expect(page.getByRole('button', {
      name: /^(?:重试加载数据|重试摄像头列表)$/,
    })).toBeVisible()
  })
}

test('HTTP 401 redirects the protected page to login', async ({ page }) => {
  await page.route('**/api/cameras/json', route => route.fulfill({
    status: 401,
    contentType: 'application/json',
    body: JSON.stringify({ code: 40101, msg: 'E2E 会话已过期', data: null }),
  }))

  await page.goto('/cameras')
  await expect(page).toHaveURL(/\/login\?next=/)
})
