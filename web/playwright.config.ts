import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { defineConfig } from '@playwright/test'

const baseURL = process.env.ARGUS_E2E_BASE_URL || 'http://127.0.0.1:18080'
const outputDir = process.env.ARGUS_E2E_OUTPUT_DIR
  || join(tmpdir(), 'argus-playwright-results')

export default defineConfig({
  testDir: './e2e',
  outputDir,
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: [['line']],
  timeout: 45_000,
  expect: { timeout: 8_000 },
  use: {
    baseURL,
    ignoreHTTPSErrors: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
  },
  projects: [
    { name: 'desktop-1920', use: { viewport: { width: 1920, height: 1080 } } },
    { name: 'desktop-1440', use: { viewport: { width: 1440, height: 900 } } },
    { name: 'tablet-1024', use: { viewport: { width: 1024, height: 768 } } },
    { name: 'mobile-status', use: { viewport: { width: 390, height: 844 } } },
  ],
})
