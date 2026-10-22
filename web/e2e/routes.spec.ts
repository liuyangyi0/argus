import { expect, test } from '@playwright/test'
import { authenticateIfRequired, readApiData } from './support/auth'

const cameraId = process.env.ARGUS_E2E_CAMERA_ID || 'validation_camera_01'
const alertId = process.env.ARGUS_E2E_ALERT_ID
const expectSceneGuardBlocked = process.env.ARGUS_E2E_EXPECT_SCENE_GUARD_BLOCKED === '1'

const coreRoutes = [
  { path: '/overview', title: /总览.*Argus/ },
  { path: '/cameras', title: /摄像头.*Argus/ },
  { path: `/cameras/${encodeURIComponent(cameraId)}`, title: /摄像头详情.*Argus/ },
  { path: '/alerts', title: /告警.*Argus/ },
  { path: '/reports', title: /报表.*Argus/ },
  { path: '/models/comparison', title: /A\/B 对比.*Argus/ },
  { path: '/system/overview', title: /系统概览.*Argus/ },
  { path: '/system/model-status', title: /模型状态.*Argus/ },
]

test.beforeEach(async ({ page }) => {
  await authenticateIfRequired(page)
})

for (const route of coreRoutes) {
  test(`route ${route.path} renders a meaningful page`, async ({ page }) => {
    const consoleErrors: string[] = []
    page.on('console', entry => {
      if (entry.type() === 'error') consoleErrors.push(entry.text())
    })

    await page.goto(route.path)
    await expect(page).toHaveURL(new RegExp(`${route.path.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?:\\?.*)?$`))
    await expect(page).toHaveTitle(route.title)
    await expect(page.locator('main')).toBeVisible()
    await expect(page.locator('main')).not.toBeEmpty()
    await expect(page.locator('vite-error-overlay')).toHaveCount(0)
    expect(consoleErrors, `browser console errors:\n${consoleErrors.join('\n')}`).toEqual([])
  })
}

test('replay loads metadata, signals, trajectory, and video surface independently', async ({ page }) => {
  test.skip(!alertId, 'Set ARGUS_E2E_ALERT_ID to a retained alert with replay evidence')

  const encodedAlertId = encodeURIComponent(alertId!)
  const [metadata, signals, trajectory] = await Promise.all([
    readApiData<{ alert_id: string; status: string; severity: string; frame_count: number }>(
      page,
      `/api/replay/${encodedAlertId}/metadata`,
    ),
    readApiData<{ timestamps: number[]; trajectory_points?: unknown }>(
      page,
      `/api/replay/${encodedAlertId}/signals`,
    ),
    readApiData<{ trajectories: unknown[] }>(
      page,
      `/api/physics/${encodedAlertId}/trajectory`,
    ),
  ])
  expect(metadata).toMatchObject({ alert_id: alertId, status: 'complete', severity: 'medium' })
  expect(metadata.frame_count).toBeGreaterThan(0)
  expect(signals.timestamps).toHaveLength(metadata.frame_count)
  expect(typeof signals.trajectory_points).toBe('object')
  expect(signals.trajectory_points).not.toBeNull()
  expect(Array.isArray(signals.trajectory_points)).toBe(false)
  expect(Object.keys(signals.trajectory_points as Record<string, unknown>).length).toBeGreaterThan(0)
  expect(trajectory.trajectories.length).toBeGreaterThan(0)

  const videoRanges = await page.evaluate(async (videoPath) => {
    const request = async (range: string) => {
      const response = await fetch(videoPath, {
        credentials: 'include',
        headers: { Range: range },
      })
      const payload = await response.arrayBuffer()
      return {
        status: response.status,
        contentRange: response.headers.get('content-range'),
        cacheControl: response.headers.get('cache-control'),
        acceptRanges: response.headers.get('accept-ranges'),
        bytes: payload.byteLength,
      }
    }
    const probe = await request('bytes=0-0')
    const match = probe.contentRange?.match(/^bytes 0-0\/(\d+)$/)
    if (probe.status !== 206 || !match) throw new Error('video range probe failed')
    const total = Number(match[1])
    const prefixEnd = Math.min(total - 1, 1023)
    const suffixBytes = Math.min(total, 256)
    return {
      total,
      prefixEnd,
      suffixBytes,
      probe,
      prefix: await request(`bytes=0-${prefixEnd}`),
      suffix: await request(`bytes=-${suffixBytes}`),
      unsatisfiable: await request(`bytes=${total}-`),
    }
  }, `/api/replay/${encodedAlertId}/video`)
  expect(videoRanges.total).toBeGreaterThan(0)
  expect(videoRanges.probe).toMatchObject({ status: 206, bytes: 1 })
  expect(videoRanges.probe.acceptRanges).toBe('bytes')
  expect(videoRanges.prefix.bytes).toBe(videoRanges.prefixEnd + 1)
  expect(videoRanges.prefix).toMatchObject({
    status: 206,
    bytes: videoRanges.prefixEnd + 1,
  })
  expect(videoRanges.prefix.contentRange).toBe(
    `bytes 0-${videoRanges.prefixEnd}/${videoRanges.total}`,
  )
  expect(videoRanges.prefix.cacheControl).toContain('private')
  expect(videoRanges.suffix).toMatchObject({ status: 206, bytes: videoRanges.suffixBytes })
  expect(videoRanges.suffix.contentRange).toBe(
    `bytes ${videoRanges.total - videoRanges.suffixBytes}-${videoRanges.total - 1}/${videoRanges.total}`,
  )
  expect(videoRanges.unsatisfiable.status).toBe(416)
  expect(videoRanges.unsatisfiable.contentRange).toBe(`bytes */${videoRanges.total}`)
  expect(videoRanges.unsatisfiable.acceptRanges).toBe('bytes')
  expect(videoRanges.unsatisfiable.cacheControl).toBe('private, no-store')

  await page.goto(`/replay/${encodeURIComponent(alertId!)}`)
  await expect(page).toHaveTitle(/录像回放.*Argus/)
  await expect(page.locator('main')).toBeVisible()
  const replay = page.locator('[data-test="replay-root"]')
  await expect(replay).toBeVisible()
  await expect(replay).toHaveAttribute('data-metadata-state', 'ready')
  await expect(replay).toHaveAttribute('data-signals-state', 'ready')
  await expect(replay).toHaveAttribute('data-trajectory-state', 'ready')
  await expect(replay).toHaveAttribute('data-video-state', 'ready', { timeout: 15_000 })
  await expect(page.locator('.replay-signals')).toBeVisible()
  await expect(page.locator('.replay-error')).toHaveCount(0)
  await expect(page.locator('vite-error-overlay')).toHaveCount(0)
})

test('camera detail keeps preview visible while SceneGuard fails closed', async ({ page }, testInfo) => {
  test.skip(
    !expectSceneGuardBlocked,
    'Set ARGUS_E2E_EXPECT_SCENE_GUARD_BLOCKED=1 for the fail-closed safety gate',
  )
  type CameraSafetyDetail = {
    scene_guard?: { enabled?: boolean; usable?: boolean }
    stats?: { frames_analyzed?: number; alerts_emitted?: number } | null
    candidate_diagnostics?: Array<{
      alert_emitted?: boolean
      final_decision?: string
      suppression_reason?: string | null
    }>
  }

  await page.goto(`/cameras/${encodeURIComponent(cameraId)}`)
  await expect(page.locator('.video-container')).toBeVisible()
  await expect.poll(async () => page.locator('.video-container').evaluate((container) => {
    const video = container.querySelector('video')
    if (video instanceof HTMLVideoElement && video.readyState >= 2 && video.videoWidth > 0) {
      return true
    }
    const image = container.querySelector('img')
    return image instanceof HTMLImageElement && image.naturalWidth > 0
  }), { timeout: 15_000 }).toBe(true)
  // Read the authoritative state after the page is visible so the assertion
  // and banner belong to the same runtime window.
  const detail = await readApiData<CameraSafetyDetail>(
    page,
    `/api/cameras/${encodeURIComponent(cameraId)}/detail/json`,
  )

  const banner = page.locator('[data-test="scene-guard-blocked"]')
  expect(detail.scene_guard?.enabled).toBe(true)
  expect(detail.scene_guard?.usable).toBe(false)
  await expect(banner).toBeVisible()
  await expect(banner).toContainText('检测已暂停')
  await expect(banner).toContainText('视频预览保持可用')

  const alertsBefore = await readApiData<{ total: number }>(
    page,
    `/api/alerts/json?camera_id=${encodeURIComponent(cameraId)}&limit=1`,
  )
  const analyzedBefore = detail.stats?.frames_analyzed
  const emittedBefore = detail.stats?.alerts_emitted
  expect(typeof analyzedBefore).toBe('number')
  expect(typeof emittedBefore).toBe('number')

  await page.waitForTimeout(3_000)
  const after = await readApiData<CameraSafetyDetail>(
    page,
    `/api/cameras/${encodeURIComponent(cameraId)}/detail/json`,
  )
  const alertsAfter = await readApiData<{ total: number }>(
    page,
    `/api/alerts/json?camera_id=${encodeURIComponent(cameraId)}&limit=1`,
  )
  expect(after.scene_guard?.usable).toBe(false)
  expect(after.stats?.frames_analyzed).toBe(analyzedBefore)
  expect(after.stats?.alerts_emitted).toBe(emittedBefore)
  expect(alertsAfter.total).toBe(alertsBefore.total)

  const latest = after.candidate_diagnostics?.at(-1)
  expect(latest?.final_decision).toBe('skipped')
  expect(latest?.alert_emitted).toBe(false)
  expect(latest?.suppression_reason).toMatch(/^scene_guard:/)

  await page.screenshot({
    path: testInfo.outputPath('camera-detail-scene-guard.png'),
    fullPage: false,
  })
})
