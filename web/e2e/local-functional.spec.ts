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

test('managed task configuration, frozen bundle and data workflow require no inference', async ({ page }, testInfo) => {
  await authenticateIfRequired(page)
  await page.goto('/cameras/local-camera')
  const workbench = page.getByRole('region', { name: '多任务工作台' })
  await expect(workbench).toBeVisible()
  const name = `No-compute task ${testInfo.project.name} ${Date.now()}`
  await workbench.locator('[data-test="task-template"]').selectOption('fast_motion.v1')
  await workbench.locator('[data-test="task-name"]').fill(name)
  await workbench.locator('[data-test="create-task"]').click()
  const option = workbench.locator('[data-test="task-select"] option').filter({ hasText: name })
  await expect(option).toHaveCount(1)
  const taskId = await option.getAttribute('value')
  expect(taskId).toBeTruthy()
  await workbench.locator('[data-test="task-select"]').selectOption(taskId!)
  await workbench.getByRole('button', { name: '冻结当前执行包', exact: true }).click()
  await expect(workbench.locator('[data-test="bundle-select"] option')).toHaveCount(2)
  await expect(workbench.getByRole('button', { name: '生产发布：NO-GO', exact: true })).toBeDisabled()
  await workbench.getByRole('button', { name: '新建数据流程', exact: true }).click()
  await expect(workbench.locator('[data-test="workflow-select"] option')).toHaveCount(2)
  await expect(workbench).toContainText('本机不会执行训练')
  const readiness = await readApiData<{ production_ready: boolean }>(page, `/api/analysis-tasks/${taskId}/release-readiness`)
  expect(readiness.production_ready).toBe(false)
  await workbench.screenshot({ path: testInfo.outputPath('managed-task-workbench.png') })
})

test('managed task controls remain unavailable to a viewer', async ({ page }) => {
  await page.context().clearCookies()
  await authenticateIfRequired(page, { username: 'local-viewer', password: 'Argus-local-only!42' })
  await page.goto('/cameras/local-camera')
  await expect(page.getByRole('heading', { name: '分析任务', exact: true })).toBeVisible()
  await expect(page.getByRole('region', { name: '多任务工作台' })).toHaveCount(0)
  const denied = await page.request.post('/api/analysis-tasks', {
    headers: { 'Idempotency-Key': 'viewer-must-not-create-task' },
    data: { camera_id: 'local-camera', template_id: 'fast_motion.v1', name: 'Denied' },
  })
  expect(denied.status()).toBe(403)
})

test('managed durable deployment distinguishes accepted intent from applied state and source ownership', async ({ page }, testInfo) => {
  await authenticateIfRequired(page)
  const created = await page.request.post('/api/analysis-tasks', {
    headers: { 'Idempotency-Key': `durable-ui-${testInfo.project.name}-${Date.now()}` },
    data: { camera_id: 'local-camera', template_id: 'fast_motion.v1', name: 'Mock durable deployment' },
  })
  expect(created.status()).toBe(201)
  const taskId = (await created.json()).data.task_id as string
  let sourceRunning = false
  let deploymentRevision = 0
  let bundleId: string | null = null
  let desiredState = 'stopped'
  let observedState = 'stopped'
  const writes: { body: Record<string, unknown>; key: string | undefined }[] = []
  const legacyCommands: string[] = []
  await page.route(`**/api/analysis-tasks/${taskId}/runtime/*`, async route => {
    legacyCommands.push(route.request().url())
    await route.fulfill({ status: 409, json: { code: 409, msg: 'Direct runtime commands forbidden', data: null } })
  })
  // HTTP transport fixtures only; no camera/worker exists in this server.
  const source = () => ({ camera_id: 'local-camera', state: sourceRunning ? 'running' : 'stopped',
    connected: sourceRunning, running: sourceRunning, cleanup_pending: false })
  await page.route('**/api/cameras/local-camera/source', route => route.fulfill({ json: { code: 0, msg: 'ok', data: source() } }))
  await page.route('**/api/cameras/local-camera/source/start', async route => {
    sourceRunning = true
    await route.fulfill({ json: { code: 0, msg: 'ok', data: source() } })
  })
  await page.route(`**/api/analysis-tasks/${taskId}/deployment`, route => route.fulfill({ json: {
    code: 0, msg: 'ok', data: { document: { task_id: taskId, deployment_revision: deploymentRevision,
      desired_state: desiredState, desired_bundle_id: bundleId, applied_bundle_id: null, state: observedState } },
  } }))
  await page.route(`**/api/analysis-tasks/${taskId}/deployment-operations`, route => route.fulfill({ json: {
    code: 0, msg: 'ok', data: { documents: [] },
  } }))
  await page.route(`**/api/analysis-tasks/${taskId}/deployments`, async route => {
    const body = route.request().postDataJSON() as Record<string, unknown>
    writes.push({ body, key: route.request().headers()['idempotency-key'] })
    expect(body.expected_deployment_revision).toBe(deploymentRevision)
    deploymentRevision += 1
    desiredState = body.action === 'stop' ? 'stopped' : 'running'
    bundleId = body.action === 'stop' ? bundleId : String(body.bundle_id)
    observedState = body.action === 'stop' ? 'stopped' : 'waiting_scene'
    await route.fulfill({ status: 202, json: { code: 0, msg: 'ok', data: {
      document: { task_id: taskId, operation_id: `mock-op-${deploymentRevision}`, status: 'pending' },
    } } })
  })
  await page.goto('/cameras/local-camera')
  const sourcePanel = page.getByRole('region', { name: '独立摄像头源' })
  await sourcePanel.locator('[data-test="start-source"]').click()
  await expect(sourcePanel).toContainText('已连接')
  const workbench = page.getByRole('region', { name: '多任务工作台' })
  await workbench.locator('[data-test="task-select"]').selectOption(taskId)
  await workbench.getByRole('button', { name: '冻结当前执行包', exact: true }).click()
  const select = workbench.locator('[data-test="bundle-select"]')
  await expect(select.locator('option')).toHaveCount(2)
  const selectedId = await select.locator('option').nth(1).getAttribute('value')
  await select.selectOption(selectedId!)
  const panel = workbench.getByRole('region', { name: '任务持久化部署' })
  await panel.locator('[data-test="deploy-task"]').click()
  await expect(panel).toContainText('尚不代表已应用')
  await expect(panel).toContainText('等待场景安全门')
  await expect(panel.locator('dd').filter({ hasText: '未确认' })).toHaveCount(1)
  await expect(workbench.getByRole('button', { name: '生产发布：NO-GO', exact: true })).toBeDisabled()
  await panel.locator('[data-test="stop-task"]').click()
  await expect(panel).toContainText('已停止')
  await expect(sourcePanel).toContainText('已连接')
  expect(writes.map(write => write.body.action)).toEqual(['deploy', 'stop'])
  expect(writes.every(write => Boolean(write.key))).toBe(true)
  expect(legacyCommands).toEqual([])
})

test('managed review blocks failed images and preserves drafts after report updates', async ({ page }, testInfo) => {
  await authenticateIfRequired(page)
  const created = await page.request.post('/api/analysis-tasks', {
    headers: { 'Idempotency-Key': `review-ui-${testInfo.project.name}-${Date.now()}` },
    data: { camera_id: 'local-camera', template_id: 'object_detection.v1', name: 'Mock review only' },
  })
  expect(created.status()).toBe(201)
  const taskId = (await created.json()).data.task_id as string
  let revision = 1
  let imageAvailable = false
  let reviewWrites = 0
  const annotation = { schema: 'argus.task-annotation.v1', coordinate_space: 'normalized', kind: 'detection', objects: [] }
  await page.route(`**/api/analysis-tasks/${taskId}/workflows`, route => route.fulfill({ json: {
    code: 0, msg: 'ok', data: { documents: [{
      workflow_id: 'mock-review', task_revision: 1, revision, state: 'review', manifest: {
        samples: [{ sample_id: 'sample-1', image_sha256: 'b'.repeat(64), identity: { frame_index: 1 } }],
        reviews: { 'sample-1': { decision: revision === 1 ? 'approved' : 'needs_review', notes: revision === 1 ? 'Original note' : 'Updated elsewhere', annotation } },
      },
    }] },
  } }))
  // Synthetic image transport only. Real PNG/SHA checks are covered by the
  // guarded backend tests; no camera, video reader or inference runs here.
  await page.route(`**/api/analysis-tasks/${taskId}/workflows/mock-review/samples/sample-1/image`, route => (
    imageAvailable
      ? route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32" fill="white"/></svg>' })
      : route.fulfill({ status: 503, json: { code: 503, msg: 'Mock image unavailable', data: null } })
  ))
  await page.route(`**/api/analysis-tasks/${taskId}/workflows/mock-review/review`, async route => {
    reviewWrites += 1
    await route.fulfill({ status: 409, json: { code: 409, msg: 'No human review should be submitted', data: null } })
  })
  await page.goto('/cameras/local-camera')
  const workbench = page.getByRole('region', { name: '多任务工作台' })
  await workbench.locator('[data-test="task-select"]').selectOption(taskId)
  const data = workbench.getByRole('region', { name: '任务采集与审核' })
  await data.locator('[data-test="sample-select"]').selectOption('sample-1')
  const approve = data.getByRole('button', { name: '批准此标注', exact: true })
  await expect(data).toContainText('审核图片加载失败')
  await expect(approve).toBeDisabled()
  imageAvailable = true
  await data.getByRole('button', { name: '重试加载图片', exact: true }).click()
  await expect(approve).toBeEnabled()
  const notes = data.getByLabel('审核备注', { exact: true })
  await notes.fill('Unsaved human draft')
  revision = 2
  await data.getByRole('button', { name: '刷新流程', exact: true }).click()
  await expect(data).toContainText('当前审核：待人工复核')
  await expect(notes).toHaveValue('Unsaved human draft')
  await expect(approve).toBeDisabled()
  await data.getByRole('button', { name: '重新载入当前审核', exact: true }).click()
  await expect(notes).toHaveValue('Updated elsewhere')
  expect(reviewWrites).toBe(0)
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

test('tablet submenu supports keyboard navigation and restores focus after Escape', async ({ page }) => {
  const width = page.viewportSize()?.width ?? 0
  test.skip(width < 768 || width >= 1280, 'Exercises the compact tablet sidebar')
  await authenticateIfRequired(page)
  const trigger = page.getByRole('button', { name: '模型管理', exact: true })
  await trigger.focus()
  await trigger.press('Enter')
  const navigation = page.getByRole('navigation', { name: '模型管理导航' })
  await expect(navigation).toBeVisible()
  await expect(page.getByRole('dialog')).toHaveCount(1)
  await expect(trigger).toHaveAttribute('aria-expanded', 'true')
  const lastLink = navigation.getByRole('link', { name: '阈值预览', exact: true })
  await lastLink.focus()
  await lastLink.press('Tab')
  await expect(page.getByRole('button', { name: '关闭导航', exact: true })).toBeFocused()
  await page.keyboard.press('Shift+Tab')
  await expect(lastLink).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(navigation).toBeHidden()
  await expect(trigger).toBeFocused()

  await trigger.press('Enter')
  const labeling = navigation.getByRole('link', { name: '标注队列', exact: true })
  await labeling.focus()
  await labeling.press('Enter')
  await expect(page).toHaveURL(/\/models\/labeling(?:\?.*)?$/)
  await expect(navigation).toBeHidden()
  await expect(page.locator('main')).toBeVisible()

  await page.getByRole('button', { name: '系统设置', exact: true }).click()
  await page.getByRole('navigation', { name: '系统设置导航' })
    .getByRole('link', { name: '审计日志', exact: true }).click()
  await expect(page).toHaveURL(/\/system\/audit(?:\?.*)?$/)
  await expect(page.locator('main')).toBeVisible()
})

test('tablet viewer submenu hides restricted actions while preserving monitoring links', async ({ page }) => {
  const width = page.viewportSize()?.width ?? 0
  test.skip(width < 768 || width >= 1280, 'Exercises the compact tablet sidebar')
  await page.context().clearCookies()
  await authenticateIfRequired(page, { username: 'local-viewer', password: 'Argus-local-only!42' })
  await page.getByRole('button', { name: '模型管理', exact: true }).click()
  const models = page.getByRole('navigation', { name: '模型管理导航' })
  await expect(models).toBeVisible()
  await expect(models.getByRole('link', { name: '模型与发布', exact: true })).toHaveCount(0)
  await expect(models.getByRole('link', { name: '训练与评估', exact: true })).toHaveCount(0)
  await models.getByRole('link', { name: '采集集合', exact: true }).click()
  await expect(page).toHaveURL(/\/models\/collections(?:\?.*)?$/)

  await page.getByRole('button', { name: '系统设置', exact: true }).click()
  const system = page.getByRole('navigation', { name: '系统设置导航' })
  await expect(system.getByRole('link', { name: '配置管理', exact: true })).toHaveCount(0)
  await expect(system.getByRole('link', { name: '审计日志', exact: true })).toHaveCount(0)
  await expect(system.getByRole('link', { name: '用户管理', exact: true })).toHaveCount(0)
  await system.getByRole('link', { name: '降级事件', exact: true }).click()
  await expect(page).toHaveURL(/\/system\/degradation(?:\?.*)?$/)
  await expect(page.locator('main')).toBeVisible()
})
