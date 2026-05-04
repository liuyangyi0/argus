import { createRouter, createWebHistory } from 'vue-router'
import type { RouteLocationNormalized } from 'vue-router'
import BasicLayout from '../layouts/BasicLayout.vue'

// Map legacy `?tab=` query values to new child routes
const MODELS_TAB_TO_PATH: Record<string, string> = {
  baselines: 'baseline',
  baseline: 'baseline',
  training: 'training',
  models: 'registry',
  registry: 'registry',
  comparison: 'comparison',
  labeling: 'labeling',
  threshold: 'threshold',
}

const SYSTEM_TAB_TO_PATH: Record<string, string> = {
  overview: 'overview',
  'model-status': 'model-status',
  config: 'config',
  audit: 'audit',
  degradation: 'degradation',
  modules: 'modules',
  classifier: 'classifier',
  segmenter: 'segmenter',
  imaging: 'imaging',
  'cross-camera': 'cross-camera',
  users: 'users',
  regions: 'regions',
}

function redirectFromTabQuery(prefix: string, mapping: Record<string, string>, defaultChild: string) {
  return (to: RouteLocationNormalized) => {
    const tab = typeof to.query.tab === 'string' ? to.query.tab : ''
    const child = mapping[tab] || defaultChild
    const { tab: _omit, ...rest } = to.query
    return { path: `${prefix}/${child}`, query: rest, hash: to.hash }
  }
}

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      component: BasicLayout,
      redirect: '/overview',
      children: [
        {
          path: 'overview',
          name: 'overview',
          component: () => import('../views/Overview.vue'),
          meta: { title: '总览' },
        },
        {
          path: 'cameras',
          name: 'cameras',
          component: () => import('../views/Cameras.vue'),
          meta: { title: '摄像头' },
        },
        {
          path: 'cameras/:id',
          name: 'camera-detail',
          component: () => import('../views/CameraDetail.vue'),
          meta: { title: '摄像头详情', requiresAuth: true },
        },
        {
          path: 'alerts',
          name: 'alerts',
          component: () => import('../views/Alerts.vue'),
          meta: { title: '告警' },
        },
        {
          path: 'reports',
          name: 'reports',
          component: () => import('../views/Reports.vue'),
          meta: { title: '报表' },
        },
        {
          path: 'models',
          component: () => import('../views/Models.vue'),
          meta: { title: '模型', requiresAuth: true },
          beforeEnter: (to) => {
            // Only redirect when targeting the parent itself (no child segment)
            // and a legacy ?tab= param needs to be honored, OR no child at all (default).
            if (to.matched.length && to.path.replace(/\/$/, '') === '/models') {
              return redirectFromTabQuery('/models', MODELS_TAB_TO_PATH, 'baseline')(to)
            }
            return true
          },
          children: [
            {
              path: 'baseline',
              name: 'models-baseline',
              component: () => import('../views/models/BaselineView.vue'),
              meta: { title: '基线管理', requiresAuth: true, requiresRole: ['admin', 'operator'] },
            },
            {
              path: 'training',
              name: 'models-training',
              component: () => import('../views/models/TrainingView.vue'),
              meta: { title: '训练与评估', requiresAuth: true, requiresRole: ['admin', 'operator'] },
            },
            {
              path: 'registry',
              name: 'models-registry',
              component: () => import('../views/models/ModelsRegistryView.vue'),
              meta: { title: '模型与发布', requiresAuth: true, requiresRole: ['admin', 'operator'] },
            },
            {
              path: 'comparison',
              name: 'models-comparison',
              component: () => import('../views/models/ComparisonView.vue'),
              meta: { title: 'A/B 对比', requiresAuth: true },
            },
            {
              path: 'labeling',
              name: 'models-labeling',
              component: () => import('../views/models/LabelingView.vue'),
              meta: { title: '标注队列', requiresAuth: true },
            },
            {
              path: 'threshold',
              name: 'models-threshold',
              component: () => import('../views/models/ThresholdView.vue'),
              meta: { title: '阈值预览', requiresAuth: true },
            },
            {
              path: 'collections',
              name: 'models-collections',
              component: () => import('../views/CollectionsView.vue'),
              meta: { title: '采集集合', requiresAuth: true },
            },
          ],
        },
        {
          path: 'training',
          redirect: '/models/training',
        },
        {
          path: 'system',
          component: () => import('../views/System.vue'),
          meta: { title: '系统', requiresAuth: true },
          beforeEnter: (to) => {
            if (to.matched.length && to.path.replace(/\/$/, '') === '/system') {
              return redirectFromTabQuery('/system', SYSTEM_TAB_TO_PATH, 'overview')(to)
            }
            return true
          },
          children: [
            {
              path: 'overview',
              name: 'system-overview',
              component: () => import('../views/system/OverviewView.vue'),
              meta: { title: '系统概览', requiresAuth: true },
            },
            {
              path: 'model-status',
              name: 'system-model-status',
              component: () => import('../views/system/ModelStatusView.vue'),
              meta: { title: '模型状态', requiresAuth: true },
            },
            {
              path: 'config',
              name: 'system-config',
              component: () => import('../views/system/ConfigView.vue'),
              meta: { title: '配置管理', requiresAuth: true, requiresRole: ['admin'] },
            },
            {
              path: 'audit',
              name: 'system-audit',
              component: () => import('../views/system/AuditView.vue'),
              meta: { title: '审计日志', requiresAuth: true, requiresRole: ['admin'] },
            },
            {
              path: 'degradation',
              name: 'system-degradation',
              component: () => import('../views/system/DegradationView.vue'),
              meta: { title: '降级事件', requiresAuth: true },
            },
            {
              path: 'modules',
              name: 'system-modules',
              component: () => import('../views/system/ModulesView.vue'),
              meta: { title: '功能模块', requiresAuth: true },
            },
            {
              path: 'classifier',
              name: 'system-classifier',
              component: () => import('../views/system/ClassifierView.vue'),
              meta: { title: '分类器', requiresAuth: true },
            },
            {
              path: 'segmenter',
              name: 'system-segmenter',
              component: () => import('../views/system/SegmenterView.vue'),
              meta: { title: '分割器', requiresAuth: true },
            },
            {
              path: 'imaging',
              name: 'system-imaging',
              component: () => import('../views/system/ImagingView.vue'),
              meta: { title: '多模态成像', requiresAuth: true },
            },
            {
              path: 'cross-camera',
              name: 'system-cross-camera',
              component: () => import('../views/system/CrossCameraView.vue'),
              meta: { title: '跨相机', requiresAuth: true },
            },
            {
              path: 'users',
              name: 'system-users',
              component: () => import('../views/system/UsersView.vue'),
              meta: { title: '用户管理', requiresAuth: true, requiresRole: ['admin'] },
            },
            {
              path: 'regions',
              name: 'system-regions',
              component: () => import('../views/system/RegionsView.vue'),
              meta: { title: '区域管理', requiresAuth: true },
            },
          ],
        },
        {
          path: 'replay/:alertId',
          name: 'replay',
          component: () => import('../views/ReplayView.vue'),
          meta: { title: '录像回放', requiresAuth: true },
        },
        {
          path: 'replay/:alertId/storyboard',
          name: 'replay-storyboard',
          component: () => import('../views/StoryboardReplay.vue'),
          meta: { title: '多机位回放', requiresAuth: true },
        },
      ]
    }
  ],
})

router.beforeEach((to) => {
  document.title = `${to.meta.title || 'Argus'} - Argus`
  // future hook: when /api/me available, check meta.requiresRole here
  if (to.meta.requiresRole) { /* no-op until /api/me lands; axios 401/403 enforces today */ }
})

export default router
