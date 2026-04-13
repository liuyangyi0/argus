import { createRouter, createWebHistory } from 'vue-router'
import BasicLayout from '../layouts/BasicLayout.vue'

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
          meta: { title: '摄像头详情' },
        },
        {
          path: 'alerts',
          name: 'alerts',
          component: () => import('../views/Alerts.vue'),
          meta: { title: '告警' },
        },
        {
          path: 'models',
          name: 'models',
          component: () => import('../views/Models.vue'),
          meta: { title: '模型' },
        },
        {
          path: 'training',
          redirect: '/models?tab=training',
        },
        {
          path: 'system',
          name: 'system',
          component: () => import('../views/System.vue'),
          meta: { title: '系统' },
        },
      ]
    }
  ],
})

router.beforeEach((to) => {
  document.title = `${to.meta.title || 'Argus'} - Argus`
})

export default router
