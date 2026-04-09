<script setup lang="ts">
import { h, ref, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { theme, Layout, Menu, Typography } from 'ant-design-vue'
import {
  DesktopOutlined,
  CameraOutlined,
  BellOutlined,
  ExperimentOutlined,
  SettingOutlined,
} from '@ant-design/icons-vue'
import DegradationBar from './components/DegradationBar.vue'

const router = useRouter()
const route = useRoute()
const collapsed = ref(false)

const selectedKeys = computed(() => {
  const path = route.path
  if (path.startsWith('/cameras')) return ['cameras']
  if (path.startsWith('/alerts')) return ['alerts']
  if (path.startsWith('/models')) return ['models']
  if (path.startsWith('/system')) return ['system']
  return ['overview']
})

const menuItems = [
  { key: 'overview', icon: () => h(DesktopOutlined), label: '值班台', path: '/overview' },
  { key: 'cameras', icon: () => h(CameraOutlined), label: '摄像头', path: '/cameras' },
  { key: 'alerts', icon: () => h(BellOutlined), label: '告警', path: '/alerts' },
  { key: 'models', icon: () => h(ExperimentOutlined), label: '模型管理', path: '/models' },
  { key: 'system', icon: () => h(SettingOutlined), label: '系统', path: '/system' },
]

function onMenuClick({ key }: { key: string | number }) {
  const item = menuItems.find(m => m.key === String(key))
  if (item) router.push(item.path)
}
</script>

<template>
  <a-config-provider
    :theme="{
      algorithm: theme.darkAlgorithm,
      token: {
        colorPrimary: '#3b82f6',
        colorBgContainer: '#1a1a2e',
        colorBgElevated: '#1e1e36',
        colorBgLayout: '#0f0f1a',
        borderRadius: 6,
        fontSize: 14,
      },
    }"
  >
    <Layout style="min-height: 100vh">
      <Layout.Sider
        v-model:collapsed="collapsed"
        :trigger="null"
        collapsible
        :width="220"
        theme="dark"
        style="background: #0f0f1a; border-right: 1px solid #1f2937"
      >
        <div style="padding: 20px 16px; text-align: center; border-bottom: 1px solid #1f2937">
          <Typography.Title
            :level="4"
            style="color: #3b82f6; margin: 0; letter-spacing: 3px"
          >
            {{ collapsed ? 'A' : 'ARGUS' }}
          </Typography.Title>
        </div>
        <Menu
          :selected-keys="selectedKeys"
          mode="inline"
          theme="dark"
          style="background: transparent; border: none; margin-top: 8px"
          :items="menuItems"
          @click="onMenuClick"
        />
        <div
          v-if="!collapsed"
          style="position: absolute; bottom: 16px; left: 16px; right: 16px; font-size: 12px; color: #4a5568"
        >
          v0.2.0
        </div>
      </Layout.Sider>
      <Layout>
        <DegradationBar />
        <Layout.Content style="padding: 24px; overflow-y: auto">
          <router-view v-slot="{ Component }">
            <keep-alive :include="['OverviewPage', 'CamerasPage', 'AlertsPage']">
              <component :is="Component" />
            </keep-alive>
          </router-view>
        </Layout.Content>
      </Layout>
    </Layout>
  </a-config-provider>
</template>
