<script setup lang="ts">
import { h, ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { theme, Layout, Menu, Typography, Tooltip } from 'ant-design-vue'
import {
  DesktopOutlined,
  CameraOutlined,
  BellOutlined,
  ExperimentOutlined,
  SettingOutlined,
  BulbOutlined,
  BulbFilled,
} from '@ant-design/icons-vue'
import DegradationBar from './components/DegradationBar.vue'
import { useThemeStore } from './stores/theme'

const router = useRouter()
const route = useRoute()
const collapsed = ref(false)
const themeStore = useThemeStore()

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

const siderBg = computed(() => themeStore.isDark ? '#0f0f1a' : '#ffffff')
const siderBorder = computed(() => themeStore.isDark ? '#1f2937' : '#e5e7eb')
const versionColor = computed(() => themeStore.isDark ? '#4a5568' : '#9ca3af')

// Mobile: auto-collapse sidebar on small screens
const isMobile = ref(false)
function checkMobile() {
  isMobile.value = window.innerWidth < 768
  if (isMobile.value) collapsed.value = true
}
onMounted(() => {
  checkMobile()
  window.addEventListener('resize', checkMobile)
})
onUnmounted(() => window.removeEventListener('resize', checkMobile))
</script>

<template>
  <a-config-provider :theme="themeStore.themeConfig">
    <Layout style="min-height: 100vh">
      <Layout.Sider
        v-model:collapsed="collapsed"
        :trigger="null"
        collapsible
        :width="220"
        :theme="themeStore.isDark ? 'dark' : 'light'"
        :style="{ background: siderBg, borderRight: `1px solid ${siderBorder}` }"
      >
        <div :style="{ padding: '20px 16px', textAlign: 'center', borderBottom: `1px solid ${siderBorder}` }">
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
          :theme="themeStore.isDark ? 'dark' : 'light'"
          style="background: transparent; border: none; margin-top: 8px"
          :items="menuItems"
          @click="onMenuClick"
        />

        <!-- Theme toggle -->
        <div
          :style="{
            position: 'absolute',
            bottom: collapsed ? '16px' : '40px',
            left: '16px',
            right: '16px',
            textAlign: 'center',
          }"
        >
          <Tooltip :title="themeStore.isDark ? '切换亮色主题' : '切换暗色主题'">
            <a-button
              type="text"
              shape="circle"
              @click="themeStore.toggle()"
            >
              <template #icon>
                <BulbFilled v-if="themeStore.isDark" style="color: #f59e0b" />
                <BulbOutlined v-else style="color: #6b7280" />
              </template>
            </a-button>
          </Tooltip>
        </div>

        <div
          v-if="!collapsed"
          :style="{ position: 'absolute', bottom: '16px', left: '16px', right: '16px', fontSize: '12px', color: versionColor }"
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
