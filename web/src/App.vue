<script setup lang="ts">
import { h, ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { Layout, Menu, Typography, Tooltip, Modal } from 'ant-design-vue'
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
import ErrorBoundary from './components/ErrorBoundary.vue'
import { useThemeStore } from './stores/theme'
import { useWebSocket } from './composables/useWebSocket'

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

// WebSocket connection state for global banner
const { connected: wsConnected, reconnecting: wsReconnecting, fallbackMode: wsFallbackMode, retryCount: wsRetryCount, nextRetryIn: wsNextRetryIn } = useWebSocket({
  topics: ['health'],
})

const siderBg = computed(() => themeStore.isDark ? '#0f0f1a' : '#ffffff')
const siderBorder = computed(() => themeStore.isDark ? '#1f2937' : '#e5e7eb')
const versionColor = computed(() => themeStore.isDark ? '#4a5568' : '#9ca3af')

// Mobile: auto-collapse sidebar on small screens
const isMobile = ref(false)
function checkMobile() {
  isMobile.value = window.innerWidth < 768
  if (isMobile.value) collapsed.value = true
}
let _resizeTimer: ReturnType<typeof setTimeout> | null = null
function debouncedCheckMobile() {
  if (_resizeTimer) clearTimeout(_resizeTimer)
  _resizeTimer = setTimeout(checkMobile, 200)
}
// Global keyboard shortcuts
const shortcutHelpVisible = ref(false)
const navKeys: Record<string, string> = { '1': '/overview', '2': '/cameras', '3': '/alerts', '4': '/models', '5': '/system' }

function handleKeyDown(e: KeyboardEvent) {
  // Ignore when typing in inputs
  const tag = (e.target as HTMLElement)?.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

  if (e.key === '?') { shortcutHelpVisible.value = !shortcutHelpVisible.value; return }
  if (e.key === 'Escape') { shortcutHelpVisible.value = false; return }
  if (navKeys[e.key]) { router.push(navKeys[e.key]); return }
}

onMounted(() => {
  checkMobile()
  window.addEventListener('resize', debouncedCheckMobile)
  window.addEventListener('keydown', handleKeyDown)
})
onUnmounted(() => {
  window.removeEventListener('resize', debouncedCheckMobile)
  window.removeEventListener('keydown', handleKeyDown)
})
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
            :style="{ color: themeStore.tokens.colorPrimary, margin: 0, letterSpacing: '3px' }"
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
              :aria-label="themeStore.isDark ? '切换亮色主题' : '切换暗色主题'"
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
        <!-- WebSocket disconnect banner -->
        <div
          v-if="!wsConnected && (wsReconnecting || wsFallbackMode)"
          role="alert"
          aria-live="polite"
          :style="{
            background: wsFallbackMode ? '#faad14' : '#ff4d4f',
            color: '#fff',
            textAlign: 'center',
            padding: '6px 16px',
            fontSize: '13px',
            fontWeight: 500,
            zIndex: 1000,
          }"
        >
          <template v-if="wsFallbackMode">
            WebSocket 连接失败，已切换至轮询模式
          </template>
          <template v-else>
            正在重新连接... ({{ wsRetryCount }}/3)
            <span v-if="wsNextRetryIn > 0" style="margin-left: 8px; opacity: 0.8">
              {{ wsNextRetryIn }}s 后重试
            </span>
          </template>
        </div>
        <DegradationBar />
        <Layout.Content style="padding: 24px; overflow-y: auto">
          <ErrorBoundary>
            <router-view v-slot="{ Component }">
              <Transition name="page-fade" mode="out-in">
                <keep-alive :include="['OverviewPage', 'CamerasPage', 'AlertsPage']">
                  <component :is="Component" />
                </keep-alive>
              </Transition>
            </router-view>
          </ErrorBoundary>
        </Layout.Content>
      </Layout>
    </Layout>
    <!-- Keyboard shortcuts help -->
    <Modal v-model:open="shortcutHelpVisible" title="键盘快捷键" :footer="null" width="360px">
      <div style="display: grid; grid-template-columns: 60px 1fr; gap: 8px 16px; font-size: 13px">
        <kbd>?</kbd><span>显示快捷键帮助</span>
        <kbd>1</kbd><span>值班台</span>
        <kbd>2</kbd><span>摄像头</span>
        <kbd>3</kbd><span>告警</span>
        <kbd>4</kbd><span>模型管理</span>
        <kbd>5</kbd><span>系统</span>
        <kbd>Esc</kbd><span>关闭弹窗</span>
      </div>
    </Modal>
  </a-config-provider>
</template>

<style>
.page-fade-enter-active,
.page-fade-leave-active {
  transition: opacity 0.15s ease;
}
.page-fade-enter-from,
.page-fade-leave-to {
  opacity: 0;
}
</style>
