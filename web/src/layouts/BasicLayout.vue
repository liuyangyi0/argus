<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import DegradationBar from '../components/DegradationBar.vue'
import ErrorBoundary from '../components/ErrorBoundary.vue'
import { useWebSocket } from '../composables/useWebSocket'

const router = useRouter()
const route = useRoute()

// Active state matching for sidebar links
const isActive = (pathPrefix: string) => route.path.startsWith(pathPrefix)

// WebSocket connection state for global banner
const { connected: wsConnected, reconnecting: wsReconnecting, fallbackMode: wsFallbackMode, retryCount: wsRetryCount, nextRetryIn: wsNextRetryIn } = useWebSocket({
  topics: ['health'],
})

// Global keyboard shortcuts
const shortcutHelpVisible = ref(false)
const navKeys: Record<string, string> = { '1': '/overview', '2': '/cameras', '3': '/alerts', '4': '/reports', '5': '/models', '6': '/system' }

function handleKeyDown(e: KeyboardEvent) {
  // Ignore when typing in inputs
  const tag = (e.target as HTMLElement)?.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

  if (e.key === '?') { shortcutHelpVisible.value = !shortcutHelpVisible.value; return }
  if (e.key === 'Escape') { shortcutHelpVisible.value = false; return }
  if (navKeys[e.key]) { router.push(navKeys[e.key]); return }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeyDown)
})
onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyDown)
})
</script>

<template>
  <div class="app">
    <!-- LEFT SIDEBAR -->
    <aside class="sidebar glass">
      <div>
        <div class="brand">
          <div class="brand-mark">A</div>
          <div class="brand-text">Argus<small>MONITORING</small></div>
        </div>
        <div class="nav">
          <div class="nav-label">工作台</div>
          <router-link to="/overview" :class="{ active: isActive('/overview') }">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>值班台
          </router-link>
          <router-link to="/cameras" :class="{ active: isActive('/cameras') }">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="m22 8-6 4 6 4V8Z"/><rect x="2" y="6" width="14" height="12" rx="2"/></svg>摄像头
          </router-link>
          <router-link to="/alerts" :class="{ active: isActive('/alerts') }">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9M10.3 21a1.94 1.94 0 0 0 3.4 0"/></svg>告警中心
          </router-link>
          <router-link to="/reports" :class="{ active: isActive('/reports') }">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M3 3v18h18"/><path d="M7 16l4-8 4 4 4-6"/></svg>报表
          </router-link>
          <div class="nav-label">系统</div>
          <router-link to="/models" :class="{ active: isActive('/models') }">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/></svg>模型管理
          </router-link>
          <router-link to="/system" :class="{ active: isActive('/system') }">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1A1.7 1.7 0 0 0 4.6 9a1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z"/></svg>设置
          </router-link>
        </div>
      </div>
      <div class="side-foot">
        <span class="mono">v0.2.0</span>
        <div class="pulse"><span class="pulse-dot"></span>Online</div>
      </div>
    </aside>

    <!-- CONTENT WRAPPER -->
    <div style="display: flex; flex-direction: column; flex: 1; min-width: 0;">
      <!-- WebSocket disconnect banner (inline in column) -->
      <div
        v-if="!wsConnected && (wsReconnecting || wsFallbackMode)"
        role="alert"
        aria-live="polite"
        :style="{
          background: wsFallbackMode ? 'rgba(217,119,6,.08)' : 'rgba(229,72,77,.06)',
          borderBottom: '0.5px solid ' + (wsFallbackMode ? 'rgba(217,119,6,.22)' : 'rgba(229,72,77,.18)'),
          color: wsFallbackMode ? '#d97706' : '#b01e22',
          textAlign: 'center',
          padding: '8px 16px',
          fontSize: '12px',
          fontWeight: 600,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          gap: '8px'
        }"
      >
        <span class="d" :style="{
          width: '6px', height: '6px', borderRadius: '50%',
          background: wsFallbackMode ? '#d97706' : '#e5484d',
          boxShadow: '0 0 0 3px ' + (wsFallbackMode ? 'rgba(217,119,6,.18)' : 'rgba(229,72,77,.18)')
        }"></span>
        <template v-if="wsFallbackMode">WebSocket 连接断开，已切换至受限模式</template>
        <template v-else>正在重新连接实况数据网流... ({{ wsRetryCount }}/3)<span v-if="wsNextRetryIn > 0" style="margin-left: 8px; opacity: 0.8; font-weight: 500;">{{ wsNextRetryIn }}s 后重试</span></template>
      </div>

      <ErrorBoundary>
        <DegradationBar />
        <!-- Flex row container for route views -->
        <div style="display: flex; flex-direction: row; flex: 1; min-height: 0; gap: 12px;">
          <router-view v-slot="{ Component }">
            <keep-alive :include="['OverviewPage', 'CamerasPage', 'AlertsPage']">
              <component :is="Component" />
            </keep-alive>
          </router-view>
        </div>
      </ErrorBoundary>
    </div>

    <!-- Keyboard shortcuts window -->
    <div v-if="shortcutHelpVisible" class="shortcut-help glass">
      <h4>键盘快捷键</h4>
      <ul>
        <li><kbd>?</kbd><span>显示/隐藏</span></li>
        <li><kbd>1</kbd><span>值班台</span></li>
        <li><kbd>2</kbd><span>摄像头</span></li>
        <li><kbd>3</kbd><span>告警</span></li>
        <li><kbd>4</kbd><span>模型管理</span></li>
        <li><kbd>5</kbd><span>系统</span></li>
        <li><kbd>Esc</kbd><span>关闭</span></li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
/* ============ LEFT SIDEBAR ============ */
.sidebar {
  width: 230px;
  flex-shrink: 0;
  padding: 20px 14px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.brand { display: flex; align-items: center; gap: 11px; padding: 4px 10px 22px; }
.brand-mark {
  width: 30px; height: 30px; border-radius: 8px;
  background: linear-gradient(140deg,#1f2937 0%,#0a0a0c 100%);
  display: grid; place-items: center; color: #fff; font-weight: 800; font-size: 13px;
  letter-spacing: -0.02em;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.18), inset 0 -1px 2px rgba(0,0,0,.4), 0 3px 8px rgba(10,10,15,.18);
}
.brand-text { font-size: 15px; font-weight: 700; letter-spacing: -0.022em; color: var(--ink); line-height: 1.15; }
.brand-text small { display: block; font-size: 9.5px; font-weight: 600; color: var(--ink-5); letter-spacing: .1em; margin-top: 2px; }

.nav { display: flex; flex-direction: column; gap: 1px; padding: 0 2px; }
.nav-label { font-size: 10.5px; font-weight: 600; color: var(--ink-5); letter-spacing: .08em; text-transform: uppercase; padding: 14px 12px 6px; }
.nav-label:first-child { padding-top: 4px; }
.nav a {
  display: flex; align-items: center; gap: 11px; padding: 8px 12px; border-radius: var(--r-sm);
  color: var(--ink-3); text-decoration: none; font-size: 13px; font-weight: 500;
  transition: background .15s, color .15s; position: relative;
}
.nav a svg { width: 16px; height: 16px; flex-shrink: 0; stroke-width: 1.8; color: var(--ink-4); transition: color .15s; }
.nav a:hover { background: rgba(10,10,15,.04); color: var(--ink); }
.nav a:hover svg { color: var(--ink-2); }
.nav a.active {
  color: var(--ink); font-weight: 600;
  background: #fff;
  box-shadow: 0 1px 2px rgba(10,10,15,.05), 0 4px 10px -2px rgba(10,10,15,.06), inset 0 0 0 0.5px rgba(10,10,15,.05);
}
.nav a.active svg { color: var(--ink); }

.side-foot { padding: 12px 12px 4px; border-top: 0.5px solid var(--line); display: flex; align-items: center; justify-content: space-between; font-size: 11px; color: var(--ink-5); font-weight: 500; }

/* Keyboard Help Window */
.shortcut-help {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 250px;
  padding: 20px;
  z-index: 1000;
}
.shortcut-help h4 { margin: 0 0 16px; font-size: 14px; }
.shortcut-help ul { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 8px; }
.shortcut-help li { display: flex; align-items: center; gap: 12px; font-size: 12px; color: var(--ink-3); }
.shortcut-help kbd { background: #fff; border: 1px solid var(--line-2); padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 11px; }
</style>
