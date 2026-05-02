<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import DegradationBar from '../components/DegradationBar.vue'
import ErrorBoundary from '../components/ErrorBoundary.vue'
import { useWebSocket } from '../composables/useWebSocket'

const router = useRouter()
const route = useRoute()

// Active state matching for sidebar links
const isActive = (pathPrefix: string) => route.path.startsWith(pathPrefix)
const isExact = (path: string) => route.path === path || route.path.startsWith(path + '/')

// Sub-menu definitions (mirrors router children for Models / System)
const modelsChildren: Array<{ path: string; label: string }> = [
  { path: '/models/baseline', label: '基线管理' },
  { path: '/models/training', label: '训练与评估' },
  { path: '/models/registry', label: '模型与发布' },
  { path: '/models/comparison', label: 'A/B 对比' },
  { path: '/models/labeling', label: '标注队列' },
  { path: '/models/threshold', label: '阈值预览' },
]

const systemChildren: Array<{ path: string; label: string }> = [
  { path: '/system/overview', label: '系统概览' },
  { path: '/system/model-status', label: '模型状态' },
  { path: '/system/config', label: '配置管理' },
  { path: '/system/audit', label: '审计日志' },
  { path: '/system/degradation', label: '降级事件' },
  { path: '/system/modules', label: '功能模块' },
  { path: '/system/classifier', label: '分类器' },
  { path: '/system/segmenter', label: '分割器' },
  { path: '/system/imaging', label: '多模态成像' },
  { path: '/system/cross-camera', label: '跨相机' },
  { path: '/system/users', label: '用户管理' },
  { path: '/system/regions', label: '区域管理' },
  { path: '/system/notification-templates', label: '通知内容配置' },
]

const modelsOpen = ref(isActive('/models'))
const systemOpen = ref(isActive('/system'))

// Auto-open the parent group whose route is active
watch(
  () => route.path,
  (p) => {
    if (p.startsWith('/models')) modelsOpen.value = true
    if (p.startsWith('/system')) systemOpen.value = true
  },
)

function toggleModels() {
  modelsOpen.value = !modelsOpen.value
}
function toggleSystem() {
  systemOpen.value = !systemOpen.value
}

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

          <!-- Models sub-menu -->
          <div class="sub-group" :class="{ 'is-open': modelsOpen }">
            <button
              type="button"
              class="sub-toggle"
              :class="{ active: isActive('/models') }"
              :aria-expanded="modelsOpen"
              @click="toggleModels"
            >
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/></svg>
              <span class="sub-label">模型管理</span>
              <svg class="caret" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg>
            </button>
            <div v-show="modelsOpen" class="sub-list">
              <router-link
                v-for="item in modelsChildren"
                :key="item.path"
                :to="item.path"
                :class="{ active: isExact(item.path) }"
                class="sub-item"
              >
                <span class="sub-dot" />{{ item.label }}
              </router-link>
            </div>
          </div>

          <!-- System sub-menu -->
          <div class="sub-group" :class="{ 'is-open': systemOpen }">
            <button
              type="button"
              class="sub-toggle"
              :class="{ active: isActive('/system') }"
              :aria-expanded="systemOpen"
              @click="toggleSystem"
            >
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1A1.7 1.7 0 0 0 4.6 9a1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z"/></svg>
              <span class="sub-label">设置</span>
              <svg class="caret" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg>
            </button>
            <div v-show="systemOpen" class="sub-list">
              <router-link
                v-for="item in systemChildren"
                :key="item.path"
                :to="item.path"
                :class="{ active: isExact(item.path) }"
                class="sub-item"
              >
                <span class="sub-dot" />{{ item.label }}
              </router-link>
            </div>
          </div>
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
  overflow-y: auto;
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

/* ====== Sub-menu group ====== */
.sub-group { display: flex; flex-direction: column; }
.sub-toggle {
  display: flex; align-items: center; gap: 11px; padding: 8px 12px;
  border: none; background: transparent;
  border-radius: var(--r-sm);
  color: var(--ink-3); font-size: 13px; font-weight: 500;
  cursor: pointer; text-align: left; width: 100%;
  transition: background .15s, color .15s;
  font-family: inherit;
}
.sub-toggle svg { width: 16px; height: 16px; flex-shrink: 0; stroke-width: 1.8; color: var(--ink-4); transition: color .15s, transform .2s; }
.sub-toggle .sub-label { flex: 1; }
.sub-toggle .caret { width: 12px; height: 12px; opacity: .6; transition: transform .2s; }
.sub-group.is-open .sub-toggle .caret { transform: rotate(90deg); }
.sub-toggle:hover { background: rgba(10,10,15,.04); color: var(--ink); }
.sub-toggle:hover svg { color: var(--ink-2); }
.sub-toggle.active {
  color: var(--ink); font-weight: 600;
}
.sub-toggle.active svg { color: var(--ink); }

.sub-list {
  display: flex; flex-direction: column; gap: 1px;
  margin: 2px 0 4px 12px;
  padding-left: 10px;
  border-left: 1px solid var(--line);
}
.sub-item {
  display: flex !important; align-items: center; gap: 8px;
  padding: 6px 10px !important; border-radius: var(--r-sm);
  color: var(--ink-4) !important; font-size: 12.5px !important; font-weight: 500 !important;
  text-decoration: none; transition: background .15s, color .15s;
}
.sub-item .sub-dot {
  width: 4px; height: 4px; border-radius: 50%;
  background: var(--ink-5); flex-shrink: 0;
  transition: background .15s, transform .15s;
}
.sub-item:hover { background: rgba(10,10,15,.04); color: var(--ink-2) !important; }
.sub-item:hover .sub-dot { background: var(--ink-3); }
.sub-item.active {
  color: var(--ink) !important; font-weight: 600 !important;
  background: #fff !important;
  box-shadow: 0 1px 2px rgba(10,10,15,.05), inset 0 0 0 0.5px rgba(10,10,15,.05) !important;
}
.sub-item.active .sub-dot { background: var(--ink); transform: scale(1.4); }

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
