<script setup lang="ts">
import { ref, computed, h, onMounted, onUnmounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { Button, Drawer, notification } from 'ant-design-vue'
import {
  AlertOutlined,
  BarChartOutlined,
  BellOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  AppstoreOutlined,
  RightOutlined,
  SettingOutlined,
  VideoCameraOutlined,
} from '@ant-design/icons-vue'
import DegradationBar from '../components/DegradationBar.vue'
import ErrorBoundary from '../components/ErrorBoundary.vue'
import ErrorCenterDrawer from '../components/system/ErrorCenterDrawer.vue'
import { useWebSocket } from '../composables/useWebSocket'
import { useSystemMode } from '../composables/useSystemMode'
import { ROLE_META } from '../constants/roles'
import { useAuthStore } from '../stores/useAuthStore'
import { useErrorStore } from '../stores/useErrorStore'

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()

const isActive = (pathPrefix: string) => route.path.startsWith(pathPrefix)
const isExact = (path: string) => route.path === path || route.path.startsWith(path + '/')

// Sub-menu definitions (mirrors router children for Models / System).
// `roles` mirrors the route-level `requiresRole` meta — entries with no
// `roles` key are visible to every authenticated role (and anonymous, since
// these are also the public/all-role pages). Backend RBAC remains the source
// of truth; this list only hides nav entries to reduce wandering.
type NavChild = { path: string; label: string; roles?: string[] }

interface SystemErrorWsPayload {
  severity?: 'info' | 'warning' | 'error' | 'critical'
  source?: string
  code?: string | number
  message?: string
  context?: Record<string, unknown>
  timestamp?: string
}

const modelsChildren: NavChild[] = [
  { path: '/models/baseline', label: '基线管理', roles: ['admin', 'engineer'] },
  { path: '/models/collections', label: '采集集合' },
  { path: '/models/training', label: '训练与评估', roles: ['admin', 'engineer'] },
  { path: '/models/registry', label: '模型与发布', roles: ['admin', 'engineer'] },
  { path: '/models/comparison', label: 'A/B 对比' },
  { path: '/models/labeling', label: '标注队列' },
  { path: '/models/threshold', label: '阈值预览' },
]

const systemChildren: NavChild[] = [
  { path: '/system/overview', label: '系统概览' },
  { path: '/system/model-status', label: '模型状态' },
  { path: '/system/config', label: '配置管理', roles: ['admin', 'engineer'] },
  { path: '/system/audit', label: '审计日志', roles: ['admin'] },
  { path: '/system/degradation', label: '降级事件' },
  { path: '/system/modules', label: '功能模块' },
  { path: '/system/classifier', label: '分类器' },
  { path: '/system/segmenter', label: '分割器' },
  { path: '/system/imaging', label: '多模态成像' },
  { path: '/system/cross-camera', label: '跨相机' },
  { path: '/system/users', label: '用户管理', roles: ['admin'] },
  { path: '/system/regions', label: '区域管理' },
]

// Visibility helpers: when the user isn't loaded yet we keep entries hidden
// so the menu never flashes options the user can't reach. Entries with no
// `roles` are always visible (matches public router meta).
function canSeeChild(child: NavChild): boolean {
  if (!child.roles) return true
  return auth.hasRole(child.roles)
}
const visibleModelsChildren = computed(() => modelsChildren.filter(canSeeChild))
const visibleSystemChildren = computed(() => systemChildren.filter(canSeeChild))

// Models top-level entry mirrors the visible child set for the current role.
const canSeeModelsGroup = computed(() => visibleModelsChildren.value.length > 0)
const canSeeSystemGroup = computed(() => visibleSystemChildren.value.length > 0)

// User dropdown derived state
const avatarLetter = computed(() => {
  const u = auth.currentUser
  if (!u) return '?'
  const src = u.display_name || u.username || ''
  return (src.trim().charAt(0) || '?').toUpperCase()
})

const roleColor = computed(() => {
  const role = auth.currentUser?.role
  return (role && ROLE_META[role]?.color) || 'default'
})
const roleLabel = computed(() => {
  const role = auth.currentUser?.role
  return (role && ROLE_META[role]?.label) || (role ?? '')
})

async function onMenuClick({ key }: { key: string }) {
  if (key !== 'logout') return
  // Best-effort POST to backend; even on failure (network down, cookie
  // already expired) we force-redirect to /login so the UI never sits in a
  // half-authenticated state where currentUser is cleared client-side but
  // the URL still shows a protected page.
  try {
    await fetch('/logout', { method: 'POST', credentials: 'include' })
  } catch {
    /* swallow — full-page redirect below is the source of truth */
  }
  auth.clear()
  window.location.href = '/login'
}

const modelsOpen = ref(isActive('/models'))
const systemOpen = ref(isActive('/system'))
const viewportWidth = ref(typeof window === 'undefined' ? 1280 : window.innerWidth)
const compactSidebar = computed(() => viewportWidth.value >= 768 && viewportWidth.value < 1280)
const compactNavigation = ref<'models' | 'system' | null>(null)
const compactNavigationDialog = ref<HTMLElement | null>(null)
let compactNavigationTrigger: HTMLElement | null = null
const compactNavigationTitle = computed(() => compactNavigation.value === 'models' ? '模型管理导航' : '系统设置导航')
const compactNavigationItems = computed(() => compactNavigation.value === 'models'
  ? visibleModelsChildren.value : visibleSystemChildren.value)

watch(compactSidebar, () => { compactNavigation.value = null })

function closeCompactNavigation() {
  compactNavigation.value = null
}

function onCompactNavigationVisibilityChange(open: boolean) {
  if (open && compactNavigation.value) {
    const dialog = compactNavigationDialog.value
    // The animation may finish after the user has already moved focus inside
    // the drawer. Initial focus must not overwrite that keyboard interaction.
    if (dialog && !dialog.contains(document.activeElement)) {
      dialog.querySelector<HTMLElement>('a')?.focus()
    }
  } else if (!open && !compactNavigation.value && compactSidebar.value && compactNavigationTrigger?.isConnected) {
    compactNavigationTrigger.focus()
  }
}

function handleCompactNavigationKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    event.preventDefault()
    closeCompactNavigation()
    return
  }
  if (event.key !== 'Tab') return
  const items = compactNavigationDialog.value?.querySelectorAll<HTMLElement>('a[href], button')
  if (!items?.length) return
  const first = items[0]!
  const last = items[items.length - 1]!
  const focused = document.activeElement
  if (!compactNavigationDialog.value?.contains(focused)
      || (event.shiftKey && focused === first) || (!event.shiftKey && focused === last)) {
    event.preventDefault()
    const target = event.shiftKey ? last : first
    target.focus()
  }
}

// Auto-open the parent group whose route is active
watch(
  () => route.path,
  (p) => {
    compactNavigation.value = null
    if (p.startsWith('/models')) modelsOpen.value = true
    if (p.startsWith('/system')) systemOpen.value = true
  },
)

function toggleModels(event: MouseEvent) {
  if (compactSidebar.value) {
    compactNavigationTrigger = event.currentTarget as HTMLElement
    compactNavigation.value = 'models'
    return
  }
  modelsOpen.value = !modelsOpen.value
}
function toggleSystem(event: MouseEvent) {
  if (compactSidebar.value) {
    compactNavigationTrigger = event.currentTarget as HTMLElement
    compactNavigation.value = 'system'
    return
  }
  systemOpen.value = !systemOpen.value
}

// Global error center: aggregates `system_errors` topic into a Pinia store and
// pops a notification for error/critical events. The same useWebSocket call
// also drives the disconnect banner via the `health` topic so we keep a single
// long-lived subscriber on the layout.
const errorStore = useErrorStore()
const { connected: wsConnected, reconnecting: wsReconnecting, fallbackMode: wsFallbackMode, retryCount: wsRetryCount, nextRetryIn: wsNextRetryIn } = useWebSocket<SystemErrorWsPayload>({
  topics: ['health', 'system_errors'],
  onMessage: (topic, payload) => {
    if (topic !== 'system_errors') return
    errorStore.pushError(payload)
    const severity = payload?.severity
    if (severity === 'critical' || severity === 'error') {
      notification.open({
        message: `[${payload.source ?? 'unknown'}] ${payload.code ?? 'unknown'}`,
        description: payload.message ?? '',
        type: severity === 'critical' ? 'error' : 'warning',
        duration: severity === 'critical' ? 0 : 6,  // critical 不自动关闭
        btn: () =>
          h(
            Button,
            {
              size: 'small',
              type: 'link',
              onClick: () => errorStore.openDrawer(),
            },
            { default: () => '查看错误中心' },
          ),
      })
    }
  },
})

const realtimeStatus = computed(() => {
  if (wsConnected.value) return '实时已连接'
  if (wsFallbackMode.value) return '实时数据受限'
  if (wsReconnecting.value) return '实时重连中'
  return '实时未连接'
})

// 痛点 4: global pipeline mode banner (capturing / training / maintenance)
const { showBanner: showModeBanner, bannerLabel: modeBannerLabel, bannerColor: modeBannerColor } = useSystemMode()

// Global keyboard shortcuts
const shortcutHelpVisible = ref(false)
const navKeys: Record<string, string> = {
  '1': '/overview',
  '2': '/cameras',
  '3': '/alerts',
  '4': '/reports',
  '6': '/models',
  '7': '/system',
}

function handleKeyDown(e: KeyboardEvent) {
  // Keep focus inside the modal navigation and prevent global page shortcuts
  // from navigating behind it. Ant Design's drawer does not trap focus.
  if (compactNavigation.value) {
    handleCompactNavigationKeydown(e)
    return
  }
  const tag = (e.target as HTMLElement)?.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

  if (e.key === '?') {
    shortcutHelpVisible.value = !shortcutHelpVisible.value
    return
  }

  if (e.key === 'Escape') {
    shortcutHelpVisible.value = false
    return
  }

  if (navKeys[e.key]) {
    router.push(navKeys[e.key])
  }
}

function handleResize() {
  viewportWidth.value = window.innerWidth
}

onMounted(() => {
  window.addEventListener('keydown', handleKeyDown)
  window.addEventListener('resize', handleResize, { passive: true })
  // Hydrate the dropdown when landing on a public page (e.g. /overview).
  // The router guard skips /api/me for public routes, so currentUser may be
  // null even though a session cookie is set. fetchCurrentUser() is cached
  // (60s TTL) and de-dupes in-flight calls, so this is cheap and idempotent.
  if (!auth.currentUser) {
    void auth.fetchCurrentUser()
  }
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyDown)
  window.removeEventListener('resize', handleResize)
})
</script>

<template>
  <div class="app">
    <aside class="sidebar glass">
      <div>
        <div class="brand">
          <div class="brand-mark">A</div>
          <div class="brand-text">Argus<small>MONITORING</small></div>
        </div>

        <div class="nav">
          <div class="nav-label">工作台</div>

          <router-link to="/overview" aria-label="总览" :class="{ active: isActive('/overview') }">
            <DashboardOutlined /><span class="nav-text">总览</span>
          </router-link>

          <router-link to="/cameras" aria-label="摄像头" :class="{ active: isActive('/cameras') }">
            <VideoCameraOutlined /><span class="nav-text">摄像头</span>
          </router-link>

          <router-link to="/analysis-tasks" aria-label="分析任务" :class="{ active: isActive('/analysis-tasks') }">
            <AppstoreOutlined /><span class="nav-text">分析任务</span>
          </router-link>

          <router-link to="/alerts" aria-label="告警中心" :class="{ active: isActive('/alerts') }">
            <AlertOutlined /><span class="nav-text">告警中心</span>
          </router-link>

          <router-link to="/reports" aria-label="报表" :class="{ active: isActive('/reports') }">
            <BarChartOutlined /><span class="nav-text">报表</span>
          </router-link>

          <div class="nav-label">系统</div>

          <!-- Models sub-menu (role-gated by router RBAC meta) -->
          <div v-if="canSeeModelsGroup" class="sub-group" :class="{ 'is-open': modelsOpen }">
            <button
              type="button"
              class="sub-toggle"
              :class="{ active: isActive('/models') }"
              :aria-expanded="compactSidebar ? compactNavigation === 'models' : modelsOpen"
              :aria-controls="compactSidebar ? 'compact-navigation' : 'models-navigation'"
              :aria-haspopup="compactSidebar ? 'dialog' : undefined"
              aria-label="模型管理"
              title="模型管理"
              @click="toggleModels"
            >
              <DatabaseOutlined />
              <span class="sub-label">模型管理</span>
              <RightOutlined class="caret" />
            </button>
            <div id="models-navigation" v-show="modelsOpen" class="sub-list">
              <router-link
                v-for="item in visibleModelsChildren"
                :key="item.path"
                :to="item.path"
                :class="{ active: isExact(item.path) }"
                class="sub-item"
              >
                <span class="sub-dot" />{{ item.label }}
              </router-link>
            </div>
          </div>

          <!-- System sub-menu (always visible — admin-only items inside are gated individually) -->
          <div v-if="canSeeSystemGroup" class="sub-group" :class="{ 'is-open': systemOpen }">
            <button
              type="button"
              class="sub-toggle"
              :class="{ active: isActive('/system') }"
              :aria-expanded="compactSidebar ? compactNavigation === 'system' : systemOpen"
              :aria-controls="compactSidebar ? 'compact-navigation' : 'system-navigation'"
              :aria-haspopup="compactSidebar ? 'dialog' : undefined"
              aria-label="系统设置"
              title="系统设置"
              @click="toggleSystem"
            >
              <SettingOutlined />
              <span class="sub-label">设置</span>
              <RightOutlined class="caret" />
            </button>
            <div id="system-navigation" v-show="systemOpen" class="sub-list">
              <router-link
                v-for="item in visibleSystemChildren"
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
        <span
          class="realtime-status"
          :class="{ 'is-connected': wsConnected }"
          role="status"
          aria-label="实时数据连接状态"
        >{{ realtimeStatus }}</span>
      </div>
    </aside>

    <!-- CONTENT WRAPPER -->
    <div class="content-shell">
      <!-- TOPBAR with error center bell + user dropdown -->
      <header v-if="auth.currentUser" class="topbar">
        <a-badge :count="errorStore.unreadCount" :overflow-count="99" :offset="[-4, 4]">
          <a-button type="text" shape="circle" title="错误中心" @click="errorStore.openDrawer()">
            <BellOutlined />
          </a-button>
        </a-badge>
        <a-dropdown placement="bottomRight" :trigger="['click']">
          <button type="button" class="user-trigger" aria-label="打开用户菜单">
            <a-avatar size="small">{{ avatarLetter }}</a-avatar>
            <span class="user-name">{{ auth.currentUser.display_name || auth.currentUser.username }}</span>
            <a-tag :color="roleColor" class="role-tag">{{ roleLabel }}</a-tag>
          </button>
          <template #overlay>
            <a-menu @click="onMenuClick">
              <a-menu-item key="profile" disabled>
                <span>{{ auth.currentUser.username }} · {{ roleLabel }}</span>
              </a-menu-item>
              <a-menu-divider />
              <a-menu-item key="logout">登出</a-menu-item>
            </a-menu>
          </template>
        </a-dropdown>
      </header>

      <!-- WebSocket disconnect banner (inline in column) -->
      <div
        v-if="!wsConnected && (wsReconnecting || wsFallbackMode)"
        class="connection-banner"
        :class="{ 'connection-banner-fallback': wsFallbackMode }"
        role="alert"
        aria-live="polite"
      >
        <span class="connection-dot"></span>
        <template v-if="wsFallbackMode">WebSocket 连接断开，已切换到受限模式</template>
        <template v-else>
          正在重新连接实时数据流... ({{ wsRetryCount }}/3)
          <span v-if="wsNextRetryIn > 0" class="connection-countdown">
            {{ wsNextRetryIn }}s 后重试
          </span>
        </template>
      </div>

      <ErrorBoundary>
        <!-- 痛点 4: pipeline mode banner (visible only when not in normal state) -->
        <div
          v-if="showModeBanner"
          class="mode-banner"
          :style="{ backgroundColor: modeBannerColor }"
        >
          {{ modeBannerLabel }}
        </div>
        <DegradationBar />
        <!-- Flex row container for route views -->
        <div class="route-shell">
          <router-view v-slot="{ Component }">
            <keep-alive :include="['OverviewPage', 'CamerasPage', 'AlertsPage']">
              <component :is="Component" />
            </keep-alive>
          </router-view>
        </div>
      </ErrorBoundary>
    </div>


    <Drawer
      :open="compactNavigation !== null"
      placement="left"
      :width="300"
      :keyboard="true"
      :autofocus="false"
      :closable="false"
      @close="closeCompactNavigation"
      @after-open-change="onCompactNavigationVisibilityChange"
    >
      <section ref="compactNavigationDialog" role="dialog" aria-modal="true" :aria-label="compactNavigationTitle">
        <header class="compact-navigation-header">
          <h2>{{ compactNavigationTitle }}</h2>
          <Button aria-label="关闭导航" @click="closeCompactNavigation">关闭</Button>
        </header>
        <nav id="compact-navigation" class="compact-navigation" :aria-label="compactNavigationTitle">
          <router-link
            v-for="item in compactNavigationItems"
            :key="item.path"
            :to="item.path"
            :class="{ active: isExact(item.path) }"
            :aria-current="isExact(item.path) ? 'page' : undefined"
            @click="closeCompactNavigation"
          >{{ item.label }}</router-link>
        </nav>
      </section>
    </Drawer>

    <!-- Global error center drawer (mounted once at app shell level) -->
    <ErrorCenterDrawer />

    <!-- Keyboard shortcuts window -->
    <div v-if="shortcutHelpVisible" class="shortcut-help glass">
      <h4>键盘快捷键</h4>
      <ul>
        <li><kbd>?</kbd><span>显示或隐藏</span></li>
        <li><kbd>1</kbd><span>总览</span></li>
        <li><kbd>2</kbd><span>摄像头</span></li>
        <li><kbd>3</kbd><span>告警中心</span></li>
        <li><kbd>4</kbd><span>报表</span></li>
        <li><kbd>6</kbd><span>模型管理</span></li>
        <li><kbd>7</kbd><span>系统设置</span></li>
        <li><kbd>Esc</kbd><span>关闭</span></li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
/* ============ LEFT SIDEBAR ============ */
.content-shell {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-width: 0;
}
.route-shell {
  display: flex;
  flex-direction: row;
  flex: 1;
  min-height: 0;
  gap: 12px;
}
.sidebar {
  width: 230px;
  flex-shrink: 0;
  padding: 20px 14px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  overflow-y: auto;
}

.brand {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 4px 10px 22px;
}

.brand-mark {
  width: 30px;
  height: 30px;
  border-radius: 8px;
  background: linear-gradient(140deg, #1f2937 0%, #0a0a0c 100%);
  display: grid;
  place-items: center;
  color: #fff;
  font-weight: 800;
  font-size: 13px;
  letter-spacing: -0.02em;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.18), inset 0 -1px 2px rgba(0, 0, 0, 0.4), 0 3px 8px rgba(10, 10, 15, 0.18);
}

.brand-text {
  font-size: 15px;
  font-weight: 700;
  letter-spacing: -0.022em;
  color: var(--ink);
  line-height: 1.15;
}

.brand-text small {
  display: block;
  font-size: 9.5px;
  font-weight: 600;
  color: var(--ink-5);
  letter-spacing: 0.1em;
  margin-top: 2px;
}

.nav {
  display: flex;
  flex-direction: column;
  gap: 1px;
  padding: 0 2px;
}

.nav-label {
  font-size: 10.5px;
  font-weight: 600;
  color: var(--ink-5);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 14px 12px 6px;
}

.nav-label:first-child {
  padding-top: 4px;
}

.nav a {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 8px 12px;
  border-radius: var(--r-sm);
  color: var(--ink-3);
  text-decoration: none;
  font-size: 13px;
  font-weight: 500;
  transition: background 0.15s, color 0.15s;
  position: relative;
}

.nav a svg {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
  stroke-width: 1.8;
  color: var(--ink-4);
  transition: color 0.15s;
}

.nav a:hover {
  background: rgba(10, 10, 15, 0.04);
  color: var(--ink);
}

.nav a:hover svg {
  color: var(--ink-2);
}

.nav a.active {
  color: var(--ink);
  font-weight: 600;
  background: #fff;
  box-shadow: 0 1px 2px rgba(10, 10, 15, 0.05), 0 4px 10px -2px rgba(10, 10, 15, 0.06), inset 0 0 0 0.5px rgba(10, 10, 15, 0.05);
}


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
.realtime-status { color: var(--text-secondary); }
.realtime-status::before {
  content: '';
  display: inline-block;
  width: 6px;
  height: 6px;
  margin-right: 5px;
  border-radius: 50%;
  background: var(--status-warning);
}
.realtime-status.is-connected::before { background: var(--status-success); }

.compact-navigation { display: flex; flex-direction: column; gap: 6px; }
.compact-navigation-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.compact-navigation-header h2 { font-size: 16px; color: var(--text-primary); }
.compact-navigation a {
  padding: 10px 12px;
  border-radius: var(--r-sm);
  color: var(--ink-3);
  text-decoration: none;
}
.compact-navigation a:hover,
.compact-navigation a.active { color: var(--ink); background: var(--surface-canvas); }
.compact-navigation a:focus-visible,
.nav a:focus-visible,
.sub-toggle:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }


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

/* 痛点 4: pipeline mode banner */
.mode-banner {
  color: #fff;
  padding: 6px 16px;
  font-size: 13px;
  font-weight: 600;
  text-align: center;
  letter-spacing: 0.3px;
}

/* ============ TOPBAR with user dropdown ============ */
.topbar {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 8px;
  height: 44px;
  padding: 0 18px;
  border-bottom: 0.5px solid var(--line);
  background: rgba(255, 255, 255, 0.6);
  backdrop-filter: blur(8px);
  flex-shrink: 0;
}
.user-trigger {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 4px 10px;
  border-radius: var(--r-sm);
  cursor: pointer;
  transition: background 0.15s;
  user-select: none;
  border: 0;
  background: transparent;
  font: inherit;
}
.user-trigger:hover { background: rgba(10, 10, 15, 0.05); }
.user-trigger:focus-visible {
  outline: 2px solid rgba(10, 10, 15, 0.25);
  outline-offset: 2px;
}
.user-trigger .user-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--ink);
  max-width: 160px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.user-trigger .role-tag {
  margin-inline-end: 0;
  font-size: 11px;
  line-height: 18px;
  padding: 0 6px;
}

.connection-banner {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-bottom: 0.5px solid rgba(229, 72, 77, 0.18);
  background: rgba(229, 72, 77, 0.06);
  color: #b01e22;
  text-align: center;
  font-size: 12px;
  font-weight: 600;
}
.connection-banner-fallback {
  border-bottom-color: rgba(217, 119, 6, 0.22);
  background: rgba(217, 119, 6, 0.08);
  color: var(--status-warning);
}
.connection-dot {
  width: 6px;
  height: 6px;
  flex: 0 0 6px;
  border-radius: 50%;
  background: var(--status-danger);
  box-shadow: 0 0 0 3px rgba(229, 72, 77, 0.18);
}
.connection-banner-fallback .connection-dot {
  background: var(--status-warning);
  box-shadow: 0 0 0 3px rgba(217, 119, 6, 0.18);
}
.connection-countdown {
  margin-left: 8px;
  opacity: 0.8;
  font-weight: 500;
}

@media (min-width: 768px) and (max-width: 1279px) {
  .sidebar {
    width: 76px;
    padding: 16px 8px;
  }
  .brand {
    justify-content: center;
    padding: 4px 0 18px;
  }
  .brand-text,
  .nav-label,
  .nav-text,
  .sub-label,
  .sub-toggle .caret,
  .sub-list,
  .side-foot {
    display: none;
  }
  .nav a,
  .sub-toggle {
    justify-content: center;
    padding: 11px;
  }
  .nav a svg,
  .sub-toggle svg {
    width: 19px;
    height: 19px;
  }
}

@media (max-width: 767px) {
  .app {
    padding: 8px;
    gap: 8px;
  }
  .content-shell {
    padding-bottom: 66px;
  }
  .sidebar {
    position: fixed;
    z-index: 100;
    right: 8px;
    bottom: 8px;
    left: 8px;
    width: auto;
    height: 58px;
    padding: 6px;
    overflow: hidden;
  }
  .sidebar > div:first-child,
  .nav {
    width: 100%;
    height: 100%;
  }
  .brand,
  .nav-label,
  .sub-group,
  .side-foot {
    display: none;
  }
  .nav {
    flex-direction: row;
    justify-content: space-around;
    gap: 2px;
  }
  .nav a {
    flex: 1;
    flex-direction: column;
    justify-content: center;
    gap: 2px;
    padding: 4px 2px;
    font-size: 10px;
  }
  .nav a svg {
    width: 18px;
    height: 18px;
  }
  .topbar {
    height: 40px;
    padding: 0 8px;
  }
  .user-trigger .user-name,
  .user-trigger .role-tag {
    display: none;
  }
  .connection-banner {
    padding: 7px 10px;
    font-size: 11px;
  }
  .route-shell {
    overflow: auto;
  }
}
</style>
