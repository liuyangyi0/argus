<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useWallStore } from '../stores/useWallStore'

defineOptions({ name: 'OverviewPage' })

const wallStore = useWallStore()
const { cameras, health, loading } = storeToRefs(wallStore)

// System status presentation
const systemStatus = computed(() => health.value?.status ?? 'unknown')

const uptimeFormatted = computed(() => {
  const s = health.value?.uptime_seconds ?? 0
  if (s < 3600) return `${Math.floor(s / 60)}m`
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`
})

const connectedCount = computed(() => health.value?.cameras?.filter((c: any) => c.connected).length ?? 0)
const totalCameras = computed(() => health.value?.cameras?.length ?? 0)

const todayAlerts = computed(() => cameras.value.reduce((sum: number, c: any) => sum + (c.alert_count_today || 0), 0))

const activeAlerts = computed(() => {
  const res = []
  for (const c of cameras.value) {
    if (c.active_alert) {
      res.push({ camera: c, alert: c.active_alert })
    }
  }
  return res
})

const peakAnomaly = computed(() => {
  if (!cameras.value.length) return 0
  const sorted = [...cameras.value].sort((a, b) => (b.current_score ?? 0) - (a.current_score ?? 0))
  return sorted[0].current_score ?? 0
})

const avgLatency = computed(() => {
  const cams = health.value?.cameras ?? []
  if (!cams.length) return 0
  return Math.round(cams.reduce((sum: number, c: any) => sum + (c.avg_latency_ms ?? 0), 0) / cams.length)
})

const clock = ref('--:--:--')
onMounted(() => {
  wallStore.fetchInitialStatus()
  setInterval(() => {
    clock.value = new Date().toTimeString().slice(0, 8)
  }, 1000)
})

const rightTab = ref<'alerts' | 'trend'>('alerts')

// Layout & Fullscreen
const layoutOptions = ['auto', '1', '2', '4', '6']
const currentLayoutIdx = ref(0)
const toggleLayout = () => {
  currentLayoutIdx.value = (currentLayoutIdx.value + 1) % layoutOptions.length
}
const gridClass = computed(() => {
  const mode = layoutOptions[currentLayoutIdx.value]
  if (mode === 'auto') {
    return 'grid-' + Math.min(Math.max(cameras.value.length, 1), 6)
  }
  return 'grid-' + mode
})

const isFullscreen = ref(false)
const toggleFullscreen = () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {})
  } else {
    document.exitFullscreen().catch(() => {})
  }
}
onMounted(() => {
  document.addEventListener('fullscreenchange', () => {
    isFullscreen.value = !!document.fullscreenElement
  })
})
</script>

<template>
  <main>
    <div class="topbar glass" style="padding-right: 14px">
        <div style="display:flex;align-items:center;gap:14px;flex:1;overflow:hidden;">
          <h1>值班台</h1>
          <span class="sep"></span>
          <span class="crumb" style="flex-shrink:0">实时监控</span>
          <span class="sep"></span>
          
          <div class="mini-kpi" style="padding:0;background:transparent;border:none;box-shadow:none;flex:1">
            <div class="mk-item">
              <span class="d" :class="{ 'd-green': systemStatus === 'healthy', 'd-amber': systemStatus === 'degraded', 'd-red': systemStatus === 'unhealthy' }"></span>
              <span class="mk-text" style="display:flex;gap:6px;align-items:baseline">
                <strong :style="systemStatus === 'unhealthy' ? 'color:var(--red)' : (systemStatus === 'degraded' ? 'color:var(--amber)' : '')">{{ systemStatus === 'healthy' ? '运行正常' : (systemStatus === 'degraded' ? '降级运行' : '系统异常') }}</strong>
                <span class="mk-meta">持续 {{ uptimeFormatted }}</span>
              </span>
            </div>
            <div class="mk-sep"></div>
            <div class="mk-item">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="m22 8-6 4 6 4V8Z"/><rect x="2" y="6" width="14" height="12" rx="2"/></svg>
              <span class="mk-text">摄像头: <strong :style="connectedCount < totalCameras ? 'color:var(--amber)' : ''">{{ connectedCount }}/{{ totalCameras || 1 }}</strong></span>
            </div>
            <div class="mk-sep"></div>
            <div class="mk-item">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>
              <span class="mk-text">告警: <strong>{{ todayAlerts }}</strong><span v-if="activeAlerts.length > 0" style="color:var(--amber);margin-left:2px">*</span></span>
            </div>
            <div class="mk-sep"></div>
            <div class="mk-item">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
              <span class="mk-text">峰值异常: <strong :style="peakAnomaly >= 0.70 ? 'color:var(--red)' : ''">{{ peakAnomaly.toFixed(2) }}</strong></span>
            </div>
            <div class="mk-sep"></div>
            <div class="mk-item">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
              <span class="mk-text">延迟: <strong>{{ avgLatency }}ms</strong></span>
            </div>
          </div>
        </div>

        <div class="top-right" style="margin-left:14px;flex-shrink:0;position:relative">
          <span class="clock mono">{{ clock }}</span>
          
          <button class="ibtn" @click="toggleLayout" title="切换画面布局">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
               <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>
            </svg>
            <span v-if="layoutOptions[currentLayoutIdx] !== 'auto'" class="layout-badge">{{ layoutOptions[currentLayoutIdx] }}</span>
          </button>

          <button class="ibtn" @click="toggleFullscreen" :title="isFullscreen ? '退出全屏' : '全屏显示'">
            <svg v-if="!isFullscreen" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/></svg>
            <svg v-else fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 9h4V5M9 9L4 4M19 9h-4V5M15 9l5-5M5 15h4v4M9 15l-5 5M19 15h-4v4M15 15l5 5"/></svg>
          </button>
        </div>
      </div>

      <div class="viewport" style="flex:1; overflow:hidden; padding:2px 4px 12px; display:flex; min-height:0;">

        <!-- VIDEOS -->
        <div class="videos" :class="gridClass">
          <template v-for="cam in cameras" :key="cam.camera_id">
            <div class="feed" :class="{ 'live': health?.cameras?.find((c: any) => c.camera_id === cam.camera_id)?.connected }">
              <div class="feed-bg"></div>
              <div v-if="cam.current_score && cam.current_score > 0.6" style="position: absolute; bottom: 0; left: 0; right: 0; height: 35%; z-index: 1; pointer-events: none; mix-blend-mode: multiply;">
                <svg width="100%" height="100%" viewBox="0 0 100 40" preserveAspectRatio="none">
                  <path d="M0 30 Q 15 15, 30 25 T 60 20 T 100 25 L 100 50 L 0 50 Z" fill="rgba(229,72,77,.18)"/>
                  <path d="M0 30 Q 15 15, 30 25 T 60 20 T 100 25" fill="none" stroke="#e5484d" stroke-width="1.8" vector-effect="non-scaling-stroke"/>
                </svg>
              </div>

              <div class="feed-ov">
                <div style="display:flex;justify-content:space-between;align-items:flex-start">
                  <div style="display:flex;gap:6px">
                    <span class="chip"><span class="d" style="color:#2563eb"></span>{{ cam.name || cam.camera_id }}</span>
                    <span class="chip mono" style="color:var(--ink-4)">{{ health?.cameras?.find((c: any) => c.camera_id === cam.camera_id)?.frames_captured ?? '--' }} FPS</span>
                  </div>
                  <div style="display:flex;gap:6px">
                    <span class="chip green" v-if="health?.cameras?.find((c: any) => c.camera_id === cam.camera_id)?.connected">REC</span>
                    <span class="chip red" v-else>OFFLINE</span>
                    <span class="chip amber">AI</span>
                  </div>
                </div>
                
                <div style="display:flex;justify-content:flex-end;align-items:flex-end">
                  <span class="chip" :style="cam.current_score >= 0.7 ? 'background:rgba(229,72,77,.95);color:#fff;border-color:rgba(255,255,255,.4)' : ''" v-if="cam.current_score">
                    {{ cam.current_score >= 0.7 ? '异常检出' : '监测得分' }} · {{ (cam.current_score * 100).toFixed(0) }}
                  </span>
                </div>
              </div>
            </div>
          </template>
          
          <template v-if="cameras.length === 0 && !loading">
            <div class="empty"><svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="m22 8-6 4 6 4V8Z"/><rect x="2" y="6" width="14" height="12" rx="2"/></svg><span>未配置视频源</span></div>
            <div class="empty"><svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="m22 8-6 4 6 4V8Z"/><rect x="2" y="6" width="14" height="12" rx="2"/></svg><span>未配置视频源</span></div>
          </template>
          
          <div class="empty" v-if="cameras.length % 2 !== 0 && cameras.length > 0">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="m22 8-6 4 6 4V8Z"/><rect x="2" y="6" width="14" height="12" rx="2"/></svg><span>未配置视频源</span>
          </div>
        </div>

      </div>
  </main>

  <aside class="right glass">
    <div style="padding: 0 6px 16px;">
      <div class="seg" style="display: flex; padding: 4px; background: rgba(10,10,15,.05); border-radius: 9px; gap: 2px;">
        <button style="flex:1; padding: 6px 0; display:flex; justify-content:center; align-items:center; gap: 6px;" :class="{ on: rightTab === 'alerts' }" @click="rightTab = 'alerts'">
          待处理告警 <span class="count" :style="rightTab === 'alerts' ? '' : 'background:transparent;border-color:transparent;color:var(--ink-5)'">{{ String(activeAlerts.length).padStart(2, '0') }}</span>
        </button>
        <button style="flex:1; padding: 6px 0;" :class="{ on: rightTab === 'trend' }" @click="rightTab = 'trend'">异常趋势</button>
      </div>
    </div>
      <div class="right-body" v-if="rightTab === 'alerts'">
        <template v-if="activeAlerts.length > 0">
          <div class="alert-card" v-for="item in activeAlerts" :key="item.alert.alert_id">
            <div class="a-row">
              <div>
                <div class="a-title">高置信度异常产生</div>
                <div class="a-time">{{ item.camera.name || item.camera.camera_id }} · {{ typeof item.alert.timestamp === 'string' ? new Date(item.alert.timestamp).toLocaleTimeString() : new Date(((item.alert.timestamp ?? item.alert.created_at) ?? 0) * 1000).toLocaleTimeString() }}</div>
              </div>
              <div class="a-marker"></div>
            </div>
            <div class="a-num">{{ ((item.alert.anomaly_score ?? 0) * 100).toFixed(0) }}<small>分</small></div>
            <button class="a-btn" @click="$router.push(`/alerts?camera=${item.camera.camera_id}`)">查看详情 →</button>
          </div>
        </template>
        
        <div class="empty-state" v-else>
          <div class="er"><svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg></div>
          <span>暂无其他待处理告警</span>
        </div>
      </div>

      <!-- TREND CONTENT inside Right Panel -->
      <div class="right-body" style="padding: 0" v-else-if="rightTab === 'trend'">
        <div class="trend" style="padding: 8px 6px; flex: 1; display: flex; flex-direction: column;">
          <div class="trend-head" style="margin-bottom: 24px; flex-direction: column; align-items: flex-start; gap: 12px;">
            <div class="left">
              <h3 style="font-size: 13.5px"><svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>异常分数全局趋势</h3>
              <span class="meta" style="font-size: 11px; margin-top:4px; display:block">实时更新 · 最近 2 分钟</span>
            </div>
            <div class="seg" style="align-self:stretch; background:rgba(10,10,15,.04); padding:3px; gap:2px"><button style="flex:1">5m</button><button class="on" style="flex:1">2m</button><button style="flex:1">30s</button></div>
          </div>
          <div class="chart" style="flex:1; min-height: 180px; margin-top: auto">
            <div class="grid-h"><div></div><div></div><div></div><div></div><div></div></div>
            <svg width="100%" height="100%" viewBox="0 0 1000 160" preserveAspectRatio="none">
              <defs><linearGradient id="tg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#0a0a0c" stop-opacity=".16"/><stop offset="100%" stop-color="#0a0a0c" stop-opacity="0"/></linearGradient></defs>
              <path d="M0 160 L0 110 C 40 110, 60 70, 100 70 C 140 70, 160 95, 200 95 C 240 95, 260 50, 300 50 C 340 50, 360 120, 400 120 C 440 120, 460 30, 500 30 C 540 30, 560 80, 600 80 C 640 80, 660 20, 700 20 C 740 20, 760 60, 800 60 C 840 60, 860 130, 900 130 C 940 130, 960 70, 1000 70 L 1000 180 L 0 180 Z" fill="url(#tg)"/>
              <path d="M0 110 C 40 110, 60 70, 100 70 C 140 70, 160 95, 200 95 C 240 95, 260 50, 300 50 C 340 50, 360 120, 400 120 C 440 120, 460 30, 500 30 C 540 30, 560 80, 600 80 C 640 80, 660 20, 700 20 C 740 20, 760 60, 800 60 C 840 60, 860 130, 900 130 C 940 130, 960 70, 1000 70" fill="none" stroke="#0a0a0c" stroke-width="2.5" vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round"/>
              <circle cx="1000" cy="70" r="4.5" fill="#fff" stroke="#0a0a0c" stroke-width="2"><animate attributeName="r" values="4;7;4" dur="2s" repeatCount="indefinite"/></circle>
            </svg>
          </div>
        </div>
      </div>
    </aside>
</template>

<style scoped>
/* Scoped styles to support Overview inner structure if needed. */
main{flex:1;display:flex;flex-direction:column;min-width:0;gap:12px}

.topbar{height:60px;flex-shrink:0;display:flex;align-items:center;justify-content:space-between;padding:0 22px}
.topbar h1{font-size:18px;font-weight:700;color:var(--ink);letter-spacing:-0.028em;margin:0;}
.crumb{font-size:12px;color:var(--ink-5);font-weight:500}
.crumb b{color:var(--ink-3);font-weight:600}
.sep{width:1px;height:14px;background:var(--line-2)}

.pill{
  display:inline-flex;align-items:center;gap:7px;padding:5px 11px;border-radius:999px;
  font-size:11.5px;font-weight:600;letter-spacing:-0.005em;
  background:rgba(229,72,77,.09);color:#b01e22;
  border:0.5px solid rgba(229,72,77,.22);
}
.pill .d{width:6px;height:6px;border-radius:50%;background:var(--red);box-shadow:0 0 0 3px rgba(229,72,77,.18);animation:breathe 2s infinite}

.top-right{display:flex;align-items:center;gap:8px}
.clock{
  font-size:12.5px;color:var(--ink-3);font-weight:600;
  padding:7px 13px;border-radius:var(--r-sm);
  background:rgba(255,255,255,.7);border:0.5px solid var(--line-2);
  backdrop-filter:blur(20px);
}
.ibtn{
  width:34px;height:34px;border-radius:var(--r-sm);display:grid;place-items:center;
  background:rgba(255,255,255,.7);border:0.5px solid var(--line-2);
  color:var(--ink-3);cursor:pointer;transition:all .18s;backdrop-filter:blur(20px);
}
.ibtn:hover{background:#fff;color:var(--ink);box-shadow:var(--sh-1)}
.ibtn svg{width:15px;height:15px;stroke-width:1.9}
.layout-badge {
  position: absolute; right: 38px; top: -4px; background: var(--blue); color: #fff;
  font-size: 9px; font-weight: 800; font-family: 'JetBrains Mono', monospace;
  padding: 1px 4px; border-radius: 4px; pointer-events: none;
  box-shadow: 0 2px 4px rgba(37,99,235,.3);
}

.viewport { padding-right:0; gap: 12px; }

/* ============ MINI KPI ============ */
.mini-kpi {
  display: flex; align-items: center; padding: 10px 18px; border-radius: var(--r);
  font-size: 12px; flex-shrink: 0; gap: 14px;
}
.mk-item { display: flex; align-items: center; gap: 7px; color: var(--ink-4); }
.mk-item svg { width: 14px; height: 14px; stroke-width: 2.2; color: var(--ink-5); }
.mk-text { font-weight: 500; }
.mk-text strong { font-weight: 600; color: var(--ink-2); font-family: 'JetBrains Mono', monospace; }
.mk-sep { width: 1px; height: 12px; background: var(--line-2); }
.mk-meta { font-size: 11px; color: var(--ink-5); font-weight: 500; }
.d { width: 6px; height: 6px; border-radius: 50%; }
.d-green { background: var(--green); box-shadow: 0 0 0 3px rgba(21,163,74,.18); animation: breathe 2s infinite; }
.d-amber { background: var(--amber); box-shadow: 0 0 0 3px rgba(217,119,6,.18); animation: breathe 2s infinite; }
.d-red { background: var(--red); box-shadow: 0 0 0 3px rgba(229,72,77,.18); animation: breathe 2s infinite; }

/* ============ SUPER RESPONSIVE CCTV GRID ============ */
.videos {
  flex: 1; display: grid; gap: 12px; min-height: 0; width: 100%;
}
.videos.grid-1 { grid-template-columns: 1fr; grid-template-rows: 1fr; }
.videos.grid-2 { grid-template-columns: repeat(2, 1fr); grid-template-rows: 1fr; }
.videos.grid-3, .videos.grid-4 { grid-template-columns: repeat(2, 1fr); grid-template-rows: repeat(2, 1fr); }
.videos.grid-5, .videos.grid-6 { grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(2, 1fr); }
@media(max-width: 980px) {
  .videos.grid-2, .videos.grid-3, .videos.grid-4, .videos.grid-5, .videos.grid-6 {
    grid-template-columns: 1fr; grid-template-rows: repeat(auto-fit, minmax(0, 1fr));
  }
}

.feed {
  border-radius: var(--r-lg); overflow: hidden; position: relative;
  box-shadow: var(--sh-2); border: 0.5px solid rgba(255,255,255,.65);
  height: 100%; width: 100%;
}
.feed.live{box-shadow:var(--sh-3),0 0 0 0.5px rgba(10,10,15,.08)}
.feed-bg{
  position:absolute;inset:0;
  background:
    radial-gradient(ellipse at 30% 30%,rgba(180,195,220,.55),transparent 65%),
    radial-gradient(ellipse at 70% 75%,rgba(195,180,210,.4),transparent 65%),
    linear-gradient(155deg,#e8ebf0,#dde0e8);
}
.feed-bg::after{content:"";position:absolute;inset:0;background:repeating-linear-gradient(0deg,rgba(255,255,255,.05) 0 1px,transparent 1px 4px)}
.feed-ov{position:absolute;inset:0;padding:14px;display:flex;flex-direction:column;justify-content:space-between;z-index:2}
.chip{
  display:inline-flex;align-items:center;gap:6px;
  padding:5px 10px;border-radius:8px;font-size:10.5px;font-weight:600;
  background:rgba(255,255,255,.92);backdrop-filter:blur(20px) saturate(180%);
  border:0.5px solid rgba(255,255,255,.95);
  box-shadow:0 2px 6px rgba(10,10,15,.08),inset 0 1px 0 rgba(255,255,255,.95);
  color:var(--ink-2);letter-spacing:.005em;
}
.chip.red{color:#b01e22;background:rgba(255,242,242,.95)}
.chip.green{color:#0c7a36;background:rgba(238,250,242,.95)}
.chip.amber{color:#9a5b04;background:rgba(255,247,232,.95)}
.chip .d{width:5px;height:5px;border-radius:50%;background:currentColor}

.empty{
  border-radius:var(--r-lg); height: 100%; width: 100%;
  background:rgba(255,255,255,.42);
  border:1px dashed rgba(10,10,15,.10);
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;
  color:var(--ink-5);transition:all .25s;backdrop-filter:blur(20px);
}
.empty:hover{background:rgba(255,255,255,.6);border-color:rgba(10,10,15,.18);color:var(--ink-4)}
.empty svg{width:24px;height:24px;stroke-width:1.7}
.empty span{font-size:12px;font-weight:500}

/* ============ TREND ============ */
.trend{padding:22px 24px}
.trend-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}
.trend-head .left{display:flex;align-items:center;gap:12px}
.trend-head h3{font-size:13.5px;font-weight:700;color:var(--ink);letter-spacing:-0.022em;display:flex;align-items:center;gap:8px;margin:0;}
.trend-head h3 svg{width:14px;height:14px;stroke-width:2.2;color:var(--ink-3)}
.trend-head .meta{font-size:11px;color:var(--ink-5);font-weight:500}
.seg{display:flex;background:rgba(10,10,15,.045);border-radius:8px;padding:3px;gap:1px}
.seg button{font-size:11px;font-weight:600;color:var(--ink-4);padding:5px 12px;border:none;background:transparent;border-radius:6px;cursor:pointer;transition:all .15s;letter-spacing:.005em}
.seg button:hover{color:var(--ink-2)}
.seg button.on{background:#fff;color:var(--ink);box-shadow:0 1px 2px rgba(10,10,15,.08),0 0 0 0.5px rgba(10,10,15,.04)}
.chart{height:160px;position:relative}
.grid-h{position:absolute;inset:0;display:flex;flex-direction:column;justify-content:space-between}
.grid-h div{height:0.5px;background:rgba(10,10,15,.05)}

/* ============ RIGHT BODY ============ */
.right{width:296px;flex-shrink:0;padding:20px 16px;display:flex;flex-direction:column}
.right-head{display:flex;justify-content:space-between;align-items:center;padding:0 6px 16px}
.right-head h2{font-size:13.5px;font-weight:700;color:var(--ink);letter-spacing:-0.022em;margin:0;}
.count{
  font-size:10.5px;font-weight:700;font-family:'JetBrains Mono',monospace;
  background:rgba(217,119,6,.10);color:var(--amber);
  padding:2px 8px;border-radius:999px;border:0.5px solid rgba(217,119,6,.22);
}
.right-body{flex:1;overflow-y:auto;padding:0 6px;display:flex;flex-direction:column;gap:10px}
.alert-card{
  padding:15px 16px;border-radius:var(--r);position:relative;cursor:pointer;
  background:linear-gradient(180deg,rgba(255,255,255,.95),rgba(255,255,255,.78));
  border:0.5px solid var(--line-2);
  box-shadow:var(--sh-1),0 6px 16px -6px rgba(10,10,15,.08),inset 0 1px 0 rgba(255,255,255,1);
  transition:transform .22s,box-shadow .22s;
}
.alert-card::before{
  content:"";position:absolute;left:0;top:14px;bottom:14px;width:2.5px;border-radius:0 3px 3px 0;
  background:var(--amber);
}
.alert-card:hover{transform:translateY(-2px);box-shadow:var(--sh-2)}
.alert-card .a-row{display:flex;justify-content:space-between;align-items:flex-start}
.a-title{font-size:12.5px;font-weight:600;color:var(--ink);letter-spacing:-0.012em}
.a-time{font-size:10.5px;color:var(--ink-5);font-family:'JetBrains Mono',monospace;margin-top:3px;letter-spacing:0}
.a-num{font-size:21px;font-weight:700;color:var(--ink);margin-top:10px;letter-spacing:-0.032em}
.a-num small{font-size:11px;color:var(--ink-5);font-weight:500;margin-left:4px}
.a-marker{width:7px;height:7px;border-radius:50%;background:var(--amber);box-shadow:0 0 0 3px rgba(217,119,6,.18);margin-top:6px}
.a-btn{
  margin-top:12px;width:100%;font-size:11px;font-weight:600;
  padding:7px 12px;border-radius:8px;border:0.5px solid var(--line-2);
  background:rgba(255,255,255,.85);color:var(--ink-3);cursor:pointer;transition:all .15s;
  letter-spacing:.005em;
}
.a-btn:hover{background:#fff;color:var(--ink);box-shadow:var(--sh-1)}

.empty-state{margin-top:32px;display:flex;flex-direction:column;align-items:center;gap:12px;color:var(--ink-5)}
.er{
  width:50px;height:50px;border-radius:50%;display:grid;place-items:center;
  background:linear-gradient(180deg,#fff,rgba(255,255,255,.7));
  border:0.5px solid var(--line-2);
  box-shadow:0 4px 12px rgba(10,10,15,.04),inset 0 1px 0 rgba(255,255,255,1);
}
.er svg{color:var(--green);stroke-width:2.2;width:20px;height:20px}
.empty-state span{font-size:11.5px;font-weight:500}
</style>
