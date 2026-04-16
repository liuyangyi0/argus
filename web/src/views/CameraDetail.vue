<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Steps, Button } from 'ant-design-vue'
import { ArrowLeftOutlined } from '@ant-design/icons-vue'
import { getCameraDetail } from '../api'
import ZoneEditor from '../components/ZoneEditor.vue'
import CalibrationWizard from '../components/calibration/CalibrationWizard.vue'
import { useGo2RTC } from '../composables/useGo2RTC'

const route = useRoute()
const router = useRouter()
const cameraId = route.params.id as string
const camera = ref<any>(null)
const loading = ref(true)
let pollTimer: ReturnType<typeof setInterval> | null = null

async function fetchCamera() {
  try {
    camera.value = await getCameraDetail(cameraId) || null
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchCamera()
  pollTimer = setInterval(fetchCamera, 5000)
  startStream()
})

onUnmounted(() => {
  if (pollTimer !== null) {
    clearInterval(pollTimer)
    pollTimer = null
  }
  stopStream()
})

// go2rtc WebRTC/MSE player
const _go2rtc = useGo2RTC(cameraId)
// @ts-expect-error TS6133 — used as template ref="videoRef"
const videoRef = _go2rtc.videoRef
// @ts-expect-error TS6133 — used as template ref="mjpegRef"
const mjpegRef = _go2rtc.mjpegRef
const streamStatus = _go2rtc.status
const startStream = _go2rtc.start
const stopStream = _go2rtc.stop
const mjpegUrl = computed(() => `/api/cameras/${cameraId}/stream`)
const snapshotUrl = computed(() => `/api/cameras/${cameraId}/snapshot`)
const zoneEditorData = ref<any[]>(camera.value?.zones || [])

async function saveZones() {
  try {
    const { updateZones } = await import('../api/zones')
    await updateZones(cameraId, zoneEditorData.value)
  } catch { /* handled by global interceptor */ }
}

const lifecycleStep = computed(() => {
  if (!camera.value) return 0
  const stages = camera.value.stages
  if (stages && Array.isArray(stages)) {
    const activeIdx = stages.findIndex((s: any) => s.status === 'active')
    if (activeIdx >= 0) return activeIdx
    // All completed
    if (stages.every((s: any) => s.status === 'completed')) return stages.length - 1
    return 0
  }
  // Fallback when stages not available
  const c = camera.value
  if (!c.connected) return 0
  if (!c.stats || c.stats.frames_captured === 0) return 0
  if (c.stats.frames_analyzed === 0) return 2
  return 4
})

function formatLabel(key: string) {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function formatValue(value: any): string {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (Array.isArray(value)) return value.length ? value.map(formatValue).join(', ') : '-'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function flattenEntries(value: any, prefix = ''): Array<{ key: string; label: string; value: string }> {
  if (value === null || value === undefined) return []

  if (typeof value !== 'object' || Array.isArray(value)) {
    return [{ key: prefix, label: formatLabel(prefix || 'value'), value: formatValue(value) }]
  }

  return Object.entries(value).flatMap(([key, nested]) => {
    const nextPrefix = prefix ? `${prefix}.${key}` : key
    if (nested && typeof nested === 'object' && !Array.isArray(nested)) {
      return flattenEntries(nested, nextPrefix)
    }
    return [{ key: nextPrefix, label: formatLabel(nextPrefix), value: formatValue(nested) }]
  })
}

const basicEntries = computed(() => flattenEntries({
  camera_id: camera.value?.camera_id,
  name: camera.value?.name,
  connected: camera.value?.connected,
  running: camera.value?.running,
}))

const runtimeEntries = computed(() => flattenEntries({
  stats: camera.value?.stats,
  runtime: camera.value?.runtime,
  runner: camera.value?.runner,
}))

const healthEntries = computed(() => flattenEntries(camera.value?.health))
const detectorEntries = computed(() => flattenEntries(camera.value?.detector))
const configEntries = computed(() => flattenEntries(camera.value?.config))
const leftTab = ref<'live' | 'info' | 'zones'>('live')
</script>

<template>
  <main>
    <div class="topbar glass" style="padding-right: 14px">
      <div style="display:flex;align-items:center;gap:14px;flex:1;overflow:hidden;">
        <Button type="text" size="small" @click="router.push('/cameras')" style="margin-right: 8px; color: var(--ink-3);"><ArrowLeftOutlined /></Button>
        <span class="d" :class="camera?.connected ? 'd-green' : 'd-red'"></span>
        <h1 style="margin:0; font-family: 'JetBrains Mono', monospace">{{ cameraId }}</h1>
        <span class="crumb">{{ camera?.name || '未知设备' }}</span>
        <span class="sep"></span>
        <div style="flex:1">
          <Steps :current="lifecycleStep" size="small" style="max-width: 500px; transform: scale(0.9); transform-origin: left center;">
            <Steps.Step title="采集" />
            <Steps.Step title="基线审查" />
            <Steps.Step title="训练" />
            <Steps.Step title="发布" />
            <Steps.Step title="推理" />
          </Steps>
        </div>
      </div>
    </div>

    <div class="viewport" style="flex:1; overflow:hidden; padding:2px 4px 12px; display:flex; gap:12px; min-height:0;">
      <!-- Main Visual View -->
      <section class="main-panel glass" style="flex:1; display:flex; flex-direction:column; min-width:0; border-radius: var(--r-lg);">
        <div class="panel-header">
          <div class="seg" style="display:flex; background:rgba(10,10,15,.05); border-radius:8px; padding:3px; gap:2px">
            <button :class="{ on: leftTab === 'live' }" @click="leftTab = 'live'">实时画面</button>
            <button :class="{ on: leftTab === 'info' }" @click="leftTab = 'info'">全量属性</button>
            <button :class="{ on: leftTab === 'zones' }" @click="leftTab = 'zones'">区域编辑</button>
          </div>
        </div>

        <div class="panel-body">
          <!-- LIVE TAB -->
          <div v-if="leftTab === 'live'" class="live-wrapper">
             <div v-if="camera?.connected" class="video-container">
                <video ref="videoRef" autoplay muted playsinline v-show="streamStatus === 'playing' || streamStatus === 'connecting'" class="player-vid" />
                <img ref="mjpegRef" v-if="streamStatus === 'fallback'" :src="mjpegUrl" class="player-vid" alt="实时" />
                <div v-if="streamStatus === 'connecting'" class="overlay-msg">连接中...</div>
             </div>
             <div v-else class="video-container offline-state">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="m22 8-6 4 6 4V8Z"/><rect x="2" y="6" width="14" height="12" rx="2"/></svg>
                <span>摄像头离线</span>
             </div>
          </div>

          <!-- INFO TAB -->
          <div v-else-if="leftTab === 'info'" class="info-wrapper">
             <div class="wide-grid">
               <div class="meta-section" v-if="runtimeEntries.length">
                  <div class="meta-section-hd">运行状态</div>
                  <div class="meta-section-bd">
                     <div class="meta-row" v-for="item in runtimeEntries" :key="item.key">
                       <span class="meta-k">{{ item.label }}</span>
                       <span class="meta-v">{{ item.value }}</span>
                     </div>
                  </div>
               </div>

               <div class="meta-section" v-if="healthEntries.length">
                  <div class="meta-section-hd">外设健康探测</div>
                  <div class="meta-section-bd">
                     <div class="meta-row" v-for="item in healthEntries" :key="item.key">
                       <span class="meta-k">{{ item.label }}</span>
                       <span class="meta-v">{{ item.value }}</span>
                     </div>
                  </div>
               </div>
               
               <div class="meta-section" v-if="detectorEntries.length">
                  <div class="meta-section-hd">计算机视觉参数 (Detector)</div>
                  <div class="meta-section-bd">
                     <div class="meta-row" v-for="item in detectorEntries" :key="item.key">
                       <span class="meta-k">{{ item.label }}</span>
                       <span class="meta-v mono">{{ item.value }}</span>
                     </div>
                  </div>
               </div>

               <div class="meta-section" v-if="configEntries.length">
                  <div class="meta-section-hd">原始基线配置 (Config)</div>
                  <div class="meta-section-bd">
                     <div class="meta-row" v-for="item in configEntries" :key="item.key">
                       <span class="meta-k">{{ item.label }}</span>
                       <span class="meta-v mono">{{ item.value }}</span>
                     </div>
                  </div>
               </div>
             </div>
          </div>

          <!-- ZONES TAB -->
          <div v-else-if="leftTab === 'zones'" class="zones-wrapper">
            <ZoneEditor v-model="zoneEditorData" :image-src="snapshotUrl" :width="640" :height="480" />
            <div style="margin-top: 16px; text-align:right; flex-shrink: 0;">
               <Button type="primary" @click="saveZones">保存区域配置</Button>
            </div>
          </div>
        </div>
      </section>

      <!-- Calibration section (below main content area) -->
      <CalibrationWizard :camera-id="cameraId" style="margin-top: 12px" />

      <!-- Right Metadata Sidebar -->
      <aside class="right glass">
        <!-- Stats Grid at top -->
        <div v-if="camera?.stats" class="stats-grid">
          <div class="stat-card">
             <div class="stat-title">已采集帧</div>
             <div class="stat-val mono">{{ camera.stats.frames_captured }}</div>
          </div>
          <div class="stat-card">
             <div class="stat-title">已分析帧</div>
             <div class="stat-val mono">{{ camera.stats.frames_analyzed }}</div>
          </div>
          <div class="stat-card">
             <div class="stat-title">告警触发</div>
             <div class="stat-val mono">{{ camera.stats.alerts_emitted }}</div>
          </div>
          <div class="stat-card">
             <div class="stat-title">平均延迟</div>
             <div class="stat-val mono">{{ camera.stats.avg_latency_ms?.toFixed(1) }}<small>ms</small></div>
          </div>
        </div>

        <!-- Meta Panels -->
        <div class="meta-scroll">
          <div class="meta-section" v-if="basicEntries.length">
             <div class="meta-section-hd">基础信息</div>
             <div class="meta-section-bd">
                <div class="meta-row" v-for="item in basicEntries" :key="item.key">
                  <span class="meta-k">{{ item.label }}</span>
                  <span class="meta-v">{{ item.value }}</span>
                </div>
             </div>
          </div>
        </div>
      </aside>
    </div>
  </main>
</template>

<style scoped>
main { flex:1; display:flex; flex-direction:column; min-width:0; gap:12px; height: 100%; min-height: 0; }

.topbar { height:60px; flex-shrink:0; display:flex; align-items:center; justify-content:space-between; padding:0 22px; }
.topbar h1 { font-size:18px; font-weight:700; color:var(--ink); letter-spacing:-0.028em; margin:0; }
.crumb { font-size:12px; color:var(--ink-5); font-weight:500; }
.sep { width:1px; height:14px; background:var(--line-2); }
.d { width: 8px; height: 8px; border-radius: 50%; }
.d-green { background: var(--green); box-shadow: 0 0 0 3px rgba(21,163,74,.18); animation: breathe 2s infinite; }
.d-red { background: var(--red); box-shadow: 0 0 0 3px rgba(229,72,77,.18); }

.viewport { padding-right:0; gap: 12px; }

.main-panel {
  display: flex; flex-direction: column; overflow: hidden;
}
.panel-header {
  padding: 10px 16px; border-bottom: 1px solid var(--line-2); display: flex; align-items: center;
}
.seg button { font-size:11.5px; font-weight:600; color:var(--ink-4); padding:5px 16px; border:none; background:transparent; border-radius:6px; cursor:pointer; transition:all .15s; letter-spacing:.005em; }
.seg button:hover { color:var(--ink-2); }
.seg button.on { background:#fff; color:var(--ink); box-shadow:0 1px 2px rgba(10,10,15,.08),0 0 0 0.5px rgba(10,10,15,.04); }

.panel-body {
  flex: 1; padding: 16px; min-height: 0; overflow-y: auto; display: flex; flex-direction: column;
}
.panel-body::-webkit-scrollbar{width:6px}
.panel-body::-webkit-scrollbar-thumb{background:rgba(10,10,15,.10);border-radius:4px}
.panel-body::-webkit-scrollbar-thumb:hover{background:rgba(10,10,15,.18)}

.live-wrapper, .zones-wrapper { flex: 1; display: flex; flex-direction: column; height: 100%; min-height: 0; }
.info-wrapper { flex: 1; display: flex; flex-direction: column; }
.wide-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; align-items: start; }

.video-container {
  flex: 1; border-radius: 8px; overflow: hidden; background: #000; position: relative; display: flex; align-items: center; justify-content: center;
  box-shadow: 0 4px 20px rgba(0,0,0,.15);
}
.player-vid {
  width: 100%; height: 100%; object-fit: contain; background: #000;
}
.overlay-msg {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #fff; font-size: 14px; background: rgba(0,0,0,.6); padding: 8px 16px; border-radius: 8px;
  backdrop-filter: blur(8px);
}
.offline-state {
  background: var(--bg); border: 1px dashed var(--line-3); color: var(--ink-5); flex-direction: column; gap: 12px;
}
.offline-state svg { width: 32px; height: 32px; }
.offline-state span { font-weight: 500; font-size: 13px; }

.right { width: 320px; flex-shrink: 0; padding: 16px; display: flex; flex-direction: column; gap: 16px; overflow: hidden; border-radius: var(--r-lg); }
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; flex-shrink: 0; }
.stat-card { background: var(--bg); border: 1px solid var(--line-2); padding: 12px; border-radius: 8px; display: flex; flex-direction: column; gap: 6px; }
.stat-title { font-size: 11px; color: var(--ink-4); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-val { font-size: 18px; color: var(--ink-2); font-weight: 700; display: flex; align-items: baseline; gap: 2px; }
.stat-val small { font-size: 10px; color: var(--ink-5); font-weight: 600; }

.meta-scroll { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; padding-right: 4px; }
.meta-scroll::-webkit-scrollbar{width:6px}
.meta-scroll::-webkit-scrollbar-thumb{background:rgba(10,10,15,.10);border-radius:4px}
.meta-scroll::-webkit-scrollbar-thumb:hover{background:rgba(10,10,15,.18)}

.meta-section { border: 1px solid var(--line-2); background: var(--bg); border-radius: 8px; overflow: hidden; }
.meta-section-hd { padding: 10px 14px; border-bottom: 1px solid var(--line-2); font-size: 11.5px; font-weight: 600; color: var(--ink-3); background: rgba(10,10,15,.02); }
.meta-section-bd { padding: 8px 14px; }
.meta-row { display: flex; justify-content: space-between; align-items: flex-start; padding: 6px 0; border-bottom: 1px dashed var(--line-2); font-size: 12px; line-height: 1.6; }
.meta-row:last-child { border-bottom: none; }
.meta-k { color: var(--ink-4); font-size: 11.5px; min-width: 80px; max-width: 120px; }
.meta-v { color: var(--ink-2); font-size: 12px; text-align: right; word-break: break-all; }
.mono { font-family: 'JetBrains Mono', monospace; }
</style>
