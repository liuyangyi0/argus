<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import { getModuleStates, toggleModule as updateModuleToggle } from '../../api'
import { extractErrorMessage } from '../../utils/error'
import { notifyModuleToggleResult, type ModuleToggleFeedback } from '../../utils/moduleToggleFeedback'

interface ModuleToggle {
  key: string
  label: string
  description: string
  enabled: boolean
  available?: boolean
  feedback?: ModuleToggleFeedback
}

const modules = ref<ModuleToggle[]>([
  {
    key: 'imaging.enabled',
    label: '多模态成像',
    description: 'DoFP偏振去反射 + NIR频闪补光',
    enabled: false,
  },
  {
    key: 'classifier.enabled',
    label: 'AI 异物分类',
    description: 'YOLO-World 开放词表分类（工业物品词表）',
    enabled: false,
  },
  {
    key: 'segmenter.enabled',
    label: 'SAM2 实例分割',
    description: '精确分割异常区域边界（支持热重载）',
    enabled: false,
  },
  {
    key: 'cross_camera.enabled',
    label: '跨相机关联',
    description: '多摄像头重叠视野下的相互证实（切换后需要重启已运行的摄像头）',
    enabled: false,
  },
  {
    key: 'physics.speed_enabled',
    label: '速度监测',
    description: '异物坠落速度实时监测',
    enabled: false,
  },
  {
    key: 'continuous_recording.enabled',
    label: '连续录像',
    description: '24/7 连续录像，4小时分段，180天本地保留',
    enabled: false,
  },
  {
    key: 'physics.trajectory_enabled',
    label: '轨迹分析（二期）',
    description: '自由落体/抛物线轨迹拟合、坠落起始位置估算',
    enabled: false,
  },
  {
    key: 'physics.localization_enabled',
    label: '落点定位（二期）',
    description: '异物入水点定位（±500mm）',
    enabled: false,
  },
])

const loading = ref(false)
const loadingConfig = ref(false)
const configError = ref<string | null>(null)

async function loadConfig() {
  if (loadingConfig.value) return
  loadingConfig.value = true
  configError.value = null
  try {
    const states = await getModuleStates()
    for (const mod of modules.value) {
      mod.available = mod.key in states && typeof states[mod.key] === 'boolean'
      if (mod.available) {
        mod.enabled = states[mod.key]
      }
    }
  } catch (e) {
    configError.value = extractErrorMessage(e, '加载功能模块状态失败')
    for (const mod of modules.value) mod.available = false
  } finally {
    loadingConfig.value = false
  }
}

async function toggleModule(mod: ModuleToggle, checked: boolean) {
  if (loading.value || loadingConfig.value || !mod.available) return
  loading.value = true
  try {
    const result = await updateModuleToggle(mod.key, checked)
    mod.enabled = result.value
    mod.feedback = notifyModuleToggleResult(mod.label, checked, result)
  } catch (e) {
    message.error(extractErrorMessage(e, `${mod.label} 切换失败`))
    mod.feedback = { type: 'warning', message: '切换请求失败，运行时状态尚未确认；已重新读取配置，请核验后再操作。' }
    await loadConfig()
  } finally {
    loading.value = false
  }
}

onMounted(loadConfig)
</script>

<template>
  <a-card title="功能模块" :bordered="false">
    <p class="module-notice">开关表示配置目标，不代表模型已经加载或识别效果已验收；分类器与分割器的逐相机状态请查看对应面板。</p>
    <div v-if="configError" class="module-error" role="alert">
      {{ configError }}
      <button type="button" :disabled="loadingConfig" @click="loadConfig">重新加载状态</button>
    </div>
    <div class="module-list">
      <div v-for="mod in modules" :key="mod.key" class="module-item">
        <div class="module-info">
          <span class="module-label">{{ mod.label }}</span>
          <span class="module-desc">{{ mod.description }}</span>
          <span v-if="!mod.available" class="module-status" role="status">{{ loadingConfig ? '状态加载中' : '配置状态不可用，禁止切换' }}</span>
          <span v-if="mod.feedback" class="module-status" role="status">{{ mod.feedback.message }}</span>
        </div>
        <a-switch
          :checked="mod.enabled"
          :loading="loading"
          :disabled="loadingConfig || !mod.available"
          :aria-label="`${mod.label}配置开关`"
          @change="(checked: boolean) => toggleModule(mod, checked)"
        />
      </div>
    </div>
  </a-card>
</template>

<style scoped>
.module-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.module-notice, .module-status { font-size: 12px; color: var(--argus-text-muted); }
.module-error { color: var(--argus-danger, #b42318); margin-bottom: 12px; }
.module-status { overflow-wrap: anywhere; }
.module-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid var(--border-color, #f0f0f0);
}
.module-item:last-child {
  border-bottom: none;
}
.module-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.module-label {
  font-weight: 500;
  font-size: 14px;
}
.module-desc {
  font-size: 12px;
  color: var(--text-secondary, #888);
}
</style>
