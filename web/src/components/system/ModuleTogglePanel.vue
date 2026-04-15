<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import axios from 'axios'

interface ModuleToggle {
  key: string
  label: string
  description: string
  enabled: boolean
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
    description: 'YOLO-World 开放词表分类（核电专用词表）',
    enabled: false,
  },
  {
    key: 'segmenter.enabled',
    label: 'SAM2 实例分割',
    description: '精确分割异常区域边界（切换后需要重启已运行的摄像头）',
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

async function loadConfig() {
  try {
    const res = await axios.get('/api/config/modules')
    const states = res.data?.data || res.data
    for (const mod of modules.value) {
      if (mod.key in states && typeof states[mod.key] === 'boolean') {
        mod.enabled = states[mod.key]
      }
    }
  } catch {
    // Config endpoint may not expose all fields yet
  }
}

async function toggleModule(mod: ModuleToggle) {
  loading.value = true
  try {
    await axios.post('/api/config/modules', {
      key: mod.key,
      value: mod.enabled,
    })
    message.success(`${mod.label} ${mod.enabled ? '已启用' : '已关闭'}`)
  } catch {
    mod.enabled = !mod.enabled
    message.error(`${mod.label} 切换失败`)
  } finally {
    loading.value = false
  }
}

onMounted(loadConfig)
</script>

<template>
  <a-card title="功能模块" :bordered="false">
    <div class="module-list">
      <div v-for="mod in modules" :key="mod.key" class="module-item">
        <div class="module-info">
          <span class="module-label">{{ mod.label }}</span>
          <span class="module-desc">{{ mod.description }}</span>
        </div>
        <a-switch
          v-model:checked="mod.enabled"
          :loading="loading"
          @change="toggleModule(mod)"
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
