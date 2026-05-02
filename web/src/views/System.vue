<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { Typography } from 'ant-design-vue'

const route = useRoute()

const SECTION_TITLES: Record<string, string> = {
  'system-overview': '系统概览',
  'system-model-status': '模型状态',
  'system-config': '配置管理',
  'system-audit': '审计日志',
  'system-degradation': '降级事件',
  'system-modules': '功能模块',
  'system-classifier': '分类器',
  'system-segmenter': '分割器',
  'system-imaging': '多模态成像',
  'system-cross-camera': '跨相机',
  'system-users': '用户管理',
  'system-regions': '区域管理',
  'system-notification-templates': '通知内容配置',
}

const sectionTitle = computed(() => {
  const name = String(route.name || '')
  return SECTION_TITLES[name] || ''
})
</script>

<template>
  <main class="glass" style="padding: 24px; border-radius: var(--r-lg); min-width: 0; display: flex; flex-direction: column; flex: 1; overflow-y: auto;">
    <Typography.Title :level="3" style="margin-bottom: 8px; color: var(--ink)">
      系统管理<span v-if="sectionTitle" style="color: var(--ink-4); font-weight: 500"> / {{ sectionTitle }}</span>
    </Typography.Title>
    <div style="flex: 1; min-height: 0; display: flex; flex-direction: column;">
      <router-view />
    </div>
  </main>
</template>
