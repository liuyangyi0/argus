<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { Breadcrumb, Typography } from 'ant-design-vue'

const route = useRoute()

const SECTION_TITLES: Record<string, string> = {
  'models-baseline': '基线管理',
  'models-training': '训练与评估',
  'models-registry': '模型与发布',
  'models-comparison': 'A/B 对比',
  'models-labeling': '标注队列',
  'models-threshold': '阈值预览',
}

const sectionTitle = computed(() => {
  const name = String(route.name || '')
  return SECTION_TITLES[name] || ''
})
</script>

<template>
  <main class="glass" style="padding: 24px; border-radius: var(--r-lg); min-width: 0; display: flex; flex-direction: column; flex: 1;">
    <Breadcrumb style="margin-bottom: 12px">
      <Breadcrumb.Item>模型管理</Breadcrumb.Item>
      <Breadcrumb.Item v-if="sectionTitle">{{ sectionTitle }}</Breadcrumb.Item>
    </Breadcrumb>
    <Typography.Title :level="3" style="margin-bottom: 8px; color: var(--ink)">
      模型管理<span v-if="sectionTitle" style="color: var(--ink-4); font-weight: 500"> / {{ sectionTitle }}</span>
    </Typography.Title>
    <div style="flex: 1; min-height: 0; display: flex; flex-direction: column;">
      <router-view />
    </div>
  </main>
</template>
