<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { Breadcrumb } from 'ant-design-vue'
import PageShell from '../components/common/PageShell.vue'

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
const pageTitle = computed(() => (
  sectionTitle.value ? `模型管理 / ${sectionTitle.value}` : '模型管理'
))
</script>

<template>
  <PageShell
    class="models-page"
    :title="pageTitle"
    description="模型发布只在运行时版本、制品摘要和场景配置全部核对后才算生效。"
  >
    <template #meta>
      <Breadcrumb>
        <Breadcrumb.Item>模型管理</Breadcrumb.Item>
        <Breadcrumb.Item v-if="sectionTitle">{{ sectionTitle }}</Breadcrumb.Item>
      </Breadcrumb>
    </template>
    <div class="models-page__content">
      <router-view />
    </div>
  </PageShell>
</template>

<style scoped>
.models-page__content {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-height: 0;
}
</style>
