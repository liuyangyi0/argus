<script setup lang="ts">
import { computed } from 'vue'
import { Button, Result } from 'ant-design-vue'
import ContentSkeleton from '../ContentSkeleton.vue'
import EmptyState from '../EmptyState.vue'
import StatusBanner from './StatusBanner.vue'

const props = withDefaults(defineProps<{
  loading?: boolean
  refreshing?: boolean
  error?: string
  forbidden?: boolean
  notFound?: boolean
  empty?: boolean
  loadingLabel?: string
  refreshingLabel?: string
  emptyTitle?: string
  emptyDescription?: string
  emptyActionText?: string
  forbiddenDescription?: string
  notFoundDescription?: string
  skeletonType?: 'card' | 'table' | 'description'
  skeletonRows?: number
}>(), {
  loading: false,
  refreshing: false,
  error: '',
  forbidden: false,
  notFound: false,
  empty: false,
  loadingLabel: '正在加载数据',
  refreshingLabel: '正在同步最新状态',
  emptyTitle: '暂无数据',
  emptyDescription: '',
  emptyActionText: '',
  forbiddenDescription: '当前账号没有查看此内容的权限。',
  notFoundDescription: '目标可能已被删除，或链接已失效。',
  skeletonType: 'description',
  skeletonRows: 6,
})

const emit = defineEmits<{
  retry: []
  emptyAction: []
}>()

const state = computed(() => {
  if (props.forbidden) return 'forbidden'
  if (props.notFound) return 'not-found'
  if (props.loading) return 'loading'
  if (props.error) return 'error'
  if (props.empty) return 'empty'
  return 'ready'
})
</script>

<template>
  <div class="async-state" :data-state="state" :aria-busy="state === 'loading' || refreshing">
    <Result
      v-if="state === 'forbidden'"
      status="403"
      title="无权访问"
      :sub-title="forbiddenDescription"
    >
      <template #extra>
        <slot name="forbiddenAction">
          <Button type="primary" aria-label="重试加载数据" @click="emit('retry')">重试</Button>
        </slot>
      </template>
    </Result>

    <Result
      v-else-if="state === 'not-found'"
      status="404"
      title="内容不存在"
      :sub-title="notFoundDescription"
    >
      <template #extra>
        <slot name="notFoundAction">
          <Button type="primary" aria-label="重试加载数据" @click="emit('retry')">重试</Button>
        </slot>
      </template>
    </Result>

    <div v-else-if="state === 'loading'" class="async-state__loading" role="status" aria-live="polite">
      <span class="async-state__sr-only">{{ loadingLabel }}</span>
      <ContentSkeleton :type="skeletonType" :rows="skeletonRows" />
    </div>

    <Result
      v-else-if="state === 'error'"
      status="500"
      title="数据加载失败"
      :sub-title="error"
      role="alert"
    >
      <template #extra>
        <Button type="primary" aria-label="重试加载数据" @click="emit('retry')">重试</Button>
      </template>
    </Result>

    <EmptyState
      v-else-if="state === 'empty'"
      :title="emptyTitle"
      :description="emptyDescription"
      :action-text="emptyActionText"
      @action="emit('emptyAction')"
    />

    <template v-else>
      <StatusBanner
        v-if="refreshing"
        class="async-state__refresh"
        status="loading"
        compact
        :title="refreshingLabel"
      />
      <slot />
    </template>
  </div>
</template>

<style scoped>
.async-state {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
}

.async-state__loading {
  flex: 1;
}

.async-state__refresh {
  flex: 0 0 auto;
  margin-bottom: 10px;
}

.async-state__sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
</style>
