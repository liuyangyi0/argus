<script setup lang="ts">
import { Button } from 'ant-design-vue'
import AsyncState from '../common/AsyncState.vue'
import StatusBanner from '../common/StatusBanner.vue'

withDefaults(defineProps<{
  loading?: boolean
  loaded?: boolean
  error?: string
  cameraCount?: number
}>(), {
  loading: false,
  loaded: false,
  error: '',
  cameraCount: 0,
})

const emit = defineEmits<{ retry: [] }>()
</script>

<template>
  <AsyncState
    :loading="loading && !loaded"
    :refreshing="loading && loaded"
    skeleton-type="description"
    :skeleton-rows="6"
  >
    <StatusBanner
      v-if="error"
      class="model-camera-state__banner"
      status="error"
      title="摄像头列表加载失败"
      :description="error"
    >
      <template #action>
        <Button size="small" :loading="loading" @click="emit('retry')">重试</Button>
      </template>
    </StatusBanner>
    <StatusBanner
      v-else-if="loaded && cameraCount === 0"
      class="model-camera-state__banner"
      status="neutral"
      title="尚未配置摄像头"
      description="现有记录仍可查看；创建采集、训练或评估任务前需要先配置摄像头。"
    />
    <slot />
  </AsyncState>
</template>

<style scoped>
.model-camera-state__banner {
  margin-bottom: 12px;
}
</style>
