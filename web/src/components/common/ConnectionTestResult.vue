<script setup lang="ts">
import { computed } from 'vue'
import type { ConnectionTestResult } from '../../types/api'

const props = defineProps<{
  state: 'idle' | 'testing' | 'done'
  result?: ConnectionTestResult | null
}>()

const palette = computed(() => {
  if (props.state === 'testing') return { color: '#1890ff', icon: '⏳', label: '正在连接...' }
  if (props.state === 'done' && props.result?.ok) {
    return { color: '#52c41a', icon: '✓', label: '连接成功' }
  }
  if (props.state === 'done') {
    return { color: '#ff4d4f', icon: '✗', label: '连接失败' }
  }
  return { color: '#bfbfbf', icon: '•', label: '尚未测试' }
})
</script>

<template>
  <div class="conn-test" :style="{ borderColor: palette.color }">
    <span class="icon" :style="{ color: palette.color }">{{ palette.icon }}</span>
    <div class="body">
      <div class="title" :style="{ color: palette.color }">{{ palette.label }}</div>
      <div v-if="props.state === 'done' && props.result" class="meta">
        <template v-if="props.result.ok">
          <span v-if="props.result.latency_ms != null">
            延迟 {{ props.result.latency_ms }} ms
          </span>
          <span v-if="props.result.resolution">
            · 分辨率 {{ props.result.resolution[0] }}×{{ props.result.resolution[1] }}
          </span>
        </template>
        <template v-else>
          <span>{{ props.result.error || '未知错误' }}</span>
          <span v-if="props.result.latency_ms != null">
            · 已用 {{ props.result.latency_ms }} ms
          </span>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.conn-test {
  display: flex;
  gap: 10px;
  padding: 8px 12px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  align-items: flex-start;
  background: #fafafa;
}
.icon {
  font-size: 18px;
  line-height: 1.2;
  font-weight: 700;
}
.body {
  flex: 1;
  min-width: 0;
}
.title {
  font-weight: 600;
  font-size: 13px;
  margin-bottom: 2px;
}
.meta {
  font-size: 12px;
  color: #595959;
  word-break: break-word;
}
</style>
