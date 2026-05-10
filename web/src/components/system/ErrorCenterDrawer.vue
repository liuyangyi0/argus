<template>
  <a-drawer
    :open="errorStore.drawerOpen"
    title="错误中心"
    placement="right"
    :width="520"
    @close="errorStore.closeDrawer"
  >
    <template #extra>
      <a-space>
        <a-button size="small" @click="errorStore.markAllRead" :disabled="!errorStore.totalUnread">
          全部标记已读
        </a-button>
        <a-popconfirm title="确定清空所有错误?" @confirm="errorStore.clear">
          <a-button size="small" danger>清空</a-button>
        </a-popconfirm>
      </a-space>
    </template>

    <a-empty v-if="!errorStore.errors.length" description="暂无错误事件" />

    <a-list v-else :data-source="errorStore.errors" item-layout="vertical" size="small">
      <template #renderItem="{ item }">
        <a-list-item :key="item.id" :class="{ 'is-unread': !item.read }">
          <a-list-item-meta>
            <template #title>
              <a-space :size="8" wrap>
                <a-tag :color="severityColor(item.severity)">{{ severityLabel(item.severity) }}</a-tag>
                <span class="error-source">{{ item.source }}</span>
                <span class="error-code">{{ item.code }}</span>
              </a-space>
            </template>
            <template #description>
              <span class="error-time">{{ formatTime(item.timestamp) }}</span>
            </template>
          </a-list-item-meta>

          <div class="error-message">{{ item.message }}</div>

          <details v-if="hasContext(item)" class="error-context">
            <summary>上下文</summary>
            <pre>{{ JSON.stringify(item.context, null, 2) }}</pre>
          </details>
        </a-list-item>
      </template>
    </a-list>
  </a-drawer>
</template>

<script setup lang="ts">
import { useErrorStore, type SystemError } from '../../stores/useErrorStore'

const errorStore = useErrorStore()

function severityColor(s: string) {
  return ({ critical: 'red', error: 'volcano', warning: 'gold', info: 'blue' } as Record<string, string>)[s] || 'default'
}
function severityLabel(s: string) {
  return ({ critical: '严重', error: '错误', warning: '警告', info: '提示' } as Record<string, string>)[s] || s
}
function formatTime(ts: string) {
  try {
    return new Date(ts).toLocaleString('zh-CN', { hour12: false })
  } catch { return ts }
}
function hasContext(e: SystemError) {
  return e.context && Object.keys(e.context).length > 0
}
</script>

<style scoped>
.is-unread { background: rgba(255, 77, 79, 0.04); }
.error-source { color: var(--ink-3, #999); font-size: 12px; }
.error-code { font-family: monospace; font-size: 12px; opacity: 0.7; }
.error-time { font-size: 12px; opacity: 0.6; }
.error-message { margin-top: 4px; }
.error-context { margin-top: 6px; font-size: 12px; }
.error-context summary { cursor: pointer; opacity: 0.7; }
.error-context pre { background: rgba(0,0,0,0.04); padding: 6px; border-radius: 4px; max-height: 200px; overflow: auto; }
</style>
