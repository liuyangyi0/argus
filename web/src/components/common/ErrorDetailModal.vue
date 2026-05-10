<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  open: boolean
  title?: string
  error?: string | null
  /** Optional traceback text (collapsible). */
  traceback?: string | null
}>()
const emit = defineEmits<{ (e: 'update:open', value: boolean): void }>()

const visible = computed({
  get: () => props.open,
  set: (v: boolean) => emit('update:open', v),
})
</script>

<template>
  <a-modal
    v-model:open="visible"
    :title="props.title || '失败原因'"
    :footer="null"
    width="640px"
  >
    <div class="error-modal">
      <p class="error-message">{{ props.error || '（暂无失败原因）' }}</p>
      <a-collapse v-if="props.traceback" ghost>
        <a-collapse-panel key="traceback" header="完整堆栈">
          <pre class="traceback">{{ props.traceback }}</pre>
        </a-collapse-panel>
      </a-collapse>
    </div>
  </a-modal>
</template>

<style scoped>
.error-modal {
  font-size: 13px;
  line-height: 1.5;
}
.error-message {
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0 0 12px;
}
.traceback {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  background: #fafafa;
  padding: 12px;
  border-radius: 4px;
  max-height: 320px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
