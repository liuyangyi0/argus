<script setup lang="ts">
import { computed } from 'vue'
import type { PipelineMode } from '../../types/api'
import { modeBadge } from '../../composables/useSystemMode'

const props = defineProps<{
  mode?: PipelineMode | string | null
  /** When true, force-render the badge even for the "active" mode. */
  alwaysShow?: boolean
}>()

const meta = computed(() => modeBadge(props.mode))
const visible = computed(() => props.alwaysShow || meta.value.visible)
</script>

<template>
  <span
    v-if="visible"
    class="mode-badge"
    :style="{ backgroundColor: meta.color }"
  >
    {{ meta.label }}
  </span>
</template>

<style scoped>
.mode-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 10px;
  color: #fff;
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.3px;
  line-height: 1.4;
  white-space: nowrap;
}
</style>
