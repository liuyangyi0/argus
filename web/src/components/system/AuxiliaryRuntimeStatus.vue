<script setup lang="ts">
import { computed } from 'vue'
import { Alert } from 'ant-design-vue'
import type { AuxiliaryRuntimeSummary } from '../../api/system'
import { auxiliaryStateLabel, describeAuxiliaryRuntime } from '../../utils/auxiliaryRuntime'

const props = defineProps<{ label: string; enabled: boolean; runtime: AuxiliaryRuntimeSummary }>()
const status = computed(() => describeAuxiliaryRuntime(props.label, props.enabled, props.runtime))
</script>

<template>
  <div class="auxiliary-runtime" aria-live="polite">
    <Alert :type="status.type" :message="status.message" show-icon />
    <ul v-if="runtime.details?.length" class="runtime-details" :aria-label="`${label}逐相机运行状态`">
      <li v-for="detail in runtime.details" :key="detail.camera_id">
        <strong>{{ detail.camera_id }}</strong>：{{ auxiliaryStateLabel(detail.state) }}
        <span> · 后端：{{ detail.backend || '未加载' }}</span>
        <span v-if="detail.busy"> · 任务尚未结束</span>
        <span v-if="detail.reason"> · {{ detail.reason }}</span>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.runtime-details { margin: 8px 0 0; padding-left: 20px; font-size: 12px; overflow-wrap: anywhere; }
.runtime-details li + li { margin-top: 4px; }
</style>
