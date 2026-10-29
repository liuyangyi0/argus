<script setup lang="ts">
import type { AnalysisTask } from '../../api/analysisTasks'
import type { TaskGuidanceResponse } from '../../types/generated-openapi'

defineProps<{ task: AnalysisTask; actionable?: boolean; stale?: boolean }>()
const emit = defineEmits<{ open: [target: TaskGuidanceResponse['target']] }>()
</script>

<template>
  <section class="task-guidance" aria-label="任务下一步指引" data-test="task-guidance">
    <template v-if="task.guidance?.schema_version === 'argus.task-guidance.v1' && task.guidance.is_authorization === false">
      <h4>{{ task.guidance.title }}</h4>
      <p>{{ task.guidance.detail }}</p>
      <p v-if="task.guidance.training_requirement === 'optional'" class="scope">已有兼容模型可直接验证；采集和训练是按需定制路径，不是必经步骤。</p>
      <p v-else-if="task.guidance.training_requirement === 'not_applicable'" class="scope">固定算法任务：配置采样与规则即可，不需要模型权重或训练。</p>
      <p v-else class="scope">异物任务沿用场景 → 正常基线 → 训练 → 独立校准 → 原受控发布流程。</p>
      <p v-if="stale" role="status">状态已过期，请先刷新；下一步入口暂不可用。</p>
      <button v-if="actionable && task.guidance.target !== 'none'" type="button" :disabled="stale" @click="emit('open', task.guidance.target)">{{ task.guidance.action_label }}</button>
      <p class="scope">这是只读流程指引，不是启动许可、场景放行或生产审批。</p>
    </template>
    <p v-else role="status">流程指引尚未核实，请刷新任务。不能据此判断检测正常。</p>
  </section>
</template>

<style scoped>
.task-guidance { padding: 12px; margin: 12px 0; border: 1px solid var(--border-default); border-left: 3px solid var(--status-info); border-radius: var(--r-md, 8px); }
h4 { margin: 0; font-size: 14px; color: var(--text-primary); }
p { margin: 6px 0; font-size: 13px; line-height: 1.6; overflow-wrap: anywhere; }
.scope { color: var(--text-secondary); font-size: 12px; }
button { padding: 7px 12px; color: var(--text-primary); background: var(--bg); border: 1px solid var(--border-default); border-radius: 6px; cursor: pointer; }
button:disabled { cursor: not-allowed; opacity: .6; }
button:focus-visible { outline: 2px solid var(--status-info); outline-offset: 2px; }
</style>
