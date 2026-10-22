<script setup lang="ts">
import { computed } from 'vue'
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  LoadingOutlined,
} from '@ant-design/icons-vue'

const props = withDefaults(defineProps<{
  status?: 'info' | 'success' | 'warning' | 'error' | 'neutral' | 'loading'
  title: string
  description?: string
  compact?: boolean
}>(), {
  status: 'info',
  description: '',
  compact: false,
})

const icon = computed(() => ({
  success: CheckCircleOutlined,
  warning: ExclamationCircleOutlined,
  error: CloseCircleOutlined,
  loading: LoadingOutlined,
  info: InfoCircleOutlined,
  neutral: InfoCircleOutlined,
}[props.status]))

const role = computed(() => props.status === 'error' ? 'alert' : 'status')
</script>

<template>
  <div
    class="status-banner"
    :class="[`status-banner--${status}`, { 'status-banner--compact': compact }]"
    :role="role"
    :aria-live="status === 'error' ? 'assertive' : 'polite'"
  >
    <component :is="icon" class="status-banner__icon" :spin="status === 'loading'" aria-hidden="true" />
    <div class="status-banner__body">
      <div class="status-banner__title">{{ title }}</div>
      <div v-if="description" class="status-banner__description">{{ description }}</div>
    </div>
    <div v-if="$slots.action" class="status-banner__action">
      <slot name="action" />
    </div>
  </div>
</template>

<style scoped>
.status-banner {
  --banner-color: #1d4ed8;
  --banner-border: rgba(29, 78, 216, 0.28);
  --banner-bg: rgba(59, 130, 246, 0.08);
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 12px 14px;
  border: 1px solid var(--banner-border);
  border-radius: var(--r-sm, 6px);
  background: var(--banner-bg);
  color: var(--banner-color);
}

.status-banner--success {
  --banner-color: #166534;
  --banner-border: rgba(22, 101, 52, 0.28);
  --banner-bg: rgba(34, 197, 94, 0.08);
}

.status-banner--warning {
  --banner-color: #92400e;
  --banner-border: rgba(146, 64, 14, 0.3);
  --banner-bg: rgba(245, 158, 11, 0.1);
}

.status-banner--error {
  --banner-color: #991b1b;
  --banner-border: rgba(153, 27, 27, 0.3);
  --banner-bg: rgba(239, 68, 68, 0.08);
}

.status-banner--neutral {
  --banner-color: var(--ink-3);
  --banner-border: var(--line-2);
  --banner-bg: rgba(15, 23, 42, 0.035);
}

.status-banner--loading {
  --banner-color: #1e40af;
  --banner-border: rgba(30, 64, 175, 0.22);
  --banner-bg: rgba(59, 130, 246, 0.06);
}

.status-banner--compact {
  align-items: center;
  padding: 8px 10px;
}

.status-banner__icon {
  flex: 0 0 auto;
  margin-top: 2px;
  font-size: 16px;
}

.status-banner--compact .status-banner__icon {
  margin-top: 0;
}

.status-banner__body {
  flex: 1;
  min-width: 0;
}

.status-banner__title {
  color: currentColor;
  font-size: 13px;
  font-weight: 650;
  line-height: 1.5;
}

.status-banner__description {
  margin-top: 2px;
  color: var(--ink-3);
  font-size: 12px;
  line-height: 1.55;
}

.status-banner__action {
  flex: 0 0 auto;
}
</style>
