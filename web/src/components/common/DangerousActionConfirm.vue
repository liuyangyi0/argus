<script setup lang="ts">
import { Button, Popconfirm } from 'ant-design-vue'

withDefaults(defineProps<{
  title: string
  description?: string
  actionLabel: string
  ariaLabel?: string
  disabled?: boolean
  loading?: boolean
  size?: 'small' | 'middle' | 'large'
  buttonType?: 'default' | 'primary' | 'dashed' | 'link' | 'text'
  block?: boolean
  confirmText?: string
  cancelText?: string
}>(), {
  description: '',
  ariaLabel: '',
  disabled: false,
  loading: false,
  size: 'small',
  buttonType: 'default',
  block: false,
  confirmText: '确认',
  cancelText: '取消',
})

const emit = defineEmits<{ confirm: [] }>()
</script>

<template>
  <Popconfirm
    :title="title"
    :description="description"
    :ok-text="confirmText"
    :cancel-text="cancelText"
    ok-type="danger"
    :disabled="disabled || loading"
    @confirm="emit('confirm')"
  >
    <Button
      danger
      :type="buttonType"
      :size="size"
      :disabled="disabled"
      :loading="loading"
      :block="block"
      :aria-label="ariaLabel || actionLabel"
    >
      <template v-if="$slots.icon" #icon><slot name="icon" /></template>
      <slot>{{ actionLabel }}</slot>
    </Button>
  </Popconfirm>
</template>
