<script setup lang="ts">
import { Typography, Button } from 'ant-design-vue'
import {
  VideoCameraOutlined,
  CheckCircleOutlined,
  DatabaseOutlined,
} from '@ant-design/icons-vue'

const props = defineProps<{
  icon?: 'camera' | 'check' | 'database'
  title: string
  description?: string
  actionText?: string
  actionRoute?: string
}>()

const emit = defineEmits<{ action: [] }>()

const iconMap = {
  camera: VideoCameraOutlined,
  check: CheckCircleOutlined,
  database: DatabaseOutlined,
}
</script>

<template>
  <div class="empty-state">
    <component
      :is="iconMap[icon ?? 'database']"
      class="empty-state__icon"
    />
    <Typography.Title :level="5" class="empty-state__title">{{ title }}</Typography.Title>
    <Typography.Text v-if="description" type="secondary" class="empty-state__desc">{{ description }}</Typography.Text>
    <Button
      v-if="actionText"
      type="primary"
      class="empty-state__action"
      @click="actionRoute ? $router.push(actionRoute) : emit('action')"
    >
      {{ actionText }}
    </Button>
  </div>
</template>

<style scoped>
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 48px 24px;
  text-align: center;
}

.empty-state__icon {
  font-size: 48px;
  color: var(--ink-5);
  margin-bottom: 16px;
}

.empty-state__title {
  margin-bottom: 8px !important;
  color: var(--ink-4) !important;
}

.empty-state__desc {
  margin-bottom: 16px;
}

.empty-state__action {
  margin-top: 8px;
}
</style>
