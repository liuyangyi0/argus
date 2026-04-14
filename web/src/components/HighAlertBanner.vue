<script setup lang="ts">
import { BellOutlined } from '@ant-design/icons-vue'
import { Typography, Button } from 'ant-design-vue'
import { useRouter } from 'vue-router'
import type { CameraTileData } from './VideoTile.vue'

defineProps<{
  cameras: CameraTileData[]
}>()

const router = useRouter()
</script>

<template>
  <div class="high-alert-banner">
    <div class="high-alert-inner">
      <div class="high-alert-pulse" />
      <BellOutlined style="color: #e5484d; font-size: 16px" />
      <Typography.Text strong style="color: #e5484d; font-size: 13px">
        {{ cameras.length }} 个高级告警需要立即处理
      </Typography.Text>
      <Typography.Text type="secondary" class="alert-camera-list" style="font-size: 12px; color: #e5484daa">
        {{ cameras.map(c => c.name || c.camera_id).join(', ') }}
      </Typography.Text>
      <Button
        type="primary"
        danger
        size="small"
        style="margin-left: auto"
        @click="router.push('/alerts?severity=high')"
      >
        快速响应
      </Button>
    </div>
  </div>
</template>

<style scoped>
.high-alert-banner {
  background: rgba(239, 68, 68, 0.08);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 8px;
  overflow: hidden;
  position: relative;
  flex-shrink: 0;
}

.high-alert-inner {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  position: relative;
  z-index: 1;
}

.high-alert-pulse {
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent 0%, rgba(239, 68, 68, 0.05) 50%, transparent 100%);
  animation: alert-sweep 3s ease-in-out infinite;
}

.alert-camera-list {
  max-width: 300px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

@keyframes alert-sweep {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

@media (max-width: 640px) {
  .alert-camera-list {
    display: none;
  }
}
</style>
