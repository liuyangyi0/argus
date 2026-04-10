<script setup lang="ts">
import { ref, onErrorCaptured } from 'vue'
import { Result, Button, Typography } from 'ant-design-vue'
import { ReloadOutlined } from '@ant-design/icons-vue'

const hasError = ref(false)
const errorMessage = ref('')

onErrorCaptured((err: Error) => {
  hasError.value = true
  errorMessage.value = err.message || '未知错误'
  return false // prevent propagation
})

function retry() {
  hasError.value = false
  errorMessage.value = ''
}
</script>

<template>
  <slot v-if="!hasError" />
  <Result v-else status="error" title="页面出现异常" :sub-title="errorMessage">
    <template #extra>
      <Button type="primary" @click="retry">
        <template #icon><ReloadOutlined /></template>
        重试
      </Button>
    </template>
    <Typography.Paragraph type="secondary" style="text-align: center; font-size: 12px">
      如果问题持续出现，请刷新页面或联系管理员
    </Typography.Paragraph>
  </Result>
</template>
