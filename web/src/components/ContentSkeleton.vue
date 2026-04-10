<script setup lang="ts">
import { Skeleton, Card } from 'ant-design-vue'

defineProps<{
  type?: 'card' | 'table' | 'description'
  rows?: number
}>()
</script>

<template>
  <Card v-if="type === 'card'" :bordered="false" :body-style="{ padding: '16px' }">
    <Skeleton active :paragraph="{ rows: rows ?? 3 }" />
  </Card>
  <div v-else-if="type === 'table'" class="skeleton-table">
    <Skeleton active :paragraph="{ rows: rows ?? 5, width: ['100%', '90%', '95%', '85%', '92%'] }" :title="{ width: '60%' }" />
  </div>
  <Skeleton v-else active :paragraph="{ rows: rows ?? 4 }" />
</template>

<style scoped>
.skeleton-table {
  padding: 16px;
  background: var(--argus-card-bg-solid);
  border-radius: 8px;
}
</style>
