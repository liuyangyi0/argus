<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Button } from 'ant-design-vue'
import { ArrowLeftOutlined } from '@ant-design/icons-vue'
import ReplayPlayer from '../components/ReplayPlayer.vue'
import PageShell from '../components/common/PageShell.vue'

defineOptions({ name: 'ReplayPage' })

const route = useRoute()
const router = useRouter()

const alertId = computed(() => String(route.params.alertId || ''))
</script>

<template>
  <PageShell
    class="replay-page"
    title="录像回放"
    :description="`告警 ${alertId}`"
  >
    <template #actions>
      <Button aria-label="返回上一页" @click="router.back()">
        <template #icon><ArrowLeftOutlined /></template>
        返回
      </Button>
    </template>
    <ReplayPlayer :key="alertId" :alert-id="alertId" />
  </PageShell>
</template>

<style scoped>
.replay-page {
  overflow-y: auto;
}
</style>
