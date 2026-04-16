<script setup lang="ts">
import { useRoute, useRouter } from 'vue-router'
import ReplayPlayer from '../components/ReplayPlayer.vue'

defineOptions({ name: 'ReplayPage' })

const route = useRoute()
const router = useRouter()

const alertId = route.params.alertId as string
</script>

<template>
  <main class="replay-page">
    <div class="replay-topbar glass">
      <button class="ibtn" @click="router.back()" title="返回">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
        </svg>
      </button>
      <h1>录像回放</h1>
      <span class="sep"></span>
      <span class="crumb mono">{{ alertId }}</span>
    </div>

    <div class="replay-body">
      <ReplayPlayer :alert-id="alertId" />
    </div>
  </main>
</template>

<style scoped>
.replay-page {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  gap: 12px;
}

.replay-topbar {
  height: 60px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 0 22px;
}

.replay-topbar h1 {
  font-size: 18px;
  font-weight: 700;
  color: var(--ink);
  letter-spacing: -0.028em;
  margin: 0;
}

.sep {
  width: 1px;
  height: 14px;
  background: var(--line-2);
}

.crumb {
  font-size: 12px;
  color: var(--ink-5);
  font-weight: 500;
}

.mono {
  font-family: 'JetBrains Mono', monospace;
}

.ibtn {
  width: 34px;
  height: 34px;
  border-radius: var(--r-sm);
  display: grid;
  place-items: center;
  background: rgba(255, 255, 255, 0.7);
  border: 0.5px solid var(--line-2);
  color: var(--ink-3);
  cursor: pointer;
  transition: all 0.18s;
  backdrop-filter: blur(20px);
}

.ibtn:hover {
  background: #fff;
  color: var(--ink);
  box-shadow: var(--sh-1);
}

.ibtn svg {
  width: 15px;
  height: 15px;
  stroke-width: 1.9;
}

.replay-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 0 4px 16px;
}
</style>
