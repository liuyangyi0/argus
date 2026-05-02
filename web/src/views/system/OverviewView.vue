<script setup lang="ts">
import SystemOverviewPanel from '../../components/system/SystemOverviewPanel.vue'
import ModelStatusPanel from '../../components/system/ModelStatusPanel.vue'
</script>

<template>
  <div>
    <SystemOverviewPanel />
    <div style="margin-top: 16px">
      <ModelStatusPanel />
    </div>
  </div>
</template>
