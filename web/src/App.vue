<script setup lang="ts">
import { useThemeStore } from './stores/theme'
const themeStore = useThemeStore()
</script>

<template>
  <a-config-provider :theme="themeStore.themeConfig">
    <router-view />
  </a-config-provider>
</template>


<style>

</style>