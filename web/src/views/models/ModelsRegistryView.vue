<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useModelState } from '../../composables/useModelState'
import ModelsTab from '../../components/models/ModelsTab.vue'

const {
  cameras,
  loadCameras,
  camerasLoading,
  camerasLoaded,
  camerasError,
} = useModelState()
const route = useRoute()
const focusVersionId = computed(() => (
  typeof route.query.version_id === 'string' ? route.query.version_id : ''
))

onMounted(async () => {
  await loadCameras()
})
</script>

<template>
  <ModelsTab
    :cameras="cameras"
    :cameras-loading="camerasLoading"
    :cameras-loaded="camerasLoaded"
    :cameras-error="camerasError"
    :focus-version-id="focusVersionId"
    @retry-cameras="loadCameras"
  />
</template>
