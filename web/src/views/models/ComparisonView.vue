<script setup lang="ts">
import { onMounted } from 'vue'
import { useModelState } from '../../composables/useModelState'
import ModelComparison from '../../components/models/ModelComparison.vue'
import ModelCameraState from '../../components/models/ModelCameraState.vue'

const { cameras, loadCameras, camerasLoading, camerasLoaded, camerasError } = useModelState()

onMounted(async () => {
  await loadCameras()
})
</script>

<template>
  <ModelCameraState
    :loading="camerasLoading"
    :loaded="camerasLoaded"
    :error="camerasError"
    :camera-count="cameras.length"
    @retry="loadCameras"
  >
    <ModelComparison :cameras="cameras" />
  </ModelCameraState>
</template>
