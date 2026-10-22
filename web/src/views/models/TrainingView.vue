<script setup lang="ts">
import { onMounted } from 'vue'
import { useModelState } from '../../composables/useModelState'
import TrainingTab from '../../components/models/TrainingTab.vue'
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
    <TrainingTab :cameras="cameras" />
  </ModelCameraState>
</template>
