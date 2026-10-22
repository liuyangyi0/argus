<script setup lang="ts">
import { onMounted } from 'vue'
import { useModelState } from '../../composables/useModelState'
import LabelingTab from '../../components/models/LabelingTab.vue'
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
    <LabelingTab :cameras="cameras" />
  </ModelCameraState>
</template>
