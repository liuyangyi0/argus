<script setup lang="ts">
import { onMounted } from 'vue'
import { useModelState } from '../../composables/useModelState'
import ThresholdPreview from '../../components/models/ThresholdPreview.vue'

const { cameras, loadCameras } = useModelState()

onMounted(async () => {
  await loadCameras()
})
</script>

<template>
  <ThresholdPreview :cameras="cameras" />
</template>
