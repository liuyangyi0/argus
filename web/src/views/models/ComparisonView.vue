<script setup lang="ts">
import { onMounted } from 'vue'
import { useModelState } from '../../composables/useModelState'
import ModelComparison from '../../components/models/ModelComparison.vue'

const { cameras, loadCameras } = useModelState()

onMounted(async () => {
  await loadCameras()
})
</script>

<template>
  <ModelComparison :cameras="cameras" />
</template>
