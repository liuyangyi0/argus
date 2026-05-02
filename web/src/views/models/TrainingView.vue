<script setup lang="ts">
import { onMounted } from 'vue'
import { useModelState } from '../../composables/useModelState'
import TrainingTab from '../../components/models/TrainingTab.vue'

const { cameras, loadCameras } = useModelState()

onMounted(async () => {
  await loadCameras()
})
</script>

<template>
  <TrainingTab :cameras="cameras" />
</template>
