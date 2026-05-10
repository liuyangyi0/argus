<script setup lang="ts">
import { onMounted } from 'vue'
import { useModelState } from '../../composables/useModelState'
import ModelsTab from '../../components/models/ModelsTab.vue'

const { cameras, loadCameras } = useModelState()

onMounted(async () => {
  await loadCameras()
})
</script>

<template>
  <ModelsTab :cameras="cameras" />
</template>
