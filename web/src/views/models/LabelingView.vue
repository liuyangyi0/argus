<script setup lang="ts">
import { onMounted } from 'vue'
import { useModelState } from '../../composables/useModelState'
import LabelingTab from '../../components/models/LabelingTab.vue'

const { cameras, loadCameras } = useModelState()

onMounted(async () => {
  await loadCameras()
})
</script>

<template>
  <LabelingTab :cameras="cameras" />
</template>
