<script setup lang="ts">
import { ref } from 'vue'
import { Tabs } from 'ant-design-vue'
import TrainingCreate from './TrainingCreate.vue'
import TrainingJobsList from './TrainingJobsList.vue'
import TrainingHistory from './TrainingHistory.vue'
import BackboneList from './BackboneList.vue'

const props = defineProps<{
  cameras: any[]
}>()

const activeSubTab = ref('jobs')
const pendingCount = ref(0)
const jobsListRef = ref<InstanceType<typeof TrainingJobsList> | null>(null)

function onSubTabChange(key: string | number) {
  activeSubTab.value = String(key)
}

function onRefreshJobs() {
  jobsListRef.value?.loadJobs()
}
</script>

<template>
  <Tabs :activeKey="activeSubTab" @change="onSubTabChange" type="card" size="small">
    <Tabs.TabPane key="jobs" tab="训练任务">
      <TrainingCreate
        :cameras="cameras"
        :pending-count="pendingCount"
        @refresh="onRefreshJobs"
      />
      <div style="margin-top: 16px">
        <TrainingJobsList
          ref="jobsListRef"
          :cameras="cameras"
          @update:pending-count="pendingCount = $event"
        />
      </div>
    </Tabs.TabPane>

    <Tabs.TabPane key="history" tab="训练历史">
      <TrainingHistory />
    </Tabs.TabPane>

    <Tabs.TabPane key="backbones" tab="骨干模型">
      <BackboneList />
    </Tabs.TabPane>
  </Tabs>
</template>
