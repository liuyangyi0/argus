<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Card, Table, Space, Select, Tag } from 'ant-design-vue'
import { getDegradationHistory } from '../../api'

const degradationEvents = ref<any[]>([])
const degradationLoading = ref(false)
const degradationDays = ref(7)

async function loadDegradation() {
  degradationLoading.value = true
  try {
    const res = await getDegradationHistory(degradationDays.value)
    degradationEvents.value = res || []
  } catch { /* silent */ }
  finally { degradationLoading.value = false }
}

onMounted(() => {
  loadDegradation()
})

const degradationColumns = [
  { title: '时间', key: 'started_at', width: 180 },
  { title: '级别', key: 'level', width: 80 },
  { title: '类别', dataIndex: 'category', key: 'category', width: 140 },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id', width: 100 },
  { title: '标题', dataIndex: 'title', key: 'title', ellipsis: true },
  { title: '持续', key: 'duration', width: 100 },
  { title: '状态', key: 'status', width: 80 },
]

const degradationLevelColors: Record<string, string> = {
  info: 'blue', warning: 'gold', moderate: 'orange', severe: 'red',
}
const degradationLevelLabels: Record<string, string> = {
  info: '提示', warning: '警告', moderate: '中度', severe: '严重',
}
</script>

<template>
  <Card title="降级事件历史">
    <Space style="margin-bottom: 16px">
      <Select v-model:value="degradationDays" style="width: 120px" @change="loadDegradation">
        <Select.Option :value="1">最近 1 天</Select.Option>
        <Select.Option :value="7">最近 7 天</Select.Option>
        <Select.Option :value="30">最近 30 天</Select.Option>
        <Select.Option :value="90">最近 90 天</Select.Option>
      </Select>
    </Space>
    <Table
      :columns="degradationColumns"
      :data-source="degradationEvents"
      :loading="degradationLoading"
      :pagination="{ pageSize: 20 }"
      row-key="event_id"
      size="small"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'started_at'">
          {{ new Date(record.started_at * 1000).toLocaleString('zh-CN') }}
        </template>
        <template v-if="column.key === 'level'">
          <Tag :color="degradationLevelColors[record.level]">
            {{ degradationLevelLabels[record.level] || record.level }}
          </Tag>
        </template>
        <template v-if="column.key === 'duration'">
          {{
            record.resolved_at
              ? Math.round((record.resolved_at - record.started_at) / 60) + ' 分钟'
              : '进行中'
          }}
        </template>
        <template v-if="column.key === 'status'">
          <Tag v-if="record.resolved_at" color="green">已恢复</Tag>
          <Tag v-else color="red">活跃</Tag>
        </template>
      </template>
    </Table>
  </Card>
</template>
