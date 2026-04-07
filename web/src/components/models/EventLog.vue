<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Card, Table, Tag, Space, Collapse } from 'ant-design-vue'
import { HistoryOutlined } from '@ant-design/icons-vue'
import { getVersionEvents } from '../../api'
import { STAGE_MAP } from '../../composables/useModelState'

const versionEvents = ref<any[]>([])
const eventsLoading = ref(false)

async function loadVersionEvents() {
  eventsLoading.value = true
  try {
    const res = await getVersionEvents({ limit: 50 })
    versionEvents.value = res.data.events || []
  } catch (e) {
    console.error('Failed to load version events', e)
  } finally {
    eventsLoading.value = false
  }
}

const eventColumns = [
  { title: '时间', dataIndex: 'timestamp', key: 'timestamp', width: 160 },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '转换', key: 'transition', width: 200 },
  { title: '操作人', dataIndex: 'triggered_by', key: 'triggered_by' },
  { title: '原因', dataIndex: 'reason', key: 'reason', ellipsis: true },
]

defineExpose({ loadVersionEvents })

onMounted(loadVersionEvents)
</script>

<template>
  <Collapse ghost>
    <Collapse.Panel key="events">
      <template #header>
        <Space>
          <HistoryOutlined />
          <span>版本事件日志</span>
        </Space>
      </template>
      <Table
        :columns="eventColumns"
        :data-source="versionEvents"
        :loading="eventsLoading"
        :pagination="{ pageSize: 10, showSizeChanger: false }"
        row-key="id"
        size="small"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'timestamp'">
            {{ record.timestamp ? record.timestamp.replace('T', ' ').substring(0, 19) : '-' }}
          </template>
          <template v-if="column.key === 'transition'">
            <Space>
              <Tag :color="(STAGE_MAP[record.from_stage] || { color: 'default' }).color">
                {{ (STAGE_MAP[record.from_stage] || { text: record.from_stage || '?' }).text }}
              </Tag>
              <span>→</span>
              <Tag :color="(STAGE_MAP[record.to_stage] || { color: 'default' }).color">
                {{ (STAGE_MAP[record.to_stage] || { text: record.to_stage }).text }}
              </Tag>
            </Space>
          </template>
        </template>
      </Table>
    </Collapse.Panel>
  </Collapse>
</template>
