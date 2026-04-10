<script setup lang="ts">
import { ref } from 'vue'
import {
  Form, Select, Button, Table, Tag, Collapse,
  message,
} from 'ant-design-vue'
import { ExperimentOutlined } from '@ant-design/icons-vue'
import { batchInference } from '../../api'

const props = defineProps<{
  cameras: any[]
}>()

const batchCameraId = ref('')
const batchImagePaths = ref('')
const batchRunning = ref(false)
const batchResults = ref<any[]>([])

async function handleBatchInference() {
  const paths = batchImagePaths.value.split('\n').map(p => p.trim()).filter(Boolean)
  if (!batchCameraId.value) {
    message.error('请选择摄像头')
    return
  }
  if (paths.length === 0) {
    message.error('请输入图片路径')
    return
  }
  batchRunning.value = true
  batchResults.value = []
  try {
    const res = await batchInference(batchCameraId.value, paths)
    batchResults.value = res.results || []
    message.success(`完成推理: ${res.scored}/${res.total} 张图片`)
  } catch (e: any) {
    message.error(e.response?.data?.error || '批量推理失败')
  } finally {
    batchRunning.value = false
  }
}
</script>

<template>
  <Collapse ghost>
    <Collapse.Panel key="batch">
      <template #header>
        <span>批量推理工具</span>
      </template>
      <p style="color: #999; margin-bottom: 16px">输入图片路径（每行一个），使用摄像头的活跃模型进行异常检测评分。</p>
      <Form layout="vertical">
        <Form.Item label="摄像头">
          <Select
            v-model:value="batchCameraId"
            placeholder="选择摄像头"
            style="width: 300px"
            :options="cameras.map(c => ({ value: c.camera_id, label: `${c.camera_id} - ${c.name || ''}` }))"
          />
        </Form.Item>
        <Form.Item label="图片路径（每行一个，最多100张）">
          <textarea
            v-model="batchImagePaths"
            rows="5"
            style="width: 100%; font-family: monospace; padding: 8px; border: 1px solid #d9d9d9; border-radius: 6px"
            placeholder="/path/to/image1.jpg&#10;/path/to/image2.jpg"
          />
        </Form.Item>
        <Form.Item>
          <Button type="primary" :loading="batchRunning" @click="handleBatchInference">
            <template #icon><ExperimentOutlined /></template>
            开始推理
          </Button>
        </Form.Item>
      </Form>
      <Table
        v-if="batchResults.length > 0"
        :data-source="batchResults"
        :columns="[
          { title: '文件路径', dataIndex: 'path', key: 'path', ellipsis: true },
          { title: '分数', dataIndex: 'score', key: 'score', width: 100 },
          { title: '异常', dataIndex: 'is_anomalous', key: 'is_anomalous', width: 80 },
          { title: '错误', dataIndex: 'error', key: 'error', width: 200 },
        ]"
        :pagination="{ pageSize: 20 }"
        size="small"
        row-key="path"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'is_anomalous'">
            <Tag v-if="record.is_anomalous === true" color="red">异常</Tag>
            <Tag v-else-if="record.is_anomalous === false" color="green">正常</Tag>
            <span v-else>-</span>
          </template>
          <template v-if="column.key === 'score'">
            <span v-if="record.score !== undefined">{{ record.score.toFixed(4) }}</span>
            <span v-else>-</span>
          </template>
          <template v-if="column.key === 'error'">
            <Tag v-if="record.error" color="red">{{ record.error }}</Tag>
          </template>
        </template>
      </Table>
    </Collapse.Panel>
  </Collapse>
</template>
