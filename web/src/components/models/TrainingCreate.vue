<script setup lang="ts">
import { ref } from 'vue'
import {
  Card, Button, Form, Select, Input, Modal, Space,
  message, Badge,
} from 'ant-design-vue'
import { PlusOutlined } from '@ant-design/icons-vue'
import { createTrainingJob } from '../../api'
import { MODEL_TYPES } from '../../composables/useModelState'

const props = defineProps<{
  cameras: any[]
  pendingCount: number
}>()

const emit = defineEmits<{
  refresh: []
}>()

const createModalVisible = ref(false)
const createForm = ref({
  job_type: 'anomaly_head',
  camera_id: undefined as string | undefined,
  model_type: 'patchcore',
  zone_id: 'default',
})

async function handleCreate() {
  try {
    await createTrainingJob(createForm.value)
    message.success('训练任务已创建，等待确认')
    createModalVisible.value = false
    createForm.value = { job_type: 'anomaly_head', camera_id: undefined, model_type: 'patchcore', zone_id: 'default' }
    emit('refresh')
  } catch (e: any) {
    message.error(e.response?.data?.error || '创建失败')
  }
}
</script>

<template>
  <div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px">
      <Space>
        <span style="font-weight: 500; font-size: 15px">新建训练任务</span>
        <Badge v-if="pendingCount > 0" :count="pendingCount" :offset="[8, -4]" />
      </Space>
      <Button type="primary" @click="createModalVisible = true">
        <template #icon><PlusOutlined /></template>
        新建训练任务
      </Button>
    </div>

    <Card :bordered="false">
      <p style="color: #8890a0; margin-bottom: 16px">
        创建训练任务后需要操作员确认才会开始执行。支持异常检测头（按摄像头）和 SSL 骨干微调（全厂共享）两种类型。
      </p>
    </Card>

    <Modal
      v-model:open="createModalVisible"
      title="新建训练任务"
      @ok="handleCreate"
      ok-text="创建"
      cancel-text="取消"
    >
      <Form layout="vertical" style="margin-top: 16px">
        <Form.Item label="任务类型">
          <Select v-model:value="createForm.job_type">
            <Select.Option value="anomaly_head">异常检测头 (按摄像头)</Select.Option>
            <Select.Option value="ssl_backbone">SSL 骨干微调 (全厂共享)</Select.Option>
          </Select>
        </Form.Item>
        <Form.Item v-if="createForm.job_type === 'anomaly_head'" label="摄像头">
          <Select v-model:value="createForm.camera_id" placeholder="选择摄像头">
            <Select.Option v-for="c in cameras" :key="c.camera_id" :value="c.camera_id">
              {{ c.name || c.camera_id }}
            </Select.Option>
          </Select>
        </Form.Item>
        <Form.Item v-if="createForm.job_type === 'anomaly_head'" label="模型类型">
          <Select v-model:value="createForm.model_type">
            <Select.Option v-for="mt in MODEL_TYPES" :key="mt.value" :value="mt.value">
              {{ mt.label }}
            </Select.Option>
          </Select>
        </Form.Item>
        <Form.Item label="区域">
          <Input v-model:value="createForm.zone_id" placeholder="default" />
        </Form.Item>
      </Form>
    </Modal>
  </div>
</template>
