<script setup lang="ts">
import { ref } from 'vue'
import {
  Card, Table, Button, Tag, Space, Modal, Tooltip,
  message,
} from 'ant-design-vue'
import {
  ReloadOutlined, CheckOutlined, RollbackOutlined, DeleteOutlined,
} from '@ant-design/icons-vue'
import { activateModel, rollbackModel, deleteModel } from '../../api'

const props = defineProps<{
  models: any[]
}>()

const emit = defineEmits<{
  changed: []
}>()

const activatingModel = ref<string | null>(null)

function handleActivate(record: any) {
  Modal.confirm({
    title: '确认激活',
    content: `确定要激活模型版本 ${record.model_version_id} 吗？这将停用该摄像头的其他模型。`,
    okText: '确认',
    cancelText: '取消',
    async onOk() {
      activatingModel.value = record.model_version_id
      try {
        await activateModel(record.model_version_id)
        message.success(`模型 ${record.model_version_id} 已激活`)
        emit('changed')
      } catch (e: any) {
        message.error(e.response?.data?.error || '激活失败')
      } finally {
        activatingModel.value = null
      }
    },
  })
}

function handleRollback(record: any) {
  Modal.confirm({
    title: '确认回滚',
    content: `确定要回滚摄像头 ${record.camera_id} 到上一个模型版本吗？`,
    okText: '确认回滚',
    cancelText: '取消',
    okType: 'danger',
    async onOk() {
      try {
        const res = await rollbackModel(record.model_version_id)
        message.success(`已回滚到 ${res.data.activated}`)
        emit('changed')
      } catch (e: any) {
        message.error(e.response?.data?.error || '回滚失败')
      }
    },
  })
}

function handleDeleteModel(record: any) {
  Modal.confirm({
    title: '删除模型',
    content: `确定删除模型 ${record.model_version_id}？模型文件将从磁盘永久删除。`,
    okText: '确认删除',
    okType: 'danger',
    cancelText: '取消',
    async onOk() {
      try {
        await deleteModel(record.model_version_id)
        message.success('模型已删除')
        emit('changed')
      } catch (e: any) {
        message.error(e.response?.data?.error || '删除失败')
      }
    },
  })
}

const registryColumns = [
  { title: '版本 ID', dataIndex: 'model_version_id', key: 'version_id', ellipsis: true },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '模型类型', dataIndex: 'model_type', key: 'model_type' },
  { title: '模型哈希', key: 'model_hash', width: 100 },
  { title: '模型路径', dataIndex: 'model_path', key: 'model_path', ellipsis: true },
  { title: '创建时间', dataIndex: 'created_at', key: 'created_at', width: 160 },
  { title: '状态', key: 'is_active', width: 80 },
  { title: '操作', key: 'action', width: 160 },
]
</script>

<template>
  <Card style="margin-bottom: 16px">
    <template #title>
      <div style="display: flex; justify-content: space-between; align-items: center">
        <span>模型版本</span>
        <Button @click="$emit('changed')">
          <template #icon><ReloadOutlined /></template>
          刷新
        </Button>
      </div>
    </template>
    <p style="color: #8890a0; margin-bottom: 16px">
      管理已注册的模型版本，支持激活和回滚操作。
    </p>
    <Table
      :columns="registryColumns"
      :data-source="models"
      :pagination="{ pageSize: 10, showSizeChanger: false }"
      row-key="model_version_id"
      size="small"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'version_id'">
          <span style="font-family: monospace; font-size: 12px">{{ record.model_version_id }}</span>
        </template>
        <template v-if="column.key === 'model_hash'">
          <span style="font-family: monospace; font-size: 12px">{{ record.model_hash?.substring(0, 8) }}</span>
        </template>
        <template v-if="column.key === 'model_path'">
          <span style="font-family: monospace; font-size: 12px">{{ record.model_path || '-' }}</span>
        </template>
        <template v-if="column.key === 'created_at'">
          {{ record.created_at ? record.created_at.replace('T', ' ').substring(0, 16) : '-' }}
        </template>
        <template v-if="column.key === 'is_active'">
          <Tag :color="record.is_active ? 'green' : 'default'">
            {{ record.is_active ? '已激活' : '未激活' }}
          </Tag>
        </template>
        <template v-if="column.key === 'action'">
          <Space>
            <Tooltip title="激活此版本">
              <Button
                size="small"
                type="primary"
                :disabled="record.is_active"
                :loading="activatingModel === record.model_version_id"
                @click="handleActivate(record)"
              >
                <template #icon><CheckOutlined /></template>
                激活
              </Button>
            </Tooltip>
            <Tooltip title="回滚到上一版本">
              <Button
                size="small"
                danger
                :disabled="!record.is_active"
                @click="handleRollback(record)"
              >
                <template #icon><RollbackOutlined /></template>
                回滚
              </Button>
            </Tooltip>
            <Tooltip title="删除此版本">
              <Button
                size="small"
                danger
                :disabled="record.is_active"
                @click="handleDeleteModel(record)"
              >
                <template #icon><DeleteOutlined /></template>
              </Button>
            </Tooltip>
          </Space>
        </template>
      </template>
    </Table>
    <div v-if="models.length === 0" style="text-align: center; padding: 32px; color: #666">
      暂无注册的模型版本
    </div>
  </Card>
</template>
