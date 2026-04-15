<script setup lang="ts">
import { storeToRefs } from 'pinia'
import {
  Table, Card, Button, Space, Tag, Tooltip, Modal, message,
} from 'ant-design-vue'
import {
  ReloadOutlined, PlayCircleOutlined, ExperimentOutlined,
  SafetyCertificateOutlined, MergeCellsOutlined, DeleteOutlined,
  TeamOutlined,
} from '@ant-design/icons-vue'

import { useBaselineStore } from '../../stores/useBaselineStore'
import { BASELINE_STATE_MAP } from '../../composables/useModelState'

defineOptions({ name: 'BaselineTable' })

const emit = defineEmits<{
  (e: 'openCapture'): void
  (e: 'openAdvancedCapture'): void
}>()

const store = useBaselineStore()
const {
  baselines,
  baselinesLoading,
  cameraGroups,
  groupsLoading,
  optimizingBaseline,
  mergingFP,
  mergingGroup,
  deletingBaseline,
} = storeToRefs(store)

const baselineColumns = [
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '版本', dataIndex: 'version', key: 'version' },
  { title: '图片数量', dataIndex: 'image_count', key: 'image_count' },
  { title: '采集场景', dataIndex: 'session_label', key: 'session_label' },
  { title: '生命周期', key: 'state', width: 100 },
  { title: '操作', key: 'action', width: 360 },
]

const groupColumns = [
  { title: '组 ID', dataIndex: 'group_id', key: 'group_id' },
  { title: '名称', dataIndex: 'name', key: 'name' },
  { title: '成员摄像头', key: 'camera_ids' },
  { title: '图片数', dataIndex: 'image_count', key: 'image_count', width: 80 },
  { title: '当前版本', dataIndex: 'current_version', key: 'current_version', width: 100 },
  { title: '操作', key: 'action', width: 120 },
]

async function handleOptimize(record: any) {
  store.setOptimizing(`${record.camera_id}-${record.version}`)
  try {
    const preview = await store.previewOptimize(record.camera_id)
    const { total, keep, move } = preview
    store.setOptimizing(null)
    Modal.confirm({
      title: '确认优化',
      content: `共 ${total} 张图片，将保留 ${keep} 张，移除 ${move} 张。确认执行优化？`,
      okText: '确认优化',
      cancelText: '取消',
      async onOk() {
        try {
          const res = await store.executeOptimize(record.camera_id, record.version)
          message.success(`优化完成: 保留 ${res.selected} 张, 移除 ${res.moved} 张`)
        } catch (e: any) {
          message.error(e.response?.data?.error || '优化失败')
        }
      },
    })
  } catch (e: any) {
    message.error(e.response?.data?.error || '优化预览失败')
    store.setOptimizing(null)
  }
}

function handleMergeFP(cameraId: string) {
  Modal.confirm({
    title: '合并误报到基线',
    content: `将 ${cameraId} 的误报候选池帧合并到新基线版本（Draft 状态，需审核后才能训练）。`,
    okText: '执行合并',
    cancelText: '取消',
    async onOk() {
      try {
        const res = await store.mergeFalsePositives(cameraId)
        message.success(`误报合并完成: ${res.version}, 新增 ${res.fp_included} 张误报帧`)
      } catch (e: any) {
        message.error(e.response?.data?.error || '合并失败')
      }
    },
  })
}

function handleMergeGroup(groupId: string) {
  Modal.confirm({
    title: '合并摄像头组基线',
    content: `将组 ${groupId} 内所有成员摄像头的基线合并为一个组版本（Draft 状态）。`,
    okText: '执行合并',
    cancelText: '取消',
    async onOk() {
      try {
        const res = await store.mergeGroupBaseline(groupId)
        message.success(`组基线合并完成: ${res.version}, ${res.image_count} 张图片`)
      } catch (e: any) {
        message.error(e.response?.data?.error || '合并失败')
      }
    },
  })
}

function handleDeleteBaseline(record: any) {
  Modal.confirm({
    title: '删除基线版本',
    content: `确定删除 ${record.camera_id} / ${record.version}？该操作会删除磁盘中的整套基线数据。`,
    okText: '确认删除',
    okType: 'danger',
    cancelText: '取消',
    async onOk() {
      try {
        await store.deleteBaselineVersion(record.camera_id, record.version)
        message.success(`${record.version} 已删除`)
      } catch (e: any) {
        message.error(e.response?.data?.error || '删除失败')
      }
    },
  })
}
</script>

<template>
  <Card>
    <template #title>
      <div style="display: flex; justify-content: space-between; align-items: center">
        <span>基线数据</span>
        <Space>
          <Button @click="store.loadBaselines">
            <template #icon><ReloadOutlined /></template>
            刷新
          </Button>
          <Button type="primary" @click="emit('openCapture')">
            <template #icon><PlayCircleOutlined /></template>
            快速采集
          </Button>
          <Button @click="emit('openAdvancedCapture')">
            <template #icon><ExperimentOutlined /></template>
            高级采集
          </Button>
        </Space>
      </div>
    </template>
    <p style="color: #8890a0; margin-bottom: 16px">
      从在线摄像头采集"正常"场景的参考图片，用于训练异常检测模型。
    </p>
    <Table
      :columns="baselineColumns"
      :data-source="baselines"
      :loading="baselinesLoading"
      :pagination="false"
      :row-key="(record: any) => `${record.camera_id}-${record.version}`"
      size="small"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'image_count'">
          <span>{{ record.image_count }} 张</span>
        </template>
        <template v-if="column.key === 'session_label'">
          <Tag v-if="record.session_label">{{ record.session_label }}</Tag>
          <span v-else style="color: #666">-</span>
        </template>
        <template v-if="column.key === 'state'">
          <Tag
            v-if="record.state"
            :color="(BASELINE_STATE_MAP[record.state] || {}).color || 'default'"
          >
            {{ (BASELINE_STATE_MAP[record.state] || {}).text || record.state }}
          </Tag>
          <Tag v-else color="green">就绪</Tag>
        </template>
        <template v-if="column.key === 'action'">
          <Space>
            <Tooltip title="多样性优选">
              <Button
                size="small"
                :loading="optimizingBaseline === `${record.camera_id}-${record.version}`"
                :disabled="record.image_count < 30"
                @click="handleOptimize(record)"
              >
                优化
              </Button>
            </Tooltip>
            <Button size="small" @click="store.openVersionDrawer(record.camera_id)">
              <SafetyCertificateOutlined />
              版本管理
            </Button>
            <Tooltip title="将误报候选帧合并到新基线版本">
              <Button size="small" :loading="mergingFP === record.camera_id" @click="handleMergeFP(record.camera_id)">
                <MergeCellsOutlined />
                合并误报
              </Button>
            </Tooltip>
            <Button
              size="small"
              danger
              :loading="deletingBaseline === `${record.camera_id}-${record.version}`"
              :disabled="record.state === 'active'"
              @click="handleDeleteBaseline(record)"
            >
              <DeleteOutlined />
              删除
            </Button>
          </Space>
        </template>
      </template>
    </Table>
    <div v-if="baselines.length === 0 && !baselinesLoading" style="text-align: center; padding: 32px; color: #666">
      暂无基线数据，请先采集基线图片
    </div>
  </Card>

  <Card v-if="cameraGroups.length > 0" size="small" style="margin-top: 16px">
    <template #title>
      <Space>
        <TeamOutlined />
        <span>摄像头组共享基线</span>
      </Space>
    </template>
    <Table
      :columns="groupColumns"
      :data-source="cameraGroups"
      :loading="groupsLoading"
      :pagination="false"
      row-key="group_id"
      size="small"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'camera_ids'">
          <Tag v-for="cam in record.camera_ids" :key="cam" style="margin: 2px">{{ cam }}</Tag>
        </template>
        <template v-if="column.key === 'action'">
          <Button
            size="small"
            type="primary"
            :loading="mergingGroup === record.group_id"
            @click="handleMergeGroup(record.group_id)"
          >
            <MergeCellsOutlined />
            合并基线
          </Button>
        </template>
      </template>
    </Table>
  </Card>
</template>
