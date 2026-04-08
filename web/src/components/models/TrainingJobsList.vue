<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  Card, Table, Button, Tag, Space, Select,
  Popconfirm, message, Drawer, Descriptions, Typography, Empty,
} from 'ant-design-vue'
import {
  CheckOutlined, CloseOutlined, ReloadOutlined,
  ThunderboltOutlined, ClockCircleOutlined,
} from '@ant-design/icons-vue'
import {
  getTrainingJobs, getTrainingJob, confirmTrainingJob, rejectTrainingJob,
} from '../../api'
import { TRAINING_STATUS_MAP, JOB_TYPE_LABELS, TRIGGER_LABELS } from '../../composables/useModelState'

const props = defineProps<{
  cameras: any[]
}>()

const emit = defineEmits<{
  'update:pendingCount': [count: number]
}>()

const jobs = ref<any[]>([])
const loading = ref(false)
const detailDrawer = ref(false)
const detailJob = ref<any>(null)

const filterStatus = ref<string | undefined>(undefined)
const filterJobType = ref<string | undefined>(undefined)

async function loadJobs() {
  loading.value = true
  try {
    const params: Record<string, any> = {}
    if (filterStatus.value) params.status = filterStatus.value
    if (filterJobType.value) params.job_type = filterJobType.value
    const res = await getTrainingJobs(params)
    jobs.value = res.jobs || []
    emit('update:pendingCount', res.pending_count || 0)
  } catch (e) {
    console.error(e)
  } finally {
    loading.value = false
  }
}

async function showDetail(jobId: string) {
  try {
    const res = await getTrainingJob(jobId)
    detailJob.value = res
    detailDrawer.value = true
  } catch (e) {
    message.error('加载详情失败')
  }
}

async function handleConfirm(jobId: string) {
  try {
    await confirmTrainingJob(jobId, { confirmed_by: 'operator' })
    message.success('任务已确认，进入队列')
    loadJobs()
  } catch (e: any) {
    message.error(e.response?.data?.error || '确认失败')
  }
}

async function handleReject(jobId: string) {
  try {
    await rejectTrainingJob(jobId, { rejected_by: 'operator' })
    message.info('任务已拒绝')
    loadJobs()
  } catch (e: any) {
    message.error(e.response?.data?.error || '拒绝失败')
  }
}

const jobColumns = [
  { title: '任务ID', dataIndex: 'job_id', key: 'job_id', width: 120, ellipsis: true },
  { title: '类型', dataIndex: 'job_type', key: 'job_type', width: 110 },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id', width: 100 },
  { title: '触发方式', dataIndex: 'trigger_type', key: 'trigger_type', width: 100 },
  { title: '状态', dataIndex: 'status', key: 'status', width: 100 },
  { title: '骨干版本', dataIndex: 'base_model_version', key: 'base_model_version', width: 130, ellipsis: true },
  { title: '创建时间', dataIndex: 'created_at', key: 'created_at', width: 170 },
  { title: '耗时(秒)', dataIndex: 'duration_seconds', key: 'duration_seconds', width: 90 },
  { title: '操作', key: 'actions', width: 200, fixed: 'right' as const },
]

defineExpose({ loadJobs })

onMounted(loadJobs)
</script>

<template>
  <div>
    <Card :bordered="false" style="margin-bottom: 16px">
      <Space>
        <Select
          v-model:value="filterStatus"
          placeholder="状态筛选"
          allow-clear
          style="width: 150px"
          @change="loadJobs"
        >
          <Select.Option value="pending_confirmation">待确认</Select.Option>
          <Select.Option value="queued">排队中</Select.Option>
          <Select.Option value="running">训练中</Select.Option>
          <Select.Option value="complete">完成</Select.Option>
          <Select.Option value="failed">失败</Select.Option>
          <Select.Option value="rejected">已拒绝</Select.Option>
        </Select>
        <Select
          v-model:value="filterJobType"
          placeholder="类型筛选"
          allow-clear
          style="width: 150px"
          @change="loadJobs"
        >
          <Select.Option value="ssl_backbone">SSL 骨干</Select.Option>
          <Select.Option value="anomaly_head">异常检测头</Select.Option>
        </Select>
        <Button @click="loadJobs">
          <template #icon><ReloadOutlined /></template>
          刷新
        </Button>
      </Space>
    </Card>

    <Card :bordered="false">
      <Table
        :columns="jobColumns"
        :data-source="jobs"
        :loading="loading"
        row-key="job_id"
        size="small"
        :scroll="{ x: 1200 }"
        :pagination="{ pageSize: 20, showSizeChanger: true, showTotal: (t: number) => `共 ${t} 条` }"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'job_type'">
            <Tag :color="record.job_type === 'ssl_backbone' ? 'purple' : 'blue'">
              {{ JOB_TYPE_LABELS[record.job_type] || record.job_type }}
            </Tag>
          </template>
          <template v-else-if="column.key === 'trigger_type'">
            <Tag>
              <template #icon>
                <ThunderboltOutlined v-if="record.trigger_type === 'drift_suggested'" />
                <ClockCircleOutlined v-else-if="record.trigger_type === 'scheduled'" />
              </template>
              {{ TRIGGER_LABELS[record.trigger_type] || record.trigger_type }}
            </Tag>
          </template>
          <template v-else-if="column.key === 'status'">
            <Tag :color="(TRAINING_STATUS_MAP[record.status] || {}).color">
              {{ (TRAINING_STATUS_MAP[record.status] || {}).text || record.status }}
            </Tag>
          </template>
          <template v-else-if="column.key === 'camera_id'">
            {{ record.camera_id || '全部' }}
          </template>
          <template v-else-if="column.key === 'created_at'">
            {{ record.created_at?.replace('T', ' ').slice(0, 19) }}
          </template>
          <template v-else-if="column.key === 'duration_seconds'">
            {{ record.duration_seconds != null ? record.duration_seconds.toFixed(1) : '-' }}
          </template>
          <template v-else-if="column.key === 'actions'">
            <Space>
              <Button size="small" @click="showDetail(record.job_id)">详情</Button>
              <template v-if="record.status === 'pending_confirmation'">
                <Popconfirm title="确认启动此训练任务？" @confirm="handleConfirm(record.job_id)">
                  <Button size="small" type="primary">
                    <template #icon><CheckOutlined /></template>
                    确认
                  </Button>
                </Popconfirm>
                <Popconfirm title="确认拒绝此训练任务？" @confirm="handleReject(record.job_id)">
                  <Button size="small" danger>
                    <template #icon><CloseOutlined /></template>
                    拒绝
                  </Button>
                </Popconfirm>
              </template>
            </Space>
          </template>
        </template>
      </Table>
    </Card>

    <!-- Detail drawer -->
    <Drawer
      v-model:open="detailDrawer"
      :title="`训练任务 ${detailJob?.job_id || ''}`"
      width="600"
    >
      <template v-if="detailJob">
        <Descriptions :column="1" bordered size="small">
          <Descriptions.Item label="任务ID">{{ detailJob.job_id }}</Descriptions.Item>
          <Descriptions.Item label="类型">
            <Tag :color="detailJob.job_type === 'ssl_backbone' ? 'purple' : 'blue'">
              {{ JOB_TYPE_LABELS[detailJob.job_type] || detailJob.job_type }}
            </Tag>
          </Descriptions.Item>
          <Descriptions.Item label="摄像头">{{ detailJob.camera_id || '全部' }}</Descriptions.Item>
          <Descriptions.Item label="区域">{{ detailJob.zone_id }}</Descriptions.Item>
          <Descriptions.Item label="模型类型">{{ detailJob.model_type || '-' }}</Descriptions.Item>
          <Descriptions.Item label="状态">
            <Tag :color="(TRAINING_STATUS_MAP[detailJob.status] || {}).color">
              {{ (TRAINING_STATUS_MAP[detailJob.status] || {}).text || detailJob.status }}
            </Tag>
          </Descriptions.Item>
          <Descriptions.Item label="触发方式">
            {{ TRIGGER_LABELS[detailJob.trigger_type] || detailJob.trigger_type }}
          </Descriptions.Item>
          <Descriptions.Item label="触发者">{{ detailJob.triggered_by || '-' }}</Descriptions.Item>
          <Descriptions.Item label="确认者">{{ detailJob.confirmed_by || '-' }}</Descriptions.Item>
          <Descriptions.Item label="确认时间">{{ detailJob.confirmed_at || '-' }}</Descriptions.Item>
          <Descriptions.Item label="骨干版本">{{ detailJob.base_model_version || '预训练' }}</Descriptions.Item>
          <Descriptions.Item label="数据集版本">{{ detailJob.dataset_version || '-' }}</Descriptions.Item>
          <Descriptions.Item label="模型版本">{{ detailJob.model_version_id || '-' }}</Descriptions.Item>
          <Descriptions.Item label="产物路径">{{ detailJob.artifacts_path || '-' }}</Descriptions.Item>
          <Descriptions.Item label="创建时间">{{ detailJob.created_at }}</Descriptions.Item>
          <Descriptions.Item label="开始时间">{{ detailJob.started_at || '-' }}</Descriptions.Item>
          <Descriptions.Item label="完成时间">{{ detailJob.completed_at || '-' }}</Descriptions.Item>
          <Descriptions.Item label="耗时">
            {{ detailJob.duration_seconds != null ? `${detailJob.duration_seconds.toFixed(1)}s` : '-' }}
          </Descriptions.Item>
          <Descriptions.Item v-if="detailJob.error" label="错误">
            <span style="color: #ff4d4f">{{ detailJob.error }}</span>
          </Descriptions.Item>
        </Descriptions>

        <template v-if="detailJob.validation_report && typeof detailJob.validation_report === 'object'">
          <Typography.Title :level="5" style="margin-top: 24px">验证报告</Typography.Title>
          <Descriptions :column="1" bordered size="small">
            <Descriptions.Item label="全部通过">
              <Tag :color="detailJob.validation_report.all_passed ? 'green' : 'red'">
                {{ detailJob.validation_report.all_passed ? '是' : '否' }}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item v-if="detailJob.validation_report.auroc" label="AUROC">
              {{ detailJob.validation_report.auroc?.toFixed(4) }}
            </Descriptions.Item>
            <Descriptions.Item v-if="detailJob.validation_report.recall" label="合成异常 Recall">
              {{ detailJob.validation_report.recall?.toFixed(4) }}
            </Descriptions.Item>
            <Descriptions.Item v-if="detailJob.validation_report.replay" label="历史回放检测率">
              {{ detailJob.validation_report.replay?.new_detection_rate?.toFixed(4) || '-' }}
            </Descriptions.Item>
          </Descriptions>
        </template>

        <template v-if="detailJob.hyperparameters && typeof detailJob.hyperparameters === 'object'">
          <Typography.Title :level="5" style="margin-top: 24px">超参数</Typography.Title>
          <pre style="background: #1a1a2e; padding: 12px; border-radius: 6px; font-size: 12px; overflow-x: auto">{{ JSON.stringify(detailJob.hyperparameters, null, 2) }}</pre>
        </template>

        <template v-if="detailJob.metrics && typeof detailJob.metrics === 'object'">
          <Typography.Title :level="5" style="margin-top: 24px">训练指标</Typography.Title>
          <pre style="background: #1a1a2e; padding: 12px; border-radius: 6px; font-size: 12px; overflow-x: auto">{{ JSON.stringify(detailJob.metrics, null, 2) }}</pre>
        </template>
      </template>
    </Drawer>
  </div>
</template>
