<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import {
  Card, Table, Button, Tag, Space, Modal, Form, Select,
  Descriptions, message, Typography,
} from 'ant-design-vue'
import {
  HistoryOutlined, SwapOutlined, ReloadOutlined,
  CheckCircleOutlined, CloseCircleOutlined, LineChartOutlined,
} from '@ant-design/icons-vue'
import { getTrainingHistory, compareModels } from '../../api'
import { GRADE_COLORS, RECOMMENDATION_COLORS, RECOMMENDATION_TEXT } from '../../composables/useModelState'
import { extractErrorMessage } from '../../utils/error'
import type { TrainingRecord } from '../../types/api'
import MetricsChart from './MetricsChart.vue'

const trainingHistory = ref<TrainingRecord[]>([])
const historyLoading = ref(false)
const historyDetailVisible = ref(false)
const historyDetail = ref<TrainingRecord | null>(null)

const compareVisible = ref(false)
const compareForm = ref({ old_record_id: undefined as number | undefined, new_record_id: undefined as number | undefined })
const compareResult = ref<any>(null)
const comparing = ref(false)

const completedRecords = computed(() =>
  trainingHistory.value.filter(r => r.status === 'complete')
)

async function loadHistory() {
  historyLoading.value = true
  try {
    const res = await getTrainingHistory()
    trainingHistory.value = res.records || []
  } catch (e) {
    message.error('加载训练历史失败')
  } finally {
    historyLoading.value = false
  }
}

function showHistoryDetail(record: any) {
  historyDetail.value = record
  historyDetailVisible.value = true
}

async function handleCompare() {
  if (compareForm.value.old_record_id == null || compareForm.value.new_record_id == null) {
    message.warning('请选择两个训练记录进行对比')
    return
  }
  if (compareForm.value.old_record_id === compareForm.value.new_record_id) {
    message.warning('请选择不同的训练记录')
    return
  }
  comparing.value = true
  compareResult.value = null
  try {
    const res = await compareModels({
      old_record_id: compareForm.value.old_record_id,
      new_record_id: compareForm.value.new_record_id,
    })
    compareResult.value = res
  } catch (e) {
    message.error(extractErrorMessage(e, '对比失败'))
  } finally {
    comparing.value = false
  }
}

const historyColumns = [
  { title: 'ID', dataIndex: 'id', key: 'id', width: 60 },
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '模型', dataIndex: 'model_type', key: 'model_type' },
  { title: '基线数', dataIndex: 'baseline_count', key: 'baseline_count', width: 80 },
  { title: '训练/验证', key: 'split', width: 100 },
  { title: '质量', key: 'grade', width: 70 },
  { title: 'F1', key: 'f1', width: 80 },
  { title: 'AUROC', key: 'auroc', width: 80 },
  { title: '推荐阈值', key: 'threshold', width: 100 },
  { title: '状态', key: 'status', width: 80 },
  { title: '耗时', key: 'duration', width: 80 },
  { title: '时间', dataIndex: 'trained_at', key: 'trained_at', width: 160 },
  { title: '操作', key: 'action', width: 80 },
]

defineExpose({ loadHistory })

onMounted(loadHistory)
</script>

<template>
  <Card>
    <template #title>
      <div style="display: flex; justify-content: space-between; align-items: center">
        <Space>
          <HistoryOutlined />
          <span>训练历史</span>
        </Space>
        <Space>
          <Button @click="compareVisible = true" :disabled="completedRecords.length < 2">
            <template #icon><SwapOutlined /></template>
            A/B 对比
          </Button>
          <Button @click="loadHistory">
            <template #icon><ReloadOutlined /></template>
            刷新
          </Button>
        </Space>
      </div>
    </template>
    <Table
      :columns="historyColumns"
      :data-source="trainingHistory"
      :loading="historyLoading"
      :pagination="{ pageSize: 10, showSizeChanger: false }"
      row-key="id"
      size="small"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'split'">
          {{ record.train_count }}/{{ record.val_count }}
        </template>
        <template v-if="column.key === 'grade'">
          <Tag
            v-if="record.quality_grade"
            :color="GRADE_COLORS[record.quality_grade] || 'default'"
            style="font-weight: bold; font-size: 14px"
          >
            {{ record.quality_grade }}
          </Tag>
          <span v-else style="color: #666">-</span>
        </template>
        <template v-if="column.key === 'f1'">
          <Tag v-if="record.val_f1 != null" color="purple" style="font-weight: 600">
            {{ record.val_f1.toFixed(3) }}
          </Tag>
          <span v-else style="color: #999">-</span>
        </template>
        <template v-if="column.key === 'auroc'">
          <span v-if="record.val_auroc != null" :style="{ color: record.val_auroc >= 0.9 ? '#15a34a' : record.val_auroc >= 0.7 ? '#d29b1f' : '#e5484d' }">
            {{ record.val_auroc.toFixed(3) }}
          </span>
          <span v-else style="color: #999">-</span>
        </template>
        <template v-if="column.key === 'threshold'">
          {{ record.threshold_recommended != null ? record.threshold_recommended.toFixed(3) : '-' }}
        </template>
        <template v-if="column.key === 'status'">
          <Tag :color="record.status === 'complete' ? 'green' : 'red'">
            {{ record.status === 'complete' ? '完成' : '失败' }}
          </Tag>
        </template>
        <template v-if="column.key === 'duration'">
          {{ record.duration_seconds ? `${record.duration_seconds.toFixed(0)}s` : '-' }}
        </template>
        <template v-if="column.key === 'trained_at'">
          {{ record.trained_at ? record.trained_at.replace('T', ' ').substring(0, 16) : '-' }}
        </template>
        <template v-if="column.key === 'action'">
          <Button size="small" type="link" @click="showHistoryDetail(record)">详情</Button>
        </template>
      </template>
    </Table>
    <div v-if="trainingHistory.length === 0 && !historyLoading" style="text-align: center; padding: 32px; color: #666">
      暂无训练记录
    </div>
  </Card>

  <!-- History detail modal -->
  <Modal
    v-model:open="historyDetailVisible"
    title="训练详情"
    :footer="null"
    width="1000px"
  >
    <Descriptions v-if="historyDetail" bordered :column="2" size="small" style="margin-top: 16px">
      <Descriptions.Item label="摄像头">{{ historyDetail.camera_id }}</Descriptions.Item>
      <Descriptions.Item label="区域">{{ historyDetail.zone_id }}</Descriptions.Item>
      <Descriptions.Item label="模型类型">{{ historyDetail.model_type }}</Descriptions.Item>
      <Descriptions.Item label="导出格式">{{ historyDetail.export_format || '-' }}</Descriptions.Item>
      <Descriptions.Item label="基线版本">{{ historyDetail.baseline_version }}</Descriptions.Item>
      <Descriptions.Item label="基线数量">{{ historyDetail.baseline_count }}</Descriptions.Item>
      <Descriptions.Item label="训练/验证">{{ historyDetail.train_count }}/{{ historyDetail.val_count }}</Descriptions.Item>
      <Descriptions.Item label="质量评级">
        <Tag v-if="historyDetail.quality_grade" :color="GRADE_COLORS[historyDetail.quality_grade]" style="font-weight:bold">
          {{ historyDetail.quality_grade }}
        </Tag>
        <span v-else>-</span>
      </Descriptions.Item>
      <Descriptions.Item label="推荐阈值">
        {{ historyDetail.threshold_recommended != null ? historyDetail.threshold_recommended.toFixed(4) : '-' }}
      </Descriptions.Item>
      <Descriptions.Item label="状态">
        <Tag :color="historyDetail.status === 'complete' ? 'green' : 'red'">
          {{ historyDetail.status === 'complete' ? '完成' : '失败' }}
        </Tag>
      </Descriptions.Item>
      <Descriptions.Item label="耗时">{{ historyDetail.duration_seconds?.toFixed(1) }}s</Descriptions.Item>
      <Descriptions.Item label="训练时间">{{ historyDetail.trained_at?.replace('T', ' ').substring(0, 19) }}</Descriptions.Item>
      <Descriptions.Item label="验证分数均值">{{ historyDetail.val_score_mean?.toFixed(4) ?? '-' }}</Descriptions.Item>
      <Descriptions.Item label="验证分数标准差">{{ historyDetail.val_score_std?.toFixed(4) ?? '-' }}</Descriptions.Item>
      <Descriptions.Item label="验证分数最大值">{{ historyDetail.val_score_max?.toFixed(4) ?? '-' }}</Descriptions.Item>
      <Descriptions.Item label="验证分数 P95">{{ historyDetail.val_score_p95?.toFixed(4) ?? '-' }}</Descriptions.Item>
      <Descriptions.Item label="预验证通过">
        <CheckCircleOutlined v-if="historyDetail.pre_validation_passed" style="color: #15a34a" />
        <CloseCircleOutlined v-else style="color: #e5484d" />
      </Descriptions.Item>
      <Descriptions.Item label="推理延迟">{{ historyDetail.inference_latency_ms != null ? `${historyDetail.inference_latency_ms.toFixed(1)}ms` : '-' }}</Descriptions.Item>
      <Descriptions.Item label="损坏率">
        {{ historyDetail.corruption_rate != null ? `${(historyDetail.corruption_rate * 100).toFixed(1)}%` : '-' }}
      </Descriptions.Item>
      <Descriptions.Item label="近重复率">
        {{ historyDetail.near_duplicate_rate != null ? `${(historyDetail.near_duplicate_rate * 100).toFixed(1)}%` : '-' }}
      </Descriptions.Item>
      <Descriptions.Item label="亮度标准差">
        {{ historyDetail.brightness_std != null ? historyDetail.brightness_std.toFixed(2) : '-' }}
      </Descriptions.Item>
      <Descriptions.Item label="检查点有效">
        <CheckCircleOutlined v-if="historyDetail.checkpoint_valid" style="color: #15a34a" />
        <CloseCircleOutlined v-else-if="historyDetail.checkpoint_valid === false" style="color: #e5484d" />
        <span v-else>-</span>
      </Descriptions.Item>
      <Descriptions.Item label="导出有效">
        <CheckCircleOutlined v-if="historyDetail.export_valid" style="color: #15a34a" />
        <CloseCircleOutlined v-else-if="historyDetail.export_valid === false" style="color: #e5484d" />
        <span v-else>-</span>
      </Descriptions.Item>
      <Descriptions.Item label="冒烟测试">
        <CheckCircleOutlined v-if="historyDetail.smoke_test_passed" style="color: #15a34a" />
        <CloseCircleOutlined v-else-if="historyDetail.smoke_test_passed === false" style="color: #e5484d" />
        <span v-else>-</span>
      </Descriptions.Item>
      <Descriptions.Item v-if="historyDetail.export_path" label="导出路径" :span="2">
        <span style="font-family: monospace; font-size: 12px; word-break: break-all">{{ historyDetail.export_path }}</span>
      </Descriptions.Item>
      <Descriptions.Item v-if="historyDetail.model_path" label="模型路径" :span="2">
        <span style="font-family: monospace; font-size: 12px; word-break: break-all">{{ historyDetail.model_path }}</span>
      </Descriptions.Item>
      <Descriptions.Item v-if="historyDetail.error" label="错误信息" :span="2">
        <span style="color: #e5484d">{{ historyDetail.error }}</span>
      </Descriptions.Item>
    </Descriptions>

    <!-- Phase 1 real-labeled P/R/F1/AUROC/PR-AUC summary row -->
    <Descriptions
      v-if="historyDetail && (historyDetail.val_f1 != null || historyDetail.val_auroc != null)"
      bordered :column="4" size="small" style="margin-top: 12px"
      title="真实标注评估"
    >
      <Descriptions.Item label="Precision">
        {{ historyDetail.val_precision != null ? (historyDetail.val_precision * 100).toFixed(1) + '%' : '-' }}
      </Descriptions.Item>
      <Descriptions.Item label="Recall">
        {{ historyDetail.val_recall != null ? (historyDetail.val_recall * 100).toFixed(1) + '%' : '-' }}
      </Descriptions.Item>
      <Descriptions.Item label="F1">
        <Tag v-if="historyDetail.val_f1 != null" color="purple" style="font-weight: 600">
          {{ historyDetail.val_f1.toFixed(4) }}
        </Tag>
      </Descriptions.Item>
      <Descriptions.Item label="样本数">{{ historyDetail.val_real_sample_count ?? '-' }}</Descriptions.Item>
      <Descriptions.Item label="AUROC">{{ historyDetail.val_auroc?.toFixed(4) ?? '-' }}</Descriptions.Item>
      <Descriptions.Item label="PR-AUC">{{ historyDetail.val_pr_auc?.toFixed(4) ?? '-' }}</Descriptions.Item>
    </Descriptions>

    <!-- Phase 2: embedded MetricsChart (PR / ROC / threshold slider / CM) -->
    <div
      v-if="historyDetail && historyDetail.val_f1 != null"
      style="margin-top: 16px"
    >
      <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px">
        <LineChartOutlined />
        <Typography.Text strong>指标分析</Typography.Text>
      </div>
      <MetricsChart :record-id="historyDetail.id" />
    </div>
  </Modal>

  <!-- Compare Modal -->
  <Modal
    v-model:open="compareVisible"
    title="A/B 模型对比"
    :footer="null"
    width="720px"
    @cancel="compareResult = null"
  >
    <Form layout="vertical" style="margin-top: 16px">
      <div style="display: flex; gap: 16px">
        <Form.Item label="基准模型 (旧)" style="flex: 1">
          <Select
            v-model:value="compareForm.old_record_id"
            placeholder="选择训练记录"
            style="width: 100%"
            :options="completedRecords.map(r => ({ value: r.id, label: `#${r.id} ${r.camera_id} - ${r.model_type} (${r.quality_grade || 'N/A'})` }))"
          />
        </Form.Item>
        <Form.Item label="候选模型 (新)" style="flex: 1">
          <Select
            v-model:value="compareForm.new_record_id"
            placeholder="选择训练记录"
            style="width: 100%"
            :options="completedRecords.map(r => ({ value: r.id, label: `#${r.id} ${r.camera_id} - ${r.model_type} (${r.quality_grade || 'N/A'})` }))"
          />
        </Form.Item>
      </div>
      <Form.Item>
        <Button type="primary" :loading="comparing" @click="handleCompare">
          <template #icon><SwapOutlined /></template>
          开始对比
        </Button>
      </Form.Item>
    </Form>

    <div v-if="compareResult" style="margin-top: 16px">
      <div style="margin-bottom: 16px; text-align: center">
        <Tag
          :color="RECOMMENDATION_COLORS[compareResult.recommendation] || 'default'"
          style="font-size: 16px; padding: 6px 16px"
        >
          {{ RECOMMENDATION_TEXT[compareResult.recommendation] || compareResult.recommendation }}
        </Tag>
      </div>
      <Table
        :data-source="[
          { metric: '平均分数', old_val: compareResult.old?.mean?.toFixed(4), new_val: compareResult.new?.mean?.toFixed(4) },
          { metric: '标准差', old_val: compareResult.old?.std?.toFixed(4), new_val: compareResult.new?.std?.toFixed(4) },
          { metric: '最大值', old_val: compareResult.old?.max?.toFixed(4), new_val: compareResult.new?.max?.toFixed(4) },
          { metric: 'P95', old_val: compareResult.old?.p95?.toFixed(4), new_val: compareResult.new?.p95?.toFixed(4) },
          { metric: '推理延迟', old_val: compareResult.old?.latency ? `${compareResult.old.latency.toFixed(1)}ms` : '-', new_val: compareResult.new?.latency ? `${compareResult.new.latency.toFixed(1)}ms` : '-' },
        ]"
        :columns="[
          { title: '指标', dataIndex: 'metric', key: 'metric' },
          { title: '基准模型 (旧)', dataIndex: 'old_val', key: 'old_val' },
          { title: '候选模型 (新)', dataIndex: 'new_val', key: 'new_val' },
        ]"
        :pagination="false"
        size="small"
        row-key="metric"
      />
    </div>
  </Modal>
</template>
