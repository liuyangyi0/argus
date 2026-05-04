<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import {
  Tabs, Collapse, CollapsePanel, Table, Tag, Space, Button,
  Popconfirm, Empty, message,
} from 'ant-design-vue'
import { CheckOutlined, CloseOutlined, ReloadOutlined } from '@ant-design/icons-vue'
import TrainingCreate from './TrainingCreate.vue'
import TrainingJobsList from './TrainingJobsList.vue'
import TrainingHistory from './TrainingHistory.vue'
import BackboneList from './BackboneList.vue'
import { getTrainingJobs, confirmTrainingJob, rejectTrainingJob } from '../../api/training'
import { JOB_TYPE_LABELS, TRIGGER_LABELS } from '../../composables/useModelState'
import { extractErrorMessage } from '../../utils/error'
import type { TrainingJobInfo } from '../../types/api'

const props = defineProps<{
  cameras: any[]
}>()

const route = useRoute()

const activeSubTab = ref('jobs')
const pendingCount = ref(0)
const jobsListRef = ref<InstanceType<typeof TrainingJobsList> | null>(null)

function onSubTabChange(key: string | number) {
  activeSubTab.value = String(key)
}

function onRefreshJobs() {
  jobsListRef.value?.loadJobs()
  loadPendingJobs()
}

// ── Pending Confirmation panel (audit C-11 explicit entry point) ──
const pendingJobs = ref<TrainingJobInfo[]>([])
const pendingLoading = ref(false)
// Per-job action loading flags so each row's button shows its own spinner
// and the others stay clickable.
const pendingActionLoading = reactive<Record<string, 'confirm' | 'reject' | null>>({})

// Default-open. If the user arrived via ?tab=pending we keep it open and
// also scroll attention to it (visually it sits at the top of the tab).
const pendingPanelKeys = ref<string[]>(['pending'])

const cameFromBanner = computed(() => String(route.query.tab || '') === 'pending')

async function loadPendingJobs() {
  pendingLoading.value = true
  try {
    const res = await getTrainingJobs({ status: 'pending_confirmation', limit: 20 })
    pendingJobs.value = (res?.jobs || []) as TrainingJobInfo[]
    pendingCount.value = res?.pending_count ?? pendingJobs.value.length
  } catch (e) {
    message.error(extractErrorMessage(e, '加载待确认任务失败'))
  } finally {
    pendingLoading.value = false
  }
}

async function handlePendingConfirm(jobId: string) {
  pendingActionLoading[jobId] = 'confirm'
  try {
    await confirmTrainingJob(jobId, { confirmed_by: 'operator' })
    message.success('任务已确认，进入队列')
    onRefreshJobs()
  } catch (e) {
    const status = (e as { code?: number })?.code
    if (status === 409) {
      message.warning('任务已被其他操作员处理或状态已变更')
      onRefreshJobs()
    } else {
      message.error(extractErrorMessage(e, '确认失败'))
    }
  } finally {
    pendingActionLoading[jobId] = null
  }
}

async function handlePendingReject(jobId: string) {
  pendingActionLoading[jobId] = 'reject'
  try {
    await rejectTrainingJob(jobId, { rejected_by: 'operator' })
    message.info('任务已拒绝')
    onRefreshJobs()
  } catch (e) {
    const status = (e as { code?: number })?.code
    if (status === 409) {
      message.warning('任务已被其他操作员处理或状态已变更')
      onRefreshJobs()
    } else {
      message.error(extractErrorMessage(e, '拒绝失败'))
    }
  } finally {
    pendingActionLoading[jobId] = null
  }
}

const pendingColumns = [
  { title: '任务ID', dataIndex: 'job_id', key: 'job_id', width: 110, ellipsis: true },
  { title: '类型', dataIndex: 'job_type', key: 'job_type', width: 110 },
  { title: '触发方式', dataIndex: 'trigger_type', key: 'trigger_type', width: 110 },
  { title: '创建时间', dataIndex: 'created_at', key: 'created_at', width: 170 },
  { title: '操作', key: 'actions', width: 200 },
]

onMounted(loadPendingJobs)

// Keep the panel open whenever a banner-driven navigation lands here, even
// after Vue Router patches the same component on query change.
watch(
  () => route.query.tab,
  (tab) => {
    if (String(tab || '') === 'pending') {
      activeSubTab.value = 'jobs'
      pendingPanelKeys.value = ['pending']
      loadPendingJobs()
    }
  },
  { immediate: true },
)
</script>

<template>
  <Tabs :activeKey="activeSubTab" @change="onSubTabChange" type="card" size="small">
    <Tabs.TabPane key="jobs" tab="训练任务">
      <Collapse
        v-model:activeKey="pendingPanelKeys"
        style="margin-bottom: 16px"
        :class="{ 'pending-panel-highlight': cameFromBanner }"
      >
        <CollapsePanel key="pending">
          <template #header>
            <Space>
              <span style="font-weight: 600">待确认训练任务</span>
              <Tag v-if="pendingCount > 0" color="orange">{{ pendingCount }}</Tag>
              <Tag v-else color="default">0</Tag>
            </Space>
          </template>
          <template #extra>
            <Button
              size="small"
              type="text"
              :loading="pendingLoading"
              @click.stop="loadPendingJobs"
            >
              <template #icon><ReloadOutlined /></template>
              刷新
            </Button>
          </template>

          <Table
            :columns="pendingColumns"
            :data-source="pendingJobs"
            :loading="pendingLoading"
            row-key="job_id"
            size="small"
            :pagination="false"
          >
            <template #emptyText>
              <Empty description="暂无待确认任务" :image="Empty.PRESENTED_IMAGE_SIMPLE" />
            </template>
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'job_id'">
                <span style="font-family: 'JetBrains Mono', monospace">{{ String(record.job_id).slice(0, 8) }}</span>
              </template>
              <template v-else-if="column.key === 'job_type'">
                <Tag :color="record.job_type === 'ssl_backbone' ? 'purple' : 'blue'">
                  {{ JOB_TYPE_LABELS[record.job_type] || record.job_type }}
                </Tag>
              </template>
              <template v-else-if="column.key === 'trigger_type'">
                <Tag>{{ TRIGGER_LABELS[record.trigger_type!] || record.trigger_type }}</Tag>
              </template>
              <template v-else-if="column.key === 'created_at'">
                {{ record.created_at?.replace('T', ' ').slice(0, 19) }}
              </template>
              <template v-else-if="column.key === 'actions'">
                <Space>
                  <Popconfirm
                    title="确认启动此训练任务？"
                    @confirm="handlePendingConfirm(record.job_id)"
                  >
                    <Button
                      size="small"
                      type="primary"
                      :loading="pendingActionLoading[record.job_id] === 'confirm'"
                      :disabled="pendingActionLoading[record.job_id] != null"
                    >
                      <template #icon><CheckOutlined /></template>
                      确认
                    </Button>
                  </Popconfirm>
                  <Popconfirm
                    title="确认拒绝此训练任务？"
                    @confirm="handlePendingReject(record.job_id)"
                  >
                    <Button
                      size="small"
                      danger
                      :loading="pendingActionLoading[record.job_id] === 'reject'"
                      :disabled="pendingActionLoading[record.job_id] != null"
                    >
                      <template #icon><CloseOutlined /></template>
                      拒绝
                    </Button>
                  </Popconfirm>
                </Space>
              </template>
            </template>
          </Table>
        </CollapsePanel>
      </Collapse>

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

<style scoped>
.pending-panel-highlight {
  box-shadow: 0 0 0 2px rgba(217, 119, 6, 0.35);
  border-radius: 8px;
}
</style>
