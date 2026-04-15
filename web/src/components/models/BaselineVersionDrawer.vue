<script setup lang="ts">
import { h, ref } from 'vue'
import { storeToRefs } from 'pinia'
import {
  Drawer, Table, Tag, Button, Modal, Divider, Input, message,
} from 'ant-design-vue'

import { useBaselineStore } from '../../stores/useBaselineStore'
import { BASELINE_STATE_MAP } from '../../composables/useModelState'

defineOptions({ name: 'BaselineVersionDrawer' })

const store = useBaselineStore()
const {
  baselineVersions,
  versionsLoading,
  versionDrawerVisible,
  versionDrawerCamera,
  verifyingVersion,
  activatingVersion,
  deletingBaseline,
} = storeToRefs(store)

const versionColumns = [
  { title: '版本', dataIndex: 'version', key: 'version', width: 80 },
  { title: '状态', key: 'state', width: 90 },
  { title: '图片', dataIndex: 'image_count', key: 'image_count', width: 70 },
  { title: '审核人', dataIndex: 'verified_by', key: 'verified_by', width: 100 },
  { title: '审核时间', dataIndex: 'verified_at', key: 'verified_at', width: 140 },
  { title: '操作', key: 'action', width: 180 },
]

function handleVerify(record: any) {
  const verifiedByRef = ref('')
  Modal.confirm({
    title: '审核基线版本',
    content: () => h('div', [
      h('p', `确认审核通过 ${record.version}？`),
      h(Input, {
        placeholder: '请输入审核人姓名',
        value: verifiedByRef.value,
        'onUpdate:value': (v: string) => { verifiedByRef.value = v },
      }),
    ]),
    okText: '确认审核',
    cancelText: '取消',
    async onOk() {
      const verifiedBy = verifiedByRef.value.trim() || 'operator'
      try {
        await store.verifyBaselineVersion(record.camera_id, record.version, verifiedBy)
        message.success(`${record.version} 已审核通过`)
      } catch (e: any) {
        message.error(e.response?.data?.error || '审核失败')
      }
    },
  })
}

function handleActivate(record: any) {
  Modal.confirm({
    title: '激活基线版本',
    content: `确定将 ${record.version} 设为生产基线？当前 Active 版本将自动退役。`,
    okText: '确认激活',
    cancelText: '取消',
    async onOk() {
      try {
        await store.activateBaselineVersion(record.camera_id, record.version)
        message.success(`${record.version} 已激活`)
      } catch (e: any) {
        message.error(e.response?.data?.error || '激活失败')
      }
    },
  })
}

function handleRetire(record: any) {
  const reasonRef = ref('')
  Modal.confirm({
    title: '退役基线版本',
    content: () => h('div', [
      h('p', `确定退役 ${record.version}？退役后将保留数据但不再用于训练。`),
      h(Input, {
        placeholder: '退役原因（可选）',
        value: reasonRef.value,
        'onUpdate:value': (v: string) => { reasonRef.value = v },
      }),
    ]),
    okText: '确认退役',
    okType: 'danger',
    cancelText: '取消',
    async onOk() {
      try {
        await store.retireBaselineVersion(
          record.camera_id,
          record.version,
          reasonRef.value.trim() || '手动退役',
        )
        message.success(`${record.version} 已退役`)
      } catch (e: any) {
        message.error(e.response?.data?.error || '退役失败')
      }
    },
  })
}

function handleDelete(record: any) {
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
  <Drawer
    :open="versionDrawerVisible"
    @update:open="(v: boolean) => { if (!v) store.closeVersionDrawer() }"
    :title="`基线版本管理 — ${versionDrawerCamera}`"
    width="680"
    placement="right"
  >
    <Table
      :columns="versionColumns"
      :data-source="baselineVersions"
      :loading="versionsLoading"
      :pagination="false"
      row-key="version"
      size="small"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'state'">
          <Tag :color="(BASELINE_STATE_MAP[record.state] || {}).color || 'default'">
            {{ (BASELINE_STATE_MAP[record.state] || {}).text || record.state }}
          </Tag>
        </template>
        <template v-if="column.key === 'verified_at'">
          <span v-if="record.verified_at">{{ record.verified_at.replace('T', ' ').slice(0, 19) }}</span>
          <span v-else style="color: #999">-</span>
        </template>
        <template v-if="column.key === 'action'">
          <Button
            v-if="record.state === 'draft'"
            size="small"
            type="primary"
            :loading="verifyingVersion === record.version"
            @click="handleVerify(record)"
          >
            审核
          </Button>
          <Button
            v-if="record.state === 'verified'"
            size="small"
            type="primary"
            :loading="activatingVersion === record.version"
            @click="handleActivate(record)"
          >
            激活
          </Button>
          <Button v-if="record.state === 'active'" size="small" danger @click="handleRetire(record)">
            退役
          </Button>
          <Button
            v-if="record.state !== 'active'"
            size="small"
            danger
            :loading="deletingBaseline === `${record.camera_id}-${record.version}`"
            @click="handleDelete(record)"
          >
            删除
          </Button>
          <span v-if="record.state === 'retired'" style="color: #999">已退役</span>
        </template>
      </template>
    </Table>
    <Divider />
    <div style="color: #8890a0; font-size: 12px">
      <p><strong>状态流转:</strong> 草稿 → 已审核 → 生产中 → 已退役（严格单向，不可逆转）</p>
      <p><strong>训练要求:</strong> 仅「已审核」或「生产中」的基线可用于模型训练</p>
    </div>
  </Drawer>
</template>
