<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { Card, Table, Space, Select, Input, Button, message } from 'ant-design-vue'
import { getAuditLogs } from '../../api'

const auditEntries = ref<any[]>([])
const auditLoading = ref(false)
const auditTotal = ref(0)
const auditPagination = reactive({ current: 1, pageSize: 20 })
const auditFilters = reactive({ user: '', action: '' })
const auditUserOptions = ref<string[]>([])

async function loadAudit() {
  auditLoading.value = true
  try {
    const res = await getAuditLogs({
      page: auditPagination.current,
      page_size: auditPagination.pageSize,
      user: auditFilters.user || undefined,
      action: auditFilters.action || undefined,
    })
    auditEntries.value = res.entries
    auditTotal.value = res.total
    const uniqueUsers = new Set<string>(res.entries.map((e: any) => e.user).filter(Boolean))
    auditUserOptions.value = Array.from(uniqueUsers).sort()
  } catch (e) {
    message.error('加载审计日志失败')
  } finally {
    auditLoading.value = false
  }
}

function onAuditTableChange(pagination: any) {
  auditPagination.current = pagination.current
  auditPagination.pageSize = pagination.pageSize
  loadAudit()
}

function onAuditFilterChange() {
  auditPagination.current = 1
  loadAudit()
}

onMounted(() => {
  loadAudit()
})

const auditColumns = [
  { title: '时间', dataIndex: 'timestamp', key: 'timestamp', width: 180 },
  { title: '用户', dataIndex: 'user', key: 'user', width: 120 },
  { title: '操作', dataIndex: 'action', key: 'action', width: 150 },
  { title: '详情', dataIndex: 'details', key: 'details', ellipsis: true },
  { title: 'IP 地址', dataIndex: 'ip_address', key: 'ip_address', width: 140 },
]
</script>

<template>
  <Card title="审计日志">
    <Space style="margin-bottom: 16px">
      <Select
        v-model:value="auditFilters.user"
        placeholder="按用户筛选"
        allow-clear
        style="width: 160px"
        @change="onAuditFilterChange"
      >
        <Select.Option v-for="u in auditUserOptions" :key="u" :value="u">{{ u }}</Select.Option>
      </Select>
      <Input
        v-model:value="auditFilters.action"
        placeholder="按操作筛选"
        allow-clear
        style="width: 200px"
        @press-enter="onAuditFilterChange"
      />
      <Button @click="onAuditFilterChange">筛选</Button>
    </Space>
    <Table
      :columns="auditColumns"
      :data-source="auditEntries"
      :loading="auditLoading"
      :pagination="{
        current: auditPagination.current,
        pageSize: auditPagination.pageSize,
        total: auditTotal,
        showSizeChanger: true,
        showTotal: (t: number) => `共 ${t} 条`,
      }"
      row-key="id"
      size="small"
      @change="onAuditTableChange"
    />
  </Card>
</template>
