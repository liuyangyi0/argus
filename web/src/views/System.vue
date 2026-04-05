<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Card, Tabs, Descriptions, Typography, Empty } from 'ant-design-vue'
import { getHealth } from '../api'

const activeTab = ref('overview')
const health = ref<any>(null)

onMounted(async () => {
  try {
    const res = await getHealth()
    health.value = res.data
  } catch (e) {
    console.error('Health fetch error', e)
  }
})
</script>

<template>
  <div>
    <Typography.Title :level="3" style="margin-bottom: 24px">系统管理</Typography.Title>
    <Tabs v-model:activeKey="activeTab">
      <Tabs.TabPane key="overview" tab="系统概览">
        <Card v-if="health">
          <Descriptions :column="2" bordered size="small">
            <Descriptions.Item label="系统状态">{{ health.status?.toUpperCase() }}</Descriptions.Item>
            <Descriptions.Item label="运行时间">{{ Math.floor(health.uptime_seconds / 60) }} 分钟</Descriptions.Item>
            <Descriptions.Item label="累计告警">{{ health.total_alerts }}</Descriptions.Item>
            <Descriptions.Item label="Python 版本">{{ health.python_version }}</Descriptions.Item>
            <Descriptions.Item label="操作系统">{{ health.platform }}</Descriptions.Item>
            <Descriptions.Item label="摄像头数">{{ health.cameras?.length || 0 }}</Descriptions.Item>
          </Descriptions>
        </Card>
      </Tabs.TabPane>
      <Tabs.TabPane key="config" tab="配置管理">
        <Card><Empty description="配置管理功能开发中" /></Card>
      </Tabs.TabPane>
      <Tabs.TabPane key="backup" tab="数据备份">
        <Card><Empty description="数据备份功能开发中" /></Card>
      </Tabs.TabPane>
      <Tabs.TabPane key="audit" tab="审计日志">
        <Card><Empty description="审计日志功能开发中" /></Card>
      </Tabs.TabPane>
      <Tabs.TabPane key="users" tab="用户管理">
        <Card><Empty description="用户管理功能开发中" /></Card>
      </Tabs.TabPane>
    </Tabs>
  </div>
</template>
