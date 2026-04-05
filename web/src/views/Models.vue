<script setup lang="ts">
import { ref } from 'vue'
import { Card, Tabs, Typography, Empty } from 'ant-design-vue'

const activeTab = ref('capture')
</script>

<template>
  <div>
    <Typography.Title :level="3" style="margin-bottom: 24px">模型管理</Typography.Title>
    <Tabs v-model:activeKey="activeTab">
      <Tabs.TabPane key="capture" tab="基线采集">
        <Card>
          <Empty description="基线采集功能开发中" />
        </Card>
      </Tabs.TabPane>
      <Tabs.TabPane key="browse" tab="基线浏览">
        <Card>
          <Empty description="基线浏览功能开发中" />
        </Card>
      </Tabs.TabPane>
      <Tabs.TabPane key="train" tab="模型训练">
        <Card>
          <Empty description="模型训练功能开发中" />
        </Card>
      </Tabs.TabPane>
      <Tabs.TabPane key="models" tab="模型管理">
        <Card>
          <Empty description="模型部署功能开发中" />
        </Card>
      </Tabs.TabPane>
      <Tabs.TabPane key="history" tab="训练报告">
        <Card>
          <Empty description="训练报告功能开发中" />
        </Card>
      </Tabs.TabPane>
    </Tabs>
  </div>
</template>
