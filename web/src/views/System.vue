<script setup lang="ts">
import { ref } from 'vue'
import { Tabs, Typography } from 'ant-design-vue'

import SystemOverviewPanel from '../components/system/SystemOverviewPanel.vue'
import SystemConfigPanel from '../components/system/SystemConfigPanel.vue'
import SystemAuditPanel from '../components/system/SystemAuditPanel.vue'
import SystemDegradationPanel from '../components/system/SystemDegradationPanel.vue'
import SystemUserPanel from '../components/system/SystemUserPanel.vue'
import ModuleTogglePanel from '../components/system/ModuleTogglePanel.vue'
import ClassifierPanel from '../components/system/ClassifierPanel.vue'

const activeTab = ref('overview')

function onTabChange(key: string | number) {
  activeTab.value = String(key)
}
</script>

<template>
  <main class="glass" style="margin: 12px; padding: 24px; border-radius: var(--r-lg); min-width: 0; display: flex; flex-direction: column; flex: 1;">
    <Typography.Title :level="3" style="margin-bottom: 24px; color: var(--ink)">系统管理</Typography.Title>
    <Tabs :activeKey="activeTab" @change="onTabChange">
      <!-- Overview -->
      <Tabs.TabPane key="overview" tab="系统概览">
        <SystemOverviewPanel />
      </Tabs.TabPane>

      <!-- Config & Backup target merged into SystemConfigPanel for brevity but we use tabs for old links -->
      <Tabs.TabPane key="config" tab="配置管理">
        <SystemConfigPanel />
      </Tabs.TabPane>

      <!-- Audit -->
      <Tabs.TabPane key="audit" tab="审计日志">
        <SystemAuditPanel v-if="activeTab === 'audit'" />
      </Tabs.TabPane>

      <!-- Degradation History -->
      <Tabs.TabPane key="degradation" tab="降级事件">
        <SystemDegradationPanel v-if="activeTab === 'degradation'" />
      </Tabs.TabPane>

      <!-- Module Toggles -->
      <Tabs.TabPane key="modules" tab="功能模块">
        <ModuleTogglePanel v-if="activeTab === 'modules'" />
      </Tabs.TabPane>

      <!-- AI Classifier -->
      <Tabs.TabPane key="classifier" tab="分类器">
        <ClassifierPanel v-if="activeTab === 'classifier'" />
      </Tabs.TabPane>

      <!-- Users -->
      <Tabs.TabPane key="users" tab="用户管理">
        <SystemUserPanel v-if="activeTab === 'users'" />
      </Tabs.TabPane>
    </Tabs>
  </main>
</template>
