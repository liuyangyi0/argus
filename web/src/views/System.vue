<script setup lang="ts">
import { ref } from 'vue'
import { Tabs, Typography } from 'ant-design-vue'

import SystemOverviewPanel from '../components/system/SystemOverviewPanel.vue'
import SystemConfigPanel from '../components/system/SystemConfigPanel.vue'
import SystemAuditPanel from '../components/system/SystemAuditPanel.vue'
import SystemDegradationPanel from '../components/system/SystemDegradationPanel.vue'
import SystemUserPanel from '../components/system/SystemUserPanel.vue'
import SystemRegionPanel from '../components/system/SystemRegionPanel.vue'
import SystemNotificationTemplatePanel from '../components/system/SystemNotificationTemplatePanel.vue'
import ModuleTogglePanel from '../components/system/ModuleTogglePanel.vue'
import ClassifierPanel from '../components/system/ClassifierPanel.vue'
import SegmenterPanel from '../components/system/SegmenterPanel.vue'
import CrossCameraPanel from '../components/system/CrossCameraPanel.vue'
import ImagingPanel from '../components/system/ImagingPanel.vue'
import ModelStatusPanel from '../components/system/ModelStatusPanel.vue'

const activeTab = ref('overview')

function onTabChange(key: string | number) {
  activeTab.value = String(key)
}
</script>

<template>
  <main class="glass" style=" padding: 24px; border-radius: var(--r-lg); min-width: 0; display: flex; flex-direction: column; flex: 1; overflow-y: auto;">
    <Typography.Title :level="3" style="margin-bottom: 24px; color: var(--ink)">系统管理</Typography.Title>
    <Tabs :activeKey="activeTab" @change="onTabChange">
      <!-- Overview -->
      <Tabs.TabPane key="overview" tab="系统概览">
        <SystemOverviewPanel />
        <div style="margin-top: 16px">
          <ModelStatusPanel />
        </div>
      </Tabs.TabPane>

      <!-- Model Runtime Status -->
      <Tabs.TabPane key="model-status" tab="模型状态">
        <ModelStatusPanel v-if="activeTab === 'model-status'" />
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

      <!-- SAM2 Segmenter -->
      <Tabs.TabPane key="segmenter" tab="分割器">
        <SegmenterPanel v-if="activeTab === 'segmenter'" />
      </Tabs.TabPane>

      <!-- Multi-modal Imaging -->
      <Tabs.TabPane key="imaging" tab="多模态成像">
        <ImagingPanel v-if="activeTab === 'imaging'" />
      </Tabs.TabPane>

      <!-- Cross-Camera Correlation -->
      <Tabs.TabPane key="cross-camera" tab="跨相机">
        <CrossCameraPanel v-if="activeTab === 'cross-camera'" />
      </Tabs.TabPane>

      <!-- Users -->
      <Tabs.TabPane key="users" tab="用户管理">
        <SystemUserPanel v-if="activeTab === 'users'" />
      </Tabs.TabPane>
      <Tabs.TabPane key="regions" tab="区域管理">
        <SystemRegionPanel v-if="activeTab === 'regions'" />
      </Tabs.TabPane>
      <Tabs.TabPane key="notification-templates" tab="通知内容配置">
        <SystemNotificationTemplatePanel v-if="activeTab === 'notification-templates'" />
      </Tabs.TabPane>
    </Tabs>
  </main>
</template>
