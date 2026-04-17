<script setup lang="ts">
import { onMounted, ref } from 'vue'
import {
  Card, Tag, Switch, Button, Tooltip, Space, Typography, message, Alert,
  Select, Slider,
} from 'ant-design-vue'
import { ReloadOutlined, WarningOutlined } from '@ant-design/icons-vue'

import {
  getImagingConfig,
  toggleModule,
  type ImagingConfigPayload,
} from '../../api'

defineOptions({ name: 'ImagingPanel' })

const cfg = ref<ImagingConfigPayload | null>(null)
const loading = ref(false)
const toggling = ref(false)
const error = ref<string | null>(null)

async function load() {
  loading.value = true
  error.value = null
  try {
    cfg.value = await getImagingConfig()
  } catch (e: any) {
    error.value = e?.response?.data?.msg || e?.message || '加载成像配置失败'
    cfg.value = null
  } finally {
    loading.value = false
  }
}

async function handleToggle(checked: string | number | boolean) {
  if (!cfg.value) return
  const next = Boolean(checked)
  toggling.value = true
  try {
    await toggleModule('imaging.enabled', next)
    cfg.value.enabled = next
    message.success(next ? '多模态成像已启用' : '多模态成像已关闭')
    await load()
  } catch (e: any) {
    message.error(e?.response?.data?.msg || '切换失败')
  } finally {
    toggling.value = false
  }
}

async function handlePolarizationToggle(checked: string | number | boolean) {
  if (!cfg.value) return
  const next = Boolean(checked)
  toggling.value = true
  try {
    await toggleModule('imaging.polarization_processing', next)
    cfg.value.polarization_processing = next
    message.success(next ? '偏振处理已启用' : '偏振处理已关闭')
  } catch (e: any) {
    message.error(e?.response?.data?.msg || '切换失败')
  } finally {
    toggling.value = false
  }
}

const modeLabels: Record<string, string> = {
  visible_only: '可见光',
  polarization: '偏振去反光',
  polarization_nir: '偏振 + NIR',
}

const sdkLabels: Record<string, string> = {
  opencv: 'OpenCV',
  arena: 'Arena (LUCID)',
  spinnaker: 'Spinnaker (FLIR)',
}

const deglareLabels: Record<string, string> = {
  stokes: 'Stokes 分解',
  min_intensity: '最小强度',
}

onMounted(load)
</script>

<template>
  <Card
    title="多模态成像"
    :loading="loading"
    size="small"
    :headStyle="{ padding: '0 16px' }"
    :bodyStyle="{ padding: '12px 16px' }"
  >
    <template #extra>
      <Space :size="8">
        <Switch
          :checked="cfg?.enabled"
          :loading="toggling"
          checked-children="ON"
          un-checked-children="OFF"
          @change="handleToggle"
        />
        <Button size="small" @click="load" :loading="loading">
          <template #icon><ReloadOutlined /></template>
        </Button>
      </Space>
    </template>

    <Alert
      v-if="error"
      type="error"
      :message="error"
      show-icon
      closable
      style="margin-bottom: 12px"
    />

    <template v-if="cfg">
      <!-- Runtime status -->
      <div style="margin-bottom: 12px">
        <Tag :color="cfg.enabled ? 'green' : 'default'">
          {{ cfg.enabled ? '已启用' : '已关闭' }}
        </Tag>
        <Tag color="blue">
          {{ cfg.runtime.pipelines_with_imaging }}/{{ cfg.runtime.total_pipelines }} 管线已加载
        </Tag>
        <Tag v-if="cfg.enabled && cfg.runtime.pipelines_with_imaging < cfg.runtime.total_pipelines" color="orange">
          <WarningOutlined /> 需重启管线生效
        </Tag>
      </div>

      <!-- Config details -->
      <div class="config-grid">
        <div class="config-row">
          <Typography.Text type="secondary">成像模式</Typography.Text>
          <Tag>{{ modeLabels[cfg.mode] || cfg.mode }}</Tag>
        </div>

        <div class="config-row">
          <Typography.Text type="secondary">相机 SDK</Typography.Text>
          <Tag :color="cfg.camera_sdk === 'opencv' ? 'default' : 'blue'">
            {{ sdkLabels[cfg.camera_sdk] || cfg.camera_sdk }}
          </Tag>
        </div>

        <div class="config-row">
          <Typography.Text type="secondary">偏振处理</Typography.Text>
          <Switch
            :checked="cfg.polarization_processing"
            :loading="toggling"
            :disabled="!cfg.enabled"
            size="small"
            @change="handlePolarizationToggle"
          />
        </div>

        <div class="config-row">
          <Typography.Text type="secondary">去反光方法</Typography.Text>
          <Tag>{{ deglareLabels[cfg.deglare_method] || cfg.deglare_method }}</Tag>
        </div>

        <div class="config-row">
          <Typography.Text type="secondary">融合通道数</Typography.Text>
          <Tag>{{ cfg.fusion_channels }} 通道</Tag>
        </div>

        <div class="config-row">
          <Typography.Text type="secondary">DoLP 阈值</Typography.Text>
          <Tag>{{ cfg.dolp_threshold.toFixed(2) }}</Tag>
        </div>

        <div class="config-row">
          <Typography.Text type="secondary">NIR 频闪</Typography.Text>
          <Tag :color="cfg.nir_strobe_enabled ? 'green' : 'default'">
            {{ cfg.nir_strobe_enabled ? '已启用' : '未启用' }}
          </Tag>
        </div>
      </div>
    </template>
  </Card>
</template>

<style scoped>
.config-grid {
  display: grid;
  gap: 8px;
}
.config-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
