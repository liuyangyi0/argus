<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  Card, Tag, Switch, Button, Tooltip, Space, Typography, message, Alert,
} from 'ant-design-vue'
import { ReloadOutlined, WarningOutlined } from '@ant-design/icons-vue'

import {
  getSegmenterConfig,
  toggleModule,
  type SegmenterConfigPayload,
} from '../../api'

defineOptions({ name: 'SegmenterPanel' })

const cfg = ref<SegmenterConfigPayload | null>(null)
const loading = ref(false)
const toggling = ref(false)
const error = ref<string | null>(null)

async function load() {
  loading.value = true
  error.value = null
  try {
    cfg.value = await getSegmenterConfig()
  } catch (e: any) {
    error.value = e?.response?.data?.msg || e?.message || '加载分割器配置失败'
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
    await toggleModule('segmenter.enabled', next)
    cfg.value.enabled = next
    message.success(next
      ? '分割器已启用（已运行的摄像头需要重启才会生效）'
      : '分割器已关闭')
    await load()
  } catch (e: any) {
    message.error(e?.response?.data?.msg || '切换失败')
  } finally {
    toggling.value = false
  }
}

const runtimeSummary = computed(() => {
  const r = cfg.value?.runtime
  if (!r) return null
  if (!cfg.value?.enabled) return '分割器已关闭'
  if (r.total_pipelines === 0) return '无运行中的摄像头管线'
  if (r.pipelines_attached === 0) {
    return `${r.total_pipelines} 路管线运行中，但都没有挂载分割器（配置是后启用的 — 需重启摄像头）`
  }
  if (r.pipelines_loaded === r.pipelines_attached) {
    return `已在 ${r.pipelines_loaded} / ${r.total_pipelines} 路管线就绪`
  }
  return `${r.pipelines_loaded} / ${r.pipelines_attached} 路已挂载分割器已加载 SAM2 模型（首次推理后懒加载）`
})

// A nice human label for the model_size enum
const modelSizeLabel = computed(() => {
  const size = cfg.value?.model_size || ''
  const labels: Record<string, string> = {
    tiny: 'SAM2 Hiera Tiny（最快，精度最低）',
    small: 'SAM2 Hiera Small（推荐，速度/精度平衡）',
    base_plus: 'SAM2 Hiera Base+（更高精度，需更多显存）',
    large: 'SAM2 Hiera Large（最高精度，推理最慢）',
  }
  return labels[size] || size
})

onMounted(load)
</script>

<template>
  <Card :loading="loading && !cfg" :bordered="false">
    <template #title>
      <Space>
        <span>SAM2 实例分割器</span>
        <Tag v-if="cfg" :color="cfg.enabled ? 'green' : 'default'">
          {{ cfg.enabled ? '已启用' : '已关闭' }}
        </Tag>
      </Space>
    </template>
    <template #extra>
      <Button size="small" :loading="loading" @click="load">
        <template #icon><ReloadOutlined /></template>
        刷新
      </Button>
    </template>

    <Alert v-if="error" type="error" :message="error" show-icon style="margin-bottom: 16px" />

    <div v-if="cfg" class="segmenter-panel">
      <!-- Master toggle + runtime status -->
      <div class="panel-section">
        <div class="toggle-row">
          <div class="toggle-info">
            <Typography.Text strong>启用分割器</Typography.Text>
            <Typography.Text type="secondary" style="font-size: 12px">
              对异常热力图的峰值点调用 SAM2 获得精确的物体边界
            </Typography.Text>
          </div>
          <Switch
            :checked="cfg.enabled"
            :loading="toggling"
            @update:checked="handleToggle"
          />
        </div>
        <Typography.Text v-if="runtimeSummary" type="secondary" style="font-size: 12px">
          {{ runtimeSummary }}
        </Typography.Text>
        <Alert
          v-if="cfg.enabled && cfg.runtime.pipelines_attached === 0 && cfg.runtime.total_pipelines > 0"
          type="warning"
          show-icon
          style="margin-top: 8px"
          message="切换后的 enabled 状态只会影响新启动的摄像头管线，已运行的管线需要在 /cameras 页面手动重启。"
        >
          <template #icon><WarningOutlined /></template>
        </Alert>
      </div>

      <!-- Model + threshold summary -->
      <div class="panel-section">
        <Typography.Text strong>模型配置（只读）</Typography.Text>
        <div class="kv-grid">
          <div class="kv-row">
            <span class="k">模型尺寸</span>
            <span class="v">
              <code>{{ cfg.model_size }}</code>
              <span class="model-size-label">{{ modelSizeLabel }}</span>
            </span>
          </div>
          <div class="kv-row">
            <Tooltip title="每帧最多提取的热力图峰值点数 — 每个点独立调一次 SAM2 推理">
              <span class="k">最大峰值点数 (max_points)</span>
            </Tooltip>
            <span class="v">{{ cfg.max_points }}</span>
          </div>
          <div class="kv-row">
            <Tooltip title="只有异常分数高于此值的峰值点才会触发分割 — 低于此值的弱峰直接丢弃">
              <span class="k">峰值入选阈值 (min_anomaly_score)</span>
            </Tooltip>
            <span class="v">{{ cfg.min_anomaly_score.toFixed(2) }}</span>
          </div>
          <div class="kv-row">
            <Tooltip title="分割得到的 mask 面积小于此像素数的对象会被直接丢弃 — 用来过滤噪点">
              <span class="k">最小 mask 面积 (min_mask_area_px)</span>
            </Tooltip>
            <span class="v">{{ cfg.min_mask_area_px }} px</span>
          </div>
          <div class="kv-row">
            <Tooltip title="单次 SAM2 推理的超时时间 — 防止偶发慢调用拖垮整条推理管线">
              <span class="k">推理超时 (timeout_seconds)</span>
            </Tooltip>
            <span class="v">{{ cfg.timeout_seconds.toFixed(1) }} s</span>
          </div>
        </div>
        <Typography.Text type="secondary" style="font-size: 12px; display: block; margin-top: 8px">
          下一阶段（2.6）这些参数会变成可编辑 + 热推送到运行中管线，目前仅作只读展示。
        </Typography.Text>
      </div>
    </div>
  </Card>
</template>

<style scoped>
.segmenter-panel {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.panel-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.toggle-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}
.toggle-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.kv-grid {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 14px;
  background: var(--argus-card-bg-solid, rgba(0,0,0,0.02));
  border-radius: 6px;
}
.kv-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  gap: 12px;
}
.kv-row .k {
  color: var(--argus-text-muted);
  cursor: help;
  border-bottom: 1px dashed transparent;
}
.kv-row .k:hover {
  border-bottom-color: var(--argus-text-muted);
}
.kv-row .v {
  color: var(--argus-text);
  font-variant-numeric: tabular-nums;
}
.kv-row .v code {
  font-size: 12px;
  background: var(--argus-code-bg, rgba(0,0,0,0.04));
  padding: 1px 6px;
  border-radius: 3px;
  margin-right: 6px;
}
.model-size-label {
  color: var(--argus-text-muted);
  font-size: 11px;
}
</style>
