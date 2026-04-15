<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  Card, Tag, Switch, Button, Tooltip, Space, Typography, message, Alert,
} from 'ant-design-vue'
import { ReloadOutlined, WarningOutlined } from '@ant-design/icons-vue'

import {
  getCrossCameraConfig,
  toggleModule,
  type CrossCameraConfigPayload,
} from '../../api'

defineOptions({ name: 'CrossCameraPanel' })

const cfg = ref<CrossCameraConfigPayload | null>(null)
const loading = ref(false)
const toggling = ref(false)
const error = ref<string | null>(null)

async function load() {
  loading.value = true
  error.value = null
  try {
    cfg.value = await getCrossCameraConfig()
  } catch (e: any) {
    error.value = e?.response?.data?.msg || e?.message || '加载跨相机配置失败'
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
    await toggleModule('cross_camera.enabled', next)
    cfg.value.enabled = next
    message.success(next
      ? '跨相机关联已启用（已运行的摄像头需要重启才会生效）'
      : '跨相机关联已关闭')
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
  if (!cfg.value?.enabled) return '分组关联器已关闭'
  if (r.total_pipelines === 0) return '无运行中的摄像头管线'
  if (r.correlator_present) {
    return `${r.total_pipelines} 路管线运行中，相关器已附加`
  }
  return `${r.total_pipelines} 路管线运行中，但相关器尚未附加（配置是后启用的 — 需重启摄像头）`
})

onMounted(load)
</script>

<template>
  <Card :loading="loading && !cfg" :bordered="false">
    <template #title>
      <Space>
        <span>跨相机关联校验器</span>
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

    <div v-if="cfg" class="cross-camera-panel">
      <!-- Master toggle + runtime status -->
      <div class="panel-section">
        <div class="toggle-row">
          <div class="toggle-info">
            <Typography.Text strong>启用跨相机关联</Typography.Text>
            <Typography.Text type="secondary" style="font-size: 12px">
              多个相机视野重叠时，由另一路相机交叉证实异常
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
          v-if="cfg.enabled && !cfg.runtime.correlator_present && cfg.runtime.total_pipelines > 0"
          type="warning"
          show-icon
          style="margin-top: 8px"
          message="切换后的 enabled 状态只会影响新启动的摄像头管线，已运行的管线需要重启才会挂载关联器。"
        >
          <template #icon><WarningOutlined /></template>
        </Alert>
      </div>

      <!-- Model configuration (read-only) -->
      <div class="panel-section">
        <Typography.Text strong>模型配置</Typography.Text>
        <div class="kv-grid">
          <div class="kv-row">
            <Tooltip title="对面相机在映射位置必须达到的最小异常分数，否则视为未证实">
              <span class="k">证实阈值 (corroboration_threshold)</span>
            </Tooltip>
            <span class="v">{{ cfg.corroboration_threshold.toFixed(2) }}</span>
          </div>
          <div class="kv-row">
            <Tooltip title="另一路相机异常热力图的有效期；超过此时长视为缺失">
              <span class="k">最大滞后时间 (max_age_seconds)</span>
            </Tooltip>
            <span class="v">{{ cfg.max_age_seconds.toFixed(1) }} s</span>
          </div>
          <div class="kv-row">
            <Tooltip title="未证实告警向下降级的等级数 (0-2)">
              <span class="k">严重度降级 (uncorroborated_severity_downgrade)</span>
            </Tooltip>
            <span class="v">{{ cfg.uncorroborated_severity_downgrade }}</span>
          </div>
          <div class="kv-row">
            <Tooltip title="当前 overlap_pairs 列表的相机对数量；需要在 stage 2.9 用 UI 编辑">
              <span class="k">已配置相机对数量</span>
            </Tooltip>
            <span class="v">{{ cfg.overlap_pairs.length }}</span>
          </div>
        </div>
      </div>

      <!-- Overlap pairs list -->
      <div class="panel-section">
        <Typography.Text strong>相机对（overlap_pairs）</Typography.Text>
        <div v-if="cfg.overlap_pairs.length === 0" class="empty-pairs">
          （无已配置的相机对；在 stage 2.9 添加）
        </div>
        <div v-else class="pairs-list">
          <div
            v-for="(pair, idx) in cfg.overlap_pairs"
            :key="`${pair.camera_a}-${pair.camera_b}-${idx}`"
            class="pair-card"
          >
            <div class="pair-header">
              <Tag color="blue">{{ pair.camera_a }}</Tag>
              <span class="pair-arrow">↔</span>
              <Tag color="geekblue">{{ pair.camera_b }}</Tag>
            </div>
            <div class="matrix-label">Homography (3×3)</div>
            <div class="matrix-grid">
              <div
                v-for="(row, ri) in pair.homography"
                :key="`row-${ri}`"
                class="matrix-row"
              >
                <div
                  v-for="(cell, ci) in row"
                  :key="`cell-${ri}-${ci}`"
                  class="matrix-cell"
                >
                  {{ cell.toFixed(3) }}
                </div>
              </div>
            </div>
          </div>
        </div>
        <Typography.Text type="secondary" style="font-size: 11px; display: block; margin-top: 12px">
          下一阶段（2.9）会支持在这里在线编辑相机对 + homography 矩阵，目前仅作只读展示。
        </Typography.Text>
      </div>
    </div>
  </Card>
</template>

<style scoped>
.cross-camera-panel {
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
.empty-pairs {
  padding: 14px;
  background: var(--argus-card-bg-solid, rgba(0,0,0,0.02));
  border-radius: 6px;
  color: var(--argus-text-muted);
  font-size: 13px;
  text-align: center;
}
.pairs-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.pair-card {
  padding: 12px 14px;
  background: var(--argus-card-bg-solid, rgba(0,0,0,0.02));
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 6px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.pair-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 500;
}
.pair-arrow {
  color: var(--argus-text-muted);
  font-size: 16px;
}
.matrix-label {
  font-size: 11px;
  color: var(--argus-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.matrix-grid {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 12px;
}
.matrix-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2px;
}
.matrix-cell {
  padding: 4px 8px;
  background: rgba(0,0,0,0.03);
  border-radius: 3px;
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--argus-text);
}
</style>
