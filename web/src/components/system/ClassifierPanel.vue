<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import {
  Card, Tag, Switch, Button, Tooltip, Space, Typography, message, Alert, Empty,
} from 'ant-design-vue'
import { ReloadOutlined } from '@ant-design/icons-vue'

import { getClassifierConfig, toggleModule, type ClassifierConfigPayload } from '../../api'

defineOptions({ name: 'ClassifierPanel' })

const cfg = ref<ClassifierConfigPayload | null>(null)
const loading = ref(false)
const toggling = ref(false)
const error = ref<string | null>(null)

async function load() {
  loading.value = true
  error.value = null
  try {
    cfg.value = await getClassifierConfig()
  } catch (e: any) {
    error.value = e?.response?.data?.msg || e?.message || '加载分类器配置失败'
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
    await toggleModule('classifier.enabled', next)
    cfg.value.enabled = next
    message.success(next ? '分类器已启用' : '分类器已关闭')
    // Re-fetch so runtime counters reflect the new state.
    await load()
  } catch (e: any) {
    message.error(e?.response?.data?.msg || '切换失败')
  } finally {
    toggling.value = false
  }
}

// Vocabulary is bucketed into 4 groups so operators can see at a glance
// which labels escalate severity, which get suppressed, and which sit
// in the neutral middle. A label may appear in at most one bucket —
// the precedence order is high → low → suppressed → neutral.
interface LabelBucket {
  key: 'high' | 'low' | 'suppress' | 'neutral'
  label: string
  color: string
  labels: string[]
}

const buckets = computed<LabelBucket[]>(() => {
  if (!cfg.value) return []
  const high = new Set(cfg.value.high_risk_labels)
  const low = new Set(cfg.value.low_risk_labels)
  const suppress = new Set(cfg.value.suppress_labels)
  const neutral: string[] = []
  const highList: string[] = []
  const lowList: string[] = []
  const suppressList: string[] = []
  for (const label of cfg.value.vocabulary) {
    if (high.has(label)) highList.push(label)
    else if (low.has(label)) lowList.push(label)
    else if (suppress.has(label)) suppressList.push(label)
    else neutral.push(label)
  }
  // Include any labels in the role lists that aren't in the main vocabulary
  // either (operator curation drift) so nothing is silently hidden.
  for (const label of high) if (!cfg.value.vocabulary.includes(label)) highList.push(label)
  for (const label of low) if (!cfg.value.vocabulary.includes(label)) lowList.push(label)
  for (const label of suppress) if (!cfg.value.vocabulary.includes(label)) suppressList.push(label)

  return [
    { key: 'high', label: '高风险（触发严重度上调）', color: 'red', labels: highList },
    { key: 'low', label: '低风险', color: 'orange', labels: lowList },
    { key: 'suppress', label: '抑制区域', color: 'purple', labels: suppressList },
    { key: 'neutral', label: '中性识别', color: 'default', labels: neutral },
  ]
})

const runtimeSummary = computed(() => {
  const r = cfg.value?.runtime
  if (!r) return null
  if (!cfg.value?.enabled) return '分类器已关闭，管线未加载模型'
  if (r.total_pipelines === 0) return '无运行中的摄像头管线，无法判断加载状态'
  if (r.pipelines_loaded === r.total_pipelines) {
    return `已在全部 ${r.total_pipelines} 路管线就绪`
  }
  if (r.pipelines_loaded > 0) {
    return `${r.pipelines_loaded} / ${r.total_pipelines} 路管线已加载模型（首次推理后懒加载）`
  }
  return `${r.total_pipelines} 路管线已附加分类器，尚未触发首次加载`
})

onMounted(load)
</script>

<template>
  <Card :loading="loading && !cfg" :bordered="false">
    <template #title>
      <Space>
        <span>AI 异物分类器</span>
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

    <div v-if="cfg" class="classifier-panel">
      <!-- Master toggle + runtime status -->
      <div class="panel-section">
        <div class="toggle-row">
          <div class="toggle-info">
            <Typography.Text strong>启用分类器</Typography.Text>
            <Typography.Text type="secondary" style="font-size: 12px">
              YOLO-World 开放词表模型，对异常区域做二次识别
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
      </div>

      <!-- Model + threshold summary -->
      <div class="panel-section">
        <Typography.Text strong>模型配置</Typography.Text>
        <div class="kv-grid">
          <div class="kv-row">
            <span class="k">模型</span>
            <span class="v"><code>{{ cfg.model_name }}</code></span>
          </div>
          <div class="kv-row">
            <span class="k">推理阈值</span>
            <Tooltip title="仅异常分数高于此值的区域才会调用分类器">
              <span class="v">{{ cfg.min_anomaly_score_to_classify.toFixed(2) }}</span>
            </Tooltip>
          </div>
          <div v-if="cfg.custom_vocabulary_path" class="kv-row">
            <span class="k">自定义词表</span>
            <span class="v"><code>{{ cfg.custom_vocabulary_path }}</code></span>
          </div>
        </div>
      </div>

      <!-- Vocabulary buckets -->
      <div class="panel-section">
        <Typography.Text strong>识别词表 ({{ cfg.vocabulary.length }} 词)</Typography.Text>
        <Typography.Text type="secondary" style="font-size: 12px; display: block; margin-bottom: 8px">
          词表编辑能力将在下一阶段上线，这里仅作展示。
        </Typography.Text>
        <div v-for="bucket in buckets" :key="bucket.key" class="bucket">
          <div class="bucket-hd">
            <Tag :color="bucket.color">{{ bucket.label }}</Tag>
            <span class="bucket-count">{{ bucket.labels.length }}</span>
          </div>
          <div v-if="bucket.labels.length > 0" class="bucket-labels">
            <Tag
              v-for="label in bucket.labels"
              :key="bucket.key + ':' + label"
              :color="bucket.color"
              class="bucket-tag"
            >{{ label }}</Tag>
          </div>
          <Empty
            v-else
            :image="null"
            :description="`（无${bucket.label.split('（')[0]}标签）`"
            style="padding: 4px 0; margin: 0"
          />
        </div>
      </div>
    </div>
  </Card>
</template>

<style scoped>
.classifier-panel {
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
  font-size: 13px;
}
.kv-row .k {
  color: var(--argus-text-muted);
}
.kv-row .v {
  color: var(--argus-text);
}
.kv-row .v code {
  font-size: 12px;
  background: var(--argus-code-bg, rgba(0,0,0,0.04));
  padding: 1px 6px;
  border-radius: 3px;
}
.bucket {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px 0;
  border-bottom: 1px dashed var(--argus-border, #f0f0f0);
}
.bucket:last-child {
  border-bottom: none;
}
.bucket-hd {
  display: flex;
  align-items: center;
  gap: 8px;
}
.bucket-count {
  font-size: 11px;
  color: var(--argus-text-muted);
}
.bucket-labels {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.bucket-tag {
  margin: 0 !important;
  font-size: 11px;
}
</style>
