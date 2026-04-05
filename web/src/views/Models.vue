<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  Card, Tabs, Typography, Table, Button, Select, Form, InputNumber,
  Space, Empty, message,
} from 'ant-design-vue'
import api, { getCameras } from '../api'

const activeTab = ref('capture')
const cameras = ref<any[]>([])
const loading = ref(false)

// ── Capture state ──
const captureForm = ref({ camera_id: '', count: 100, interval: 2.0 })
const capturing = ref(false)

// ── Models state ──
const models = ref<any[]>([])
const modelsLoading = ref(false)

// ── Training history ──
const trainingHistory = ref<any[]>([])
const historyLoading = ref(false)


onMounted(async () => {
  try {
    const res = await getCameras()
    cameras.value = res.data
    if (cameras.value.length > 0) {
      captureForm.value.camera_id = cameras.value[0].camera_id
    }
  } catch (e) {
    console.error(e)
  }
})

async function startCapture() {
  capturing.value = true
  try {
    const form = new FormData()
    form.append('camera_id', captureForm.value.camera_id)
    form.append('count', String(captureForm.value.count))
    form.append('interval', String(captureForm.value.interval))
    form.append('session_label', 'daytime')
    await api.post('/baseline/capture', form)
    message.success('采集任务已启动')
  } catch (e: any) {
    message.error(e.response?.data?.error || '启动失败')
  } finally {
    capturing.value = false
  }
}

async function loadModels() {
  modelsLoading.value = true
  try {
    // Use the existing HTML endpoint but parse what we can
    const res = await api.get('/baseline/models', { responseType: 'text' })
    // Extract model info from HTML (temporary until JSON API exists)
    const html = res.data as string
    const rows: any[] = []
    const trRegex = /<tr>\s*<td>([^<]+)<\/td>\s*<td>([^<]+)<\/td>\s*<td>([^<]+)<\/td>\s*<td>([^<]+)<\/td>/g
    let match
    while ((match = trRegex.exec(html)) !== null) {
      rows.push({
        camera_id: match[1].trim(),
        format: match[2].trim(),
        size: match[3].trim(),
        time: match[4].trim(),
      })
    }
    models.value = rows
  } catch (e) {
    console.error(e)
  } finally {
    modelsLoading.value = false
  }
}

async function loadHistory() {
  historyLoading.value = true
  try {
    const res = await api.get('/baseline/training-history', { responseType: 'text' })
    const html = res.data as string
    const rows: any[] = []
    const trRegex = /<tr[^>]*>\s*<td>(\d+)<\/td>\s*<td>([^<]+)<\/td>\s*<td>([^<]+)<\/td>/g
    let match
    while ((match = trRegex.exec(html)) !== null) {
      rows.push({ id: match[1], camera_id: match[2].trim(), model_type: match[3].trim() })
    }
    trainingHistory.value = rows
  } catch (e) {
    console.error(e)
  } finally {
    historyLoading.value = false
  }
}

async function startTraining() {
  loading.value = true
  try {
    const form = new FormData()
    form.append('camera_id', captureForm.value.camera_id)
    form.append('model_type', 'patchcore')
    form.append('export_format', 'openvino')
    await api.post('/baseline/train', form)
    message.success('训练任务已启动')
  } catch (e: any) {
    message.error(e.response?.data?.error || '训练启动失败')
  } finally {
    loading.value = false
  }
}

function onTabChange(key: string | number) {
  activeTab.value = String(key)
  if (key === 'models') loadModels()
  if (key === 'history') loadHistory()
}

const modelColumns = [
  { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
  { title: '格式', dataIndex: 'format', key: 'format' },
  { title: '大小', dataIndex: 'size', key: 'size' },
  { title: '训练时间', dataIndex: 'time', key: 'time' },
]
</script>

<template>
  <div>
    <Typography.Title :level="3" style="margin-bottom: 24px">模型管理</Typography.Title>
    <Tabs :activeKey="activeTab" @change="onTabChange">
      <!-- Capture Tab -->
      <Tabs.TabPane key="capture" tab="基线采集">
        <Card title="开始基线采集">
          <p style="color: #888; margin-bottom: 16px">
            从在线摄像头采集"正常"场景的参考图片，用于训练异常检测模型。
          </p>
          <Form layout="vertical" style="max-width: 600px">
            <Form.Item label="选择摄像头">
              <Select v-model:value="captureForm.camera_id" style="width: 100%">
                <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
                  {{ cam.camera_id }} — {{ cam.name }}
                </Select.Option>
              </Select>
            </Form.Item>
            <Space>
              <Form.Item label="采集帧数">
                <InputNumber v-model:value="captureForm.count" :min="10" :max="1000" />
              </Form.Item>
              <Form.Item label="间隔（秒）">
                <InputNumber v-model:value="captureForm.interval" :min="0.5" :max="60" :step="0.5" />
              </Form.Item>
            </Space>
            <Form.Item>
              <Button type="primary" :loading="capturing" @click="startCapture">开始采集</Button>
            </Form.Item>
          </Form>
        </Card>
      </Tabs.TabPane>

      <!-- Browse Tab -->
      <Tabs.TabPane key="browse" tab="基线浏览">
        <Card>
          <Empty description="基线图片浏览器 — 需要后端 JSON API 支持" />
        </Card>
      </Tabs.TabPane>

      <!-- Train Tab -->
      <Tabs.TabPane key="train" tab="模型训练">
        <Card title="训练异常检测模型">
          <p style="color: #888; margin-bottom: 16px">
            使用基线图片训练异常检测模型。训练耗时通常 5-15 分钟。
          </p>
          <Form layout="vertical" style="max-width: 600px">
            <Form.Item label="选择摄像头">
              <Select v-model:value="captureForm.camera_id" style="width: 100%">
                <Select.Option v-for="cam in cameras" :key="cam.camera_id" :value="cam.camera_id">
                  {{ cam.camera_id }} — {{ cam.name }}
                </Select.Option>
              </Select>
            </Form.Item>
            <Form.Item>
              <Button type="primary" :loading="loading" @click="startTraining">开始训练</Button>
            </Form.Item>
          </Form>
        </Card>
      </Tabs.TabPane>

      <!-- Models Tab -->
      <Tabs.TabPane key="models" tab="模型管理">
        <Card>
          <Table
            :columns="modelColumns"
            :data-source="models"
            :loading="modelsLoading"
            :pagination="false"
            row-key="camera_id"
            size="small"
          />
          <div v-if="models.length === 0 && !modelsLoading" style="text-align: center; padding: 32px; color: #666">
            暂无已训练的模型
          </div>
        </Card>
      </Tabs.TabPane>

      <!-- History Tab -->
      <Tabs.TabPane key="history" tab="训练报告">
        <Card>
          <div v-if="trainingHistory.length === 0 && !historyLoading" style="text-align: center; padding: 32px; color: #666">
            暂无训练记录
          </div>
          <Table
            v-else
            :data-source="trainingHistory"
            :loading="historyLoading"
            :pagination="false"
            row-key="id"
            size="small"
            :columns="[
              { title: 'ID', dataIndex: 'id', key: 'id', width: 60 },
              { title: '摄像头', dataIndex: 'camera_id', key: 'camera_id' },
              { title: '模型类型', dataIndex: 'model_type', key: 'model_type' },
            ]"
          />
        </Card>
      </Tabs.TabPane>
    </Tabs>
  </div>
</template>
