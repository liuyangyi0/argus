<script setup lang="ts">
import { ref, computed } from 'vue'
import { message } from 'ant-design-vue'
import axios from 'axios'

const props = defineProps<{ cameraId: string }>()

const step = ref(0)
const calibrated = ref(false)
const reprojError = ref(0)
const loading = ref(false)
const gridPoints = ref<Array<{ world_x_mm: number; world_y_mm: number; pixel_x: number; pixel_y: number }>>([])
const cameraPosition = ref<{ x_mm: number; y_mm: number; z_mm: number } | null>(null)

const steps = [
  { title: '检查状态', description: '查看当前标定状态' },
  { title: '上传/计算', description: '上传标定文件或从棋盘格计算' },
  { title: '验证', description: '验证标定精度' },
]

async function checkStatus() {
  loading.value = true
  try {
    const res = await axios.get(`/api/calibration/${props.cameraId}`)
    const data = res.data?.data || res.data
    calibrated.value = data.calibrated
    reprojError.value = data.reprojection_error || 0
    if (calibrated.value) step.value = 2
  } catch {
    message.error('无法获取标定状态')
  } finally {
    loading.value = false
  }
}

async function uploadFile(file: File) {
  const formData = new FormData()
  formData.append('file', file)
  loading.value = true
  try {
    await axios.post(`/api/calibration/${props.cameraId}/upload`, formData)
    message.success('标定文件上传成功')
    calibrated.value = true
    step.value = 2
  } catch {
    message.error('上传失败')
  } finally {
    loading.value = false
  }
  return false // prevent default upload
}

async function computeCalibration() {
  loading.value = true
  try {
    const res = await axios.post(`/api/calibration/${props.cameraId}/compute`)
    const data = res.data?.data || res.data
    reprojError.value = data.reprojection_error || 0
    calibrated.value = true
    message.success(`标定完成，重投影误差: ${reprojError.value.toFixed(3)} px`)
    step.value = 2
  } catch (err: any) {
    message.error(err.response?.data?.message || '标定计算失败')
  } finally {
    loading.value = false
  }
}

async function verifyCalibration() {
  loading.value = true
  try {
    const res = await axios.get(`/api/calibration/${props.cameraId}/verify`)
    const data = res.data?.data || res.data
    gridPoints.value = data.grid_points || []
    cameraPosition.value = data.camera_position || null
    message.success(`验证完成，${gridPoints.value.length} 个网格点`)
  } catch {
    message.error('验证失败')
  } finally {
    loading.value = false
  }
}

const statusText = computed(() =>
  calibrated.value
    ? `已标定 (重投影误差: ${reprojError.value.toFixed(3)} px)`
    : '未标定'
)
</script>

<template>
  <a-card title="相机标定向导" :bordered="false">
    <a-steps :current="step" size="small" style="margin-bottom: 24px">
      <a-step v-for="s in steps" :key="s.title" :title="s.title" :description="s.description" />
    </a-steps>

    <!-- Step 0: Check status -->
    <div v-if="step === 0">
      <a-button type="primary" :loading="loading" @click="checkStatus">
        检查标定状态
      </a-button>
      <a-button style="margin-left: 8px" @click="step = 1">
        跳过，直接上传
      </a-button>
    </div>

    <!-- Step 1: Upload or compute -->
    <div v-if="step === 1">
      <a-space direction="vertical" style="width: 100%">
        <a-card size="small" title="方式一：上传标定文件">
          <a-upload
            :before-upload="uploadFile"
            accept=".json"
            :show-upload-list="false"
          >
            <a-button :loading="loading">选择 JSON 文件</a-button>
          </a-upload>
        </a-card>
        <a-card size="small" title="方式二：从棋盘格图片计算">
          <p style="color: #888; font-size: 12px">
            将棋盘格图片放入 data/calibration/{{ cameraId }}/frames/ 目录后点击计算
          </p>
          <a-button type="primary" :loading="loading" @click="computeCalibration">
            开始计算
          </a-button>
        </a-card>
      </a-space>
    </div>

    <!-- Step 2: Verify -->
    <div v-if="step === 2">
      <a-descriptions :column="1" bordered size="small" style="margin-bottom: 16px">
        <a-descriptions-item label="状态">
          <a-tag :color="calibrated ? 'green' : 'red'">{{ statusText }}</a-tag>
        </a-descriptions-item>
        <a-descriptions-item v-if="cameraPosition" label="相机位置">
          X={{ cameraPosition.x_mm.toFixed(0) }}mm
          Y={{ cameraPosition.y_mm.toFixed(0) }}mm
          Z={{ cameraPosition.z_mm.toFixed(0) }}mm
        </a-descriptions-item>
      </a-descriptions>
      <a-button type="primary" :loading="loading" @click="verifyCalibration">
        验证标定（生成网格叠加）
      </a-button>
      <div v-if="gridPoints.length" style="margin-top: 12px; font-size: 12px; color: #888">
        {{ gridPoints.length }} 个世界坐标→像素映射点已生成，可在实时画面中叠加显示
      </div>
    </div>
  </a-card>
</template>
