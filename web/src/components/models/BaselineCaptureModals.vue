<script setup lang="ts">
import { ref } from 'vue'
import {
  Modal, Form, Select, InputNumber, Space, Radio, Collapse, Slider, message,
} from 'ant-design-vue'

import { startCapture, startCaptureJob } from '../../api'
import { SESSION_LABELS, SAMPLING_STRATEGIES } from '../../composables/useModelState'
import { extractErrorMessage } from '../../utils/error'
import type { CameraSummary } from '../../types/api'

defineOptions({ name: 'BaselineCaptureModals' })

const props = defineProps<{
  cameras: CameraSummary[]
  captureVisible: boolean
  advancedVisible: boolean
}>()

const emit = defineEmits<{
  (e: 'update:captureVisible', value: boolean): void
  (e: 'update:advancedVisible', value: boolean): void
  (e: 'captureStarted'): void
}>()

const captureForm = ref({
  camera_id: '',
  count: 100,
  interval: 2.0,
  session_label: 'daytime',
})
const captureSubmitting = ref(false)

const advCaptureForm = ref({
  camera_id: '',
  target_frames: 1000,
  duration_hours: 24,
  sampling_strategy: 'active',
  diversity_threshold: 0.3,
  frames_per_period: 50,
})
const advCaptureSubmitting = ref(false)

async function handleCapture() {
  if (!captureForm.value.camera_id) {
    message.warning('请先选择摄像头')
    return
  }
  captureSubmitting.value = true
  try {
    const form = new FormData()
    form.append('camera_id', captureForm.value.camera_id)
    form.append('count', String(captureForm.value.count))
    form.append('interval', String(captureForm.value.interval))
    form.append('session_label', captureForm.value.session_label)
    await startCapture(form)
    message.success('采集任务已启动')
    emit('update:captureVisible', false)
    emit('captureStarted')
  } catch (e) {
    message.error(extractErrorMessage(e, '启动采集失败'))
  } finally {
    captureSubmitting.value = false
  }
}

async function handleAdvCapture() {
  if (!advCaptureForm.value.camera_id) {
    message.warning('请先选择摄像头')
    return
  }
  advCaptureSubmitting.value = true
  try {
    const form = new FormData()
    form.append('camera_id', advCaptureForm.value.camera_id)
    form.append('target_frames', String(advCaptureForm.value.target_frames))
    form.append('duration_hours', String(advCaptureForm.value.duration_hours))
    form.append('sampling_strategy', advCaptureForm.value.sampling_strategy)
    form.append('diversity_threshold', String(advCaptureForm.value.diversity_threshold))
    form.append('frames_per_period', String(advCaptureForm.value.frames_per_period))
    await startCaptureJob(form)
    message.success('高级采集任务已启动')
    emit('update:advancedVisible', false)
    emit('captureStarted')
  } catch (e) {
    message.error(extractErrorMessage(e, '启动高级采集失败'))
  } finally {
    advCaptureSubmitting.value = false
  }
}
</script>

<template>
  <!-- Quick capture -->
  <Modal
    :open="props.captureVisible"
    @update:open="(v: boolean) => emit('update:captureVisible', v)"
    title="采集新基线"
    @ok="handleCapture"
    :confirmLoading="captureSubmitting"
    okText="开始采集"
    cancelText="取消"
  >
    <Form layout="vertical" style="margin-top: 16px">
      <Form.Item label="选择摄像头">
        <Select v-model:value="captureForm.camera_id" style="width: 100%">
          <Select.Option v-for="cam in props.cameras" :key="cam.camera_id" :value="cam.camera_id">
            {{ cam.camera_id }} — {{ cam.name }}
          </Select.Option>
        </Select>
      </Form.Item>
      <Form.Item label="采集场景">
        <Select v-model:value="captureForm.session_label" style="width: 100%">
          <Select.Option v-for="sl in SESSION_LABELS" :key="sl.value" :value="sl.value">
            {{ sl.label }}
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
    </Form>
  </Modal>

  <!-- Advanced capture -->
  <Modal
    :open="props.advancedVisible"
    @update:open="(v: boolean) => emit('update:advancedVisible', v)"
    title="高级基线采集"
    @ok="handleAdvCapture"
    :confirmLoading="advCaptureSubmitting"
    okText="开始采集"
    cancelText="取消"
    width="560px"
  >
    <Form layout="vertical" style="margin-top: 16px">
      <Form.Item label="选择摄像头">
        <Select v-model:value="advCaptureForm.camera_id" style="width: 100%">
          <Select.Option v-for="cam in props.cameras" :key="cam.camera_id" :value="cam.camera_id">
            {{ cam.camera_id }} — {{ cam.name }}
          </Select.Option>
        </Select>
      </Form.Item>
      <Form.Item label="采样策略">
        <Radio.Group v-model:value="advCaptureForm.sampling_strategy" style="width: 100%">
          <div v-for="s in SAMPLING_STRATEGIES" :key="s.value" style="margin-bottom: 8px">
            <Radio :value="s.value">
              <span style="font-weight: 500">{{ s.label }}</span>
              <div style="font-size: 12px; color: #8890a0; margin-left: 24px">{{ s.desc }}</div>
            </Radio>
          </div>
        </Radio.Group>
      </Form.Item>
      <Space :size="16">
        <Form.Item label="目标帧数">
          <InputNumber
            v-model:value="advCaptureForm.target_frames"
            :min="100"
            :max="10000"
            :step="100"
            style="width: 140px"
          />
        </Form.Item>
        <Form.Item label="持续时长（小时）">
          <InputNumber
            v-model:value="advCaptureForm.duration_hours"
            :min="1"
            :max="168"
            :step="1"
            style="width: 140px"
          />
        </Form.Item>
      </Space>
      <Collapse ghost>
        <Collapse.Panel key="advanced" header="高级参数">
          <Form.Item label="多样性阈值" v-if="advCaptureForm.sampling_strategy !== 'uniform'">
            <Slider
              v-model:value="advCaptureForm.diversity_threshold"
              :min="0.1"
              :max="0.9"
              :step="0.05"
              :marks="{ 0.1: '低', 0.3: '默认', 0.9: '高' }"
            />
            <div style="font-size: 12px; color: #8890a0">
              值越高，保留的帧越多样；值越低，接受更多相似帧
            </div>
          </Form.Item>
          <Form.Item label="每时段帧数" v-if="advCaptureForm.sampling_strategy === 'scheduled'">
            <InputNumber
              v-model:value="advCaptureForm.frames_per_period"
              :min="5"
              :max="200"
              :step="5"
              style="width: 140px"
            />
            <div style="font-size: 12px; color: #8890a0; margin-top: 4px">
              每个时段（清晨/正午/傍晚/深夜）采集的目标帧数
            </div>
          </Form.Item>
        </Collapse.Panel>
      </Collapse>
    </Form>
  </Modal>
</template>
