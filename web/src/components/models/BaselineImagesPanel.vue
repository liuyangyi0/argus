<script setup lang="ts">
import { computed, h, onMounted, ref, watch } from 'vue'
import {
  Alert, Button, Empty, Popconfirm, Segmented, Space, Spin, Tooltip, Upload,
  message,
} from 'ant-design-vue'
import type { UploadChangeParam, UploadProps } from 'ant-design-vue'
import { InboxOutlined, DeleteOutlined, ReloadOutlined } from '@ant-design/icons-vue'

import {
  baselineImageContentUrl,
  deleteBaselineImage,
  listBaselineImages,
  uploadBaselineImage,
  type BaselineImageInfo,
} from '../../api/baselines'
import { extractErrorMessage } from '../../utils/error'

defineOptions({ name: 'BaselineImagesPanel' })

const props = defineProps<{
  cameraId: string
  version: string
  /** Lifecycle state of the version. 'active' locks the panel. */
  state?: string | null
  zoneId?: string
}>()

const emit = defineEmits<{
  (e: 'changed'): void
}>()

const UPLOAD_MAX_BYTES = 10 * 1024 * 1024
const ALLOWED_EXT = new Set(['png', 'jpg', 'jpeg'])

const images = ref<BaselineImageInfo[]>([])
const loading = ref(false)
const totalBytes = ref(0)
const viewMode = ref<'grid' | 'list'>('grid')
const uploading = ref(false)
const deletingFilename = ref<string | null>(null)
const serverSaysActive = ref(false)

// Panel is read-only whenever either the passed-in state *or* the server-echoed
// state reports ACTIVE. The server is authoritative — we keep the prop as a
// hint so the UI flips immediately after the drawer opens.
const isActive = computed(() => serverSaysActive.value || props.state === 'active')

const sizeHuman = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`
}

const formatCreatedAt = (iso: string): string => {
  // Collapse "2026-04-17T09:12:33.123456+00:00" to "2026-04-17 09:12:33"
  return iso.replace('T', ' ').split('.')[0].split('+')[0]
}

async function refresh() {
  if (!props.cameraId || !props.version) return
  loading.value = true
  try {
    const res = await listBaselineImages(
      props.cameraId,
      props.version,
      props.zoneId || 'default',
    )
    images.value = res.images || []
    totalBytes.value = res.total_bytes || 0
    serverSaysActive.value = !!res.is_active
  } catch (e) {
    images.value = []
    totalBytes.value = 0
    message.error(extractErrorMessage(e, '加载图片失败'))
  } finally {
    loading.value = false
  }
}

watch(
  () => [props.cameraId, props.version, props.zoneId],
  () => { void refresh() },
)

onMounted(() => { void refresh() })

function contentUrl(filename: string): string {
  return baselineImageContentUrl(
    props.cameraId, props.version, filename, props.zoneId || 'default',
  )
}

async function handleDelete(filename: string) {
  if (isActive.value) {
    message.warning('激活版本不可编辑')
    return
  }
  deletingFilename.value = filename
  try {
    await deleteBaselineImage(
      props.cameraId, props.version, filename, props.zoneId || 'default',
    )
    message.success(`${filename} 已删除`)
    await refresh()
    emit('changed')
  } catch (e) {
    message.error(extractErrorMessage(e, '删除失败'))
  } finally {
    deletingFilename.value = null
  }
}

/**
 * ant-design-vue Upload with :before-upload returning false handles the file
 * locally — we send it via our own axios call so we stay on the unified
 * {code,msg,data} envelope and can surface nice errors.
 */
const beforeUpload: UploadProps['beforeUpload'] = async (rawFile) => {
  if (isActive.value) {
    message.warning('激活版本不可上传')
    return Upload.LIST_IGNORE
  }
  const file = rawFile as File
  const ext = file.name.split('.').pop()?.toLowerCase() || ''
  if (!ALLOWED_EXT.has(ext)) {
    message.error(`只支持 png/jpg/jpeg，当前为 .${ext || '(无扩展名)'}`)
    return Upload.LIST_IGNORE
  }
  if (file.size > UPLOAD_MAX_BYTES) {
    message.error(`图片过大（${(file.size / 1024 / 1024).toFixed(1)} MB），上限 10 MB`)
    return Upload.LIST_IGNORE
  }
  uploading.value = true
  try {
    const res = await uploadBaselineImage(
      props.cameraId, props.version, file, props.zoneId || 'default',
    )
    message.success(`已上传 ${res.filename}`)
    await refresh()
    emit('changed')
  } catch (e) {
    message.error(extractErrorMessage(e, '上传失败'))
  } finally {
    uploading.value = false
  }
  return false // prevent ant Upload from doing its own XHR
}

function handleUploadChange(_info: UploadChangeParam) {
  // No-op — we handle everything in beforeUpload. This callback keeps the
  // ant Upload component's internal file list from getting stuck.
}
</script>

<template>
  <div class="baseline-images-panel">
    <div class="panel-header">
      <div class="header-left">
        <strong>图片管理</strong>
        <span class="muted">
          · {{ images.length }} 张 · {{ sizeHuman(totalBytes) }}
        </span>
      </div>
      <Space>
        <Segmented
          v-model:value="viewMode"
          :options="[{ label: '网格', value: 'grid' }, { label: '列表', value: 'list' }]"
          size="small"
        />
        <Tooltip title="刷新">
          <Button size="small" :icon="h(ReloadOutlined)" @click="refresh" />
        </Tooltip>
      </Space>
    </div>

    <Alert
      v-if="isActive"
      type="warning"
      show-icon
      message="激活版本的图片不可编辑"
      description="生产中的基线视为不可变；如需修改，请先将版本退役后再操作。"
      style="margin-bottom: 12px"
    />

    <Upload
      v-else
      :before-upload="beforeUpload"
      :show-upload-list="false"
      :multiple="false"
      :disabled="uploading"
      accept=".png,.jpg,.jpeg,image/png,image/jpeg"
      class="upload-dropzone"
      @change="handleUploadChange"
    >
      <div class="upload-inner">
        <InboxOutlined style="font-size: 28px; color: #1677ff" />
        <p style="margin: 4px 0 0 0">
          <span v-if="uploading">上传中...</span>
          <span v-else>点击或拖拽图片到此区域上传</span>
        </p>
        <p class="muted" style="font-size: 12px; margin: 0">
          仅支持 png/jpg/jpeg，单张 ≤ 10 MB
        </p>
      </div>
    </Upload>

    <Spin :spinning="loading">
      <Empty v-if="!loading && images.length === 0" description="暂无图片" />

      <div v-else-if="viewMode === 'grid'" class="img-grid">
        <div
          v-for="item in images"
          :key="item.filename"
          class="img-cell"
        >
          <Tooltip>
            <template #title>
              <div>{{ item.filename }}</div>
              <div>{{ sizeHuman(item.size_bytes) }}</div>
              <div>{{ formatCreatedAt(item.created_at) }}</div>
            </template>
            <div class="img-wrap">
              <img :src="contentUrl(item.filename)" :alt="item.filename" loading="lazy" />
              <Popconfirm
                title="确认删除？此操作不可撤销"
                ok-text="删除"
                cancel-text="取消"
                ok-type="danger"
                :disabled="isActive"
                @confirm="handleDelete(item.filename)"
              >
                <Button
                  class="delete-btn"
                  danger
                  size="small"
                  :icon="h(DeleteOutlined)"
                  :loading="deletingFilename === item.filename"
                  :disabled="isActive"
                />
              </Popconfirm>
            </div>
          </Tooltip>
        </div>
      </div>

      <div v-else class="img-list">
        <div v-for="item in images" :key="item.filename" class="img-list-row">
          <img :src="contentUrl(item.filename)" :alt="item.filename" loading="lazy" />
          <div class="meta">
            <div class="filename">{{ item.filename }}</div>
            <div class="muted" style="font-size: 12px">
              {{ sizeHuman(item.size_bytes) }} · {{ formatCreatedAt(item.created_at) }}
            </div>
          </div>
          <Popconfirm
            title="确认删除？此操作不可撤销"
            ok-text="删除"
            cancel-text="取消"
            ok-type="danger"
            :disabled="isActive"
            @confirm="handleDelete(item.filename)"
          >
            <Button
              danger
              size="small"
              :loading="deletingFilename === item.filename"
              :disabled="isActive"
            >
              删除
            </Button>
          </Popconfirm>
        </div>
      </div>
    </Spin>
  </div>
</template>

<style scoped>
.baseline-images-panel {
  padding: 8px 0;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.header-left {
  font-size: 14px;
}

.muted {
  color: #8890a0;
}

.upload-dropzone {
  display: block;
  margin-bottom: 12px;
}

.upload-inner {
  border: 1px dashed #d9d9d9;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
  background: #fafafa;
  cursor: pointer;
  transition: border-color .2s;
}

.upload-inner:hover {
  border-color: #1677ff;
}

.img-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
}

.img-cell {
  position: relative;
}

.img-wrap {
  position: relative;
  width: 100%;
  aspect-ratio: 1 / 1;
  background: #f3f3f3;
  border-radius: 4px;
  overflow: hidden;
}

.img-wrap img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.delete-btn {
  position: absolute;
  right: 4px;
  bottom: 4px;
  opacity: 0;
  transition: opacity .15s;
}

.img-cell:hover .delete-btn {
  opacity: 1;
}

.img-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.img-list-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px;
  border: 1px solid #f0f0f0;
  border-radius: 4px;
}

.img-list-row img {
  width: 64px;
  height: 64px;
  object-fit: cover;
  border-radius: 4px;
  background: #f3f3f3;
}

.img-list-row .meta {
  flex: 1;
  min-width: 0;
}

.img-list-row .filename {
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
