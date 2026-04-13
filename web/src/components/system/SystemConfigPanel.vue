<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Card, Space, Button, Typography, Switch, Select, Input, message } from 'ant-design-vue'
import { reloadConfig, createBackup, getAudioAlerts, updateAudioAlerts } from '../../api'

const configLoading = ref(false)
const audioConfig = ref<any>({ low: {}, medium: {}, high: {} })
const audioLoading = ref(false)

async function handleReloadConfig() {
  configLoading.value = true
  try {
    await reloadConfig()
    message.success('配置已重新加载')
  } catch (e: any) {
    message.error('重载失败')
  } finally {
    configLoading.value = false
  }
}

async function handleCreateBackup() {
  try {
    await createBackup()
    message.success('备份已创建')
  } catch (e: any) {
    message.error(e.message || '备份失败')
  }
}

async function loadAudioConfig() {
  audioLoading.value = true
  try {
    const res = await getAudioAlerts()
    audioConfig.value = res
  } catch { /* silent */ }
  finally { audioLoading.value = false }
}

async function saveAudioConfig() {
  try {
    await updateAudioAlerts(audioConfig.value)
    message.success('音频配置已保存')
  } catch { message.error('保存失败') }
}

onMounted(() => {
  loadAudioConfig()
})
</script>

<template>
  <div>
    <Card title="系统配置" style="margin-bottom: 16px">
      <Space>
        <Button type="primary" :loading="configLoading" @click="handleReloadConfig">重新加载配置</Button>
      </Space>
      <p style="color: #666; margin-top: 12px; font-size: 13px">
        重新从 configs/default.yaml 加载配置。修改配置文件后点击此按钮生效。
      </p>
    </Card>

    <Card title="数据备份" style="margin-bottom: 16px">
      <p style="color: #888; margin-bottom: 16px">
        备份内容：SQLite 数据库、配置文件。
      </p>
      <Button type="primary" @click="handleCreateBackup">立即备份</Button>
    </Card>

    <Card title="告警音频配置" style="max-width: 600px">
      <div v-for="(sev, key) in { low: '低级 (LOW)', medium: '中级 (MEDIUM)', high: '高级 (HIGH)' }" :key="key" style="margin-bottom: 16px; padding: 12px; border: 1px solid #2d2d4a; border-radius: 6px">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px">
          <Typography.Text strong>{{ sev }}</Typography.Text>
          <Switch
            v-if="audioConfig[key]"
            v-model:checked="audioConfig[key].enabled"
            checked-children="开"
            un-checked-children="关"
            size="small"
          />
        </div>
        <Space v-if="audioConfig[key]?.enabled">
          <Select v-model:value="audioConfig[key].sound" style="width: 200px" placeholder="声音">
            <Select.Option value="beep_single">单声提示</Select.Option>
            <Select.Option value="beep_double">双声提示</Select.Option>
            <Select.Option value="beep_double_voice">双声 + 语音播报</Select.Option>
          </Select>
          <Input
            v-if="key === 'high'"
            v-model:value="audioConfig[key].voice_template"
            placeholder="语音模板, 如: {camera} 高级别告警"
            style="width: 250px"
          />
        </Space>
      </div>
      <Button type="primary" @click="saveAudioConfig">保存音频配置</Button>
    </Card>
  </div>
</template>
