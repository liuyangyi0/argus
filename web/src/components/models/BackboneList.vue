<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Card, Table, Tag, Empty } from 'ant-design-vue'
import { getBackbones } from '../../api'

const backbones = ref<any[]>([])

async function loadBackbones() {
  try {
    const res = await getBackbones()
    backbones.value = res.backbones || []
  } catch (e) {
    console.error(e)
  }
}

const backboneColumns = [
  { title: '版本ID', dataIndex: 'backbone_version_id', key: 'id', ellipsis: true },
  { title: '类型', dataIndex: 'backbone_type', key: 'type', width: 140 },
  { title: '状态', key: 'status', width: 80 },
  { title: '创建时间', dataIndex: 'created_at', key: 'created_at', width: 170 },
]

defineExpose({ loadBackbones })

onMounted(loadBackbones)
</script>

<template>
  <Card :bordered="false">
    <Table
      :columns="backboneColumns"
      :data-source="backbones"
      row-key="backbone_version_id"
      size="small"
      :pagination="false"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'status'">
          <Tag :color="record.is_active ? 'green' : 'default'">
            {{ record.is_active ? '活跃' : '历史' }}
          </Tag>
        </template>
        <template v-else-if="column.key === 'created_at'">
          {{ record.created_at?.replace('T', ' ').slice(0, 19) }}
        </template>
      </template>
    </Table>
    <Empty v-if="backbones.length === 0" description="暂无骨干模型，使用预训练 DINOv2 权重" />
  </Card>
</template>
