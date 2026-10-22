<script setup lang="ts">
import { computed, useAttrs } from 'vue'
import { Table } from 'ant-design-vue'
import type { TableProps } from 'ant-design-vue'

defineOptions({ inheritAttrs: false })

const props = withDefaults(defineProps<{
  columns?: TableProps['columns']
  dataSource?: TableProps['dataSource']
  rowKey?: TableProps['rowKey']
  loading?: TableProps['loading']
  pagination?: TableProps['pagination']
  scroll?: TableProps['scroll']
  minWidth?: number
  accessibleLabel?: string
}>(), {
  columns: () => [],
  dataSource: () => [],
  loading: false,
  minWidth: 760,
  accessibleLabel: '数据表格，可横向滚动',
})

const attrs = useAttrs()
const effectiveScroll = computed(() => ({ x: props.minWidth, ...(props.scroll ?? {}) }))
</script>

<template>
  <div
    class="responsive-table"
    role="region"
    :aria-label="accessibleLabel"
    tabindex="0"
  >
    <Table
      v-bind="attrs"
      :columns="columns"
      :data-source="dataSource"
      :row-key="rowKey"
      :loading="loading"
      :pagination="pagination"
      :scroll="effectiveScroll"
    >
      <template v-for="(_, slotName) in $slots" #[slotName]="slotProps">
        <slot :name="slotName" v-bind="slotProps ?? {}" />
      </template>
    </Table>
  </div>
</template>

<style scoped>
.responsive-table {
  min-width: 0;
  overflow-x: auto;
  border-radius: var(--r-sm, 6px);
  outline: none;
}

.responsive-table:focus-visible {
  box-shadow: 0 0 0 3px rgba(22, 119, 255, 0.22);
}

@media (max-width: 767px) {
  .responsive-table {
    margin-inline: -4px;
  }
}
</style>
