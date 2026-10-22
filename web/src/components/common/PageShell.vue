<script setup lang="ts">
import { useId } from 'vue'

defineOptions({ inheritAttrs: false })

withDefaults(defineProps<{
  as?: 'main' | 'section' | 'div'
  title?: string
  description?: string
  busy?: boolean
  compact?: boolean
}>(), {
  as: 'main',
  title: '',
  description: '',
  busy: false,
  compact: false,
})

const titleId = `page-title-${useId()}`
</script>

<template>
  <component
    :is="as"
    v-bind="$attrs"
    class="page-shell glass"
    :class="{ 'page-shell--compact': compact }"
    :aria-labelledby="title ? titleId : undefined"
    :aria-busy="busy"
  >
    <header v-if="title || $slots.title || $slots.actions || description" class="page-shell__header">
      <div class="page-shell__heading">
        <slot name="title">
          <h1 :id="titleId" class="page-shell__title">{{ title }}</h1>
        </slot>
        <p v-if="description" class="page-shell__description">{{ description }}</p>
        <slot name="description" />
      </div>
      <div v-if="$slots.actions" class="page-shell__actions">
        <slot name="actions" />
      </div>
    </header>

    <div v-if="$slots.meta" class="page-shell__meta">
      <slot name="meta" />
    </div>

    <div class="page-shell__content">
      <slot />
    </div>
  </component>
</template>

<style scoped>
.page-shell {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  padding: 24px;
  border-radius: var(--r-lg, 12px);
}

.page-shell--compact {
  padding: 16px;
}

.page-shell__header {
  display: flex;
  flex: 0 0 auto;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 20px;
}

.page-shell__heading {
  min-width: 0;
}

.page-shell__title {
  margin: 0;
  color: var(--ink);
  font-size: 24px;
  font-weight: 700;
  line-height: 1.3;
  letter-spacing: -0.02em;
}

.page-shell__description {
  max-width: 760px;
  margin: 6px 0 0;
  color: var(--ink-4);
  font-size: 13px;
  line-height: 1.6;
}

.page-shell__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

.page-shell__meta {
  flex: 0 0 auto;
  margin-bottom: 12px;
}

.page-shell__content {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
}

@media (max-width: 1279px) {
  .page-shell {
    padding: 20px;
  }
}

@media (max-width: 767px) {
  .page-shell,
  .page-shell--compact {
    padding: 12px;
    border-radius: var(--r-md, 8px);
  }

  .page-shell__header {
    flex-direction: column;
    gap: 12px;
    margin-bottom: 14px;
  }

  .page-shell__title {
    font-size: 20px;
  }

  .page-shell__actions {
    width: 100%;
    justify-content: flex-start;
  }
}
</style>
