import { defineComponent, h } from 'vue'
import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import PageShell from '../PageShell.vue'
import AsyncState from '../AsyncState.vue'
import DangerousActionConfirm from '../DangerousActionConfirm.vue'
import ResponsiveTable from '../ResponsiveTable.vue'

const ResultStub = defineComponent({
  name: 'Result',
  props: ['status', 'title', 'subTitle'],
  template: '<section :data-status="status"><h2>{{ title }}</h2><p>{{ subTitle }}</p><slot name="extra" /></section>',
})

const ButtonStub = defineComponent({
  name: 'Button',
  emits: ['click'],
  template: '<button type="button" v-bind="$attrs" @click="$emit(\'click\')"><slot name="icon" /><slot /></button>',
})

describe('shared page state components', () => {
  it('gives the page a stable accessible title and busy state', () => {
    const wrapper = mount(PageShell, {
      props: { title: '摄像头', description: '运行状态', busy: true },
      slots: { default: '<div>content</div>' },
    })

    const page = wrapper.get('main')
    const heading = wrapper.get('h1')
    expect(page.attributes('aria-labelledby')).toBe(heading.attributes('id'))
    expect(page.attributes('aria-busy')).toBe('true')
    expect(wrapper.text()).toContain('运行状态')
  })

  it('never renders a failed query as an empty state', async () => {
    const wrapper = mount(AsyncState, {
      props: {
        error: '数据库暂时不可用',
        empty: true,
      },
      global: {
        stubs: {
          Result: ResultStub,
          Button: ButtonStub,
          ContentSkeleton: { template: '<div data-test="skeleton" />' },
          EmptyState: {
            props: ['title'],
            template: '<div data-test="empty">{{ title }}</div>',
          },
        },
      },
    })

    expect(wrapper.attributes('data-state')).toBe('error')
    expect(wrapper.text()).toContain('数据库暂时不可用')
    expect(wrapper.find('[data-test="empty"]').exists()).toBe(false)
    expect(wrapper.get('button').attributes('aria-label')).toBe('重试加载数据')
    await wrapper.get('button').trigger('click')
    expect(wrapper.emitted('retry')).toHaveLength(1)

    await wrapper.setProps({ error: '', empty: true })
    expect(wrapper.attributes('data-state')).toBe('empty')
    expect(wrapper.find('[data-test="empty"]').exists()).toBe(true)
  })

  it('distinguishes forbidden, not-found, and initial loading states', async () => {
    const wrapper = mount(AsyncState, {
      props: {
        forbidden: true,
        forbiddenDescription: '仅值班管理员可查看摄像头',
        notFoundDescription: '摄像头配置已删除',
      },
      global: {
        stubs: {
          Result: ResultStub,
          Button: ButtonStub,
          ContentSkeleton: { template: '<div data-test="skeleton" />' },
        },
      },
    })

    expect(wrapper.attributes('data-state')).toBe('forbidden')
    expect(wrapper.text()).toContain('无权访问')
    expect(wrapper.text()).toContain('仅值班管理员可查看摄像头')
    expect(wrapper.get('button[aria-label="重试加载数据"]')).toBeTruthy()

    await wrapper.setProps({ forbidden: false, notFound: true })
    expect(wrapper.attributes('data-state')).toBe('not-found')
    expect(wrapper.text()).toContain('内容不存在')
    expect(wrapper.text()).toContain('摄像头配置已删除')
    await wrapper.get('button[aria-label="重试加载数据"]').trigger('click')
    expect(wrapper.emitted('retry')).toHaveLength(1)

    await wrapper.setProps({ notFound: false, loading: true })
    expect(wrapper.attributes('data-state')).toBe('loading')
    expect(wrapper.find('[data-test="skeleton"]').exists()).toBe(true)
  })

  it('requires confirmation before emitting a dangerous action', async () => {
    const PopconfirmStub = defineComponent({
      name: 'Popconfirm',
      emits: ['confirm'],
      template: '<div><slot /><button type="button" data-test="confirm" @click="$emit(\'confirm\')">confirm</button></div>',
    })
    const wrapper = mount(DangerousActionConfirm, {
      props: {
        title: '确认删除',
        actionLabel: '删除',
        ariaLabel: '删除摄像头 cam-1',
      },
      global: {
        stubs: {
          Popconfirm: PopconfirmStub,
          APopconfirm: PopconfirmStub,
          Button: ButtonStub,
          AButton: ButtonStub,
        },
      },
    })

    expect(wrapper.get('button[aria-label="删除摄像头 cam-1"]')).toBeTruthy()
    expect(wrapper.emitted('confirm')).toBeUndefined()
    await wrapper.get('[data-test="confirm"]').trigger('click')
    expect(wrapper.emitted('confirm')).toHaveLength(1)
  })

  it('adds an accessible scrolling region and a minimum table width', () => {
    Object.defineProperty(window, 'matchMedia', {
      configurable: true,
      writable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    })
    const TableStub = defineComponent({
      name: 'Table',
      props: ['columns', 'dataSource', 'scroll'],
      template: '<div data-test="table"><slot name="bodyCell" :column="columns[0]" :record="dataSource[0]" /></div>',
    })
    const wrapper = mount(ResponsiveTable, {
      props: {
        columns: [{ key: 'name', title: '名称' }],
        dataSource: [{ name: 'cam-1' }],
        minWidth: 960,
        accessibleLabel: '摄像头表格',
      },
      slots: {
        bodyCell: ({ record }: { record: { name: string } }) => h('span', record.name),
      },
      global: { stubs: { Table: TableStub, ATable: TableStub } },
    })

    expect(wrapper.get('[role="region"]').attributes('aria-label')).toBe('摄像头表格')
    expect(wrapper.getComponent(TableStub).props('scroll')).toEqual({ x: 960 })
    expect(wrapper.text()).toContain('cam-1')
  })
})
