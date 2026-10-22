import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useThemeStore } from '../theme'

describe('light-only theme safety policy', () => {
  beforeEach(() => {
    localStorage.setItem('argus-theme', 'dark')
    document.documentElement.classList.add('dark-theme')
    document.documentElement.classList.remove('light-theme')
    setActivePinia(createPinia())
  })

  it('coerces stale dark preferences to one consistent light theme', () => {
    const store = useThemeStore()

    expect(store.mode).toBe('light')
    expect(store.isDark).toBe(false)
    expect(localStorage.getItem('argus-theme')).toBe('light')
    expect(document.documentElement.classList.contains('light-theme')).toBe(true)
    expect(document.documentElement.classList.contains('dark-theme')).toBe(false)
    expect(document.documentElement.style.colorScheme).toBe('light')
  })

  it('keeps light mode when legacy callers request a toggle or dark theme', () => {
    const store = useThemeStore()
    store.toggle()
    store.setTheme('dark')

    expect(store.mode).toBe('light')
    expect(store.isDark).toBe(false)
    expect(document.documentElement.classList.contains('dark-theme')).toBe(false)
  })
})
