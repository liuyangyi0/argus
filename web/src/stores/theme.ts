import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { theme } from 'ant-design-vue'

export type ThemeMode = 'dark' | 'light'

const STORAGE_KEY = 'argus-theme'

const DARK_TOKENS = {
  colorPrimary: '#3b82f6',
  colorBgContainer: '#1a1a2e',
  colorBgElevated: '#1e1e36',
  colorBgLayout: '#0f0f1a',
  borderRadius: 6,
  fontSize: 14,
}

const LIGHT_TOKENS = {
  colorPrimary: '#2563eb',
  colorBgContainer: '#ffffff',
  colorBgElevated: '#f8fafc',
  colorBgLayout: '#f1f5f9',
  borderRadius: 6,
  fontSize: 14,
}

function detectSystemTheme(): ThemeMode {
  if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: light)').matches) {
    return 'light'
  }
  return 'dark'
}

function loadSaved(): ThemeMode | null {
  if (typeof localStorage === 'undefined') return null
  const v = localStorage.getItem(STORAGE_KEY)
  if (v === 'dark' || v === 'light') return v
  return null
}

export const useThemeStore = defineStore('theme', () => {
  const mode = ref<ThemeMode>(loadSaved() ?? detectSystemTheme())

  const isDark = computed(() => mode.value === 'dark')

  const algorithm = computed(() =>
    isDark.value ? theme.darkAlgorithm : theme.defaultAlgorithm
  )

  const tokens = computed(() => isDark.value ? DARK_TOKENS : LIGHT_TOKENS)

  const themeConfig = computed(() => ({
    algorithm: algorithm.value,
    token: tokens.value,
  }))

  function toggle() {
    mode.value = isDark.value ? 'light' : 'dark'
    localStorage.setItem(STORAGE_KEY, mode.value)
    // Update <html> class for non-Ant components
    document.documentElement.classList.toggle('light-theme', !isDark.value)
    document.documentElement.classList.toggle('dark-theme', isDark.value)
  }

  function setTheme(t: ThemeMode) {
    mode.value = t
    localStorage.setItem(STORAGE_KEY, t)
    document.documentElement.classList.toggle('light-theme', t === 'light')
    document.documentElement.classList.toggle('dark-theme', t === 'dark')
  }

  // Initialize HTML class on load
  if (typeof document !== 'undefined') {
    document.documentElement.classList.toggle('light-theme', !isDark.value)
    document.documentElement.classList.toggle('dark-theme', isDark.value)
  }

  return { mode, isDark, algorithm, tokens, themeConfig, toggle, setTheme }
})
