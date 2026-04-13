import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { theme } from 'ant-design-vue'

export type ThemeMode = 'dark' | 'light'

const STORAGE_KEY = 'argus-theme'

const DARK_TOKENS = {
  colorPrimary: '#ffffff', // 极致极简白
  colorBgContainer: '#121217', // 高级墨黑
  colorBgElevated: '#1a1a24',
  colorBgLayout: '#0a0a0c', 
  colorTextBase: '#f1f5f9',
  borderRadius: 8,
  fontSize: 14,
  wireframe: false,
}

const LIGHT_TOKENS = {
  colorPrimary: '#0a0a0c', // SaaS极简黑
  colorBgContainer: '#ffffff',
  colorBgElevated: '#ffffff',
  colorBgLayout: '#f6f6f8',
  borderRadius: 8,
  fontSize: 14,
  wireframe: false,
}

function detectSystemTheme(): ThemeMode {
  return 'light' // Lock to light for Glassmorphism UI
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
