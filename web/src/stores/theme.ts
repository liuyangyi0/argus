import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { theme } from 'ant-design-vue'

export type ThemeMode = 'dark' | 'light'

const STORAGE_KEY = 'argus-theme'

const LIGHT_TOKENS = {
  colorPrimary: '#0a0a0c', // SaaS极简黑
  colorBgContainer: '#ffffff',
  colorBgElevated: '#ffffff',
  colorBgLayout: '#f6f6f8',
  borderRadius: 8,
  fontSize: 14,
  wireframe: false,
}

function applyLightTheme(): void {
  if (typeof document !== 'undefined') {
    document.documentElement.classList.add('light-theme')
    document.documentElement.classList.remove('dark-theme')
    document.documentElement.style.colorScheme = 'light'
  }
  if (typeof localStorage !== 'undefined') localStorage.setItem(STORAGE_KEY, 'light')
}

export const useThemeStore = defineStore('theme', () => {
  // Dark mode is intentionally suspended until every non-Ant surface is
  // tokenized. Coerce stale persisted preferences to light to prevent a mixed
  // dark-Ant/light-page UI after upgrades.
  const mode = ref<ThemeMode>('light')

  const isDark = computed(() => false)

  const algorithm = computed(() => theme.defaultAlgorithm)

  const tokens = computed(() => LIGHT_TOKENS)

  const themeConfig = computed(() => ({
    algorithm: algorithm.value,
    token: tokens.value,
  }))

  function toggle() {
    mode.value = 'light'
    applyLightTheme()
  }

  function setTheme(_requested: ThemeMode) {
    mode.value = 'light'
    applyLightTheme()
  }

  applyLightTheme()

  return { mode, isDark, algorithm, tokens, themeConfig, toggle, setTheme }
})
