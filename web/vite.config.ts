// vite.config.ts

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const backendHost = '127.0.0.1'

export default defineConfig({
  plugins: [vue()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes('node_modules/ant-design-vue') || id.includes('node_modules/@ant-design')) {
            return 'antd'
          }
          if (id.includes('node_modules/echarts') || id.includes('node_modules/vue-echarts') || id.includes('node_modules/zrender')) {
            return 'echarts'
          }
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: `http://${backendHost}:8080`,
        changeOrigin: true,
      },
      '/ws': {
        target: `ws://${backendHost}:8080`,
        ws: true,
      },
    },
  },
})
