// vite.config.ts

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const backendHost = '127.0.0.1'

export default defineConfig({
  plugins: [vue()],
  build: {
    // Vendor libraries are intentionally split into cacheable chunks. The
    // largest one is Ant Design Vue, so keep the warning limit above that
    // known vendor size instead of reporting it as an application chunk issue.
    chunkSizeWarningLimit: 1600,
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
