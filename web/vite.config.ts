// vite.config.ts

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const backendHost = '127.0.0.1'

export default defineConfig({
  plugins: [vue()],
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
