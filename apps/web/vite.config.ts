import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// API URL for proxying - defaults to localhost for local dev
const apiUrl = process.env.VITE_API_URL || 'http://localhost:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true, // Allow external connections (needed for Docker)
    proxy: {
      '/api': {
        target: apiUrl,
        changeOrigin: true,
      },
      '/mcp': {
        target: apiUrl,
        changeOrigin: true,
      },
    },
  },
})
