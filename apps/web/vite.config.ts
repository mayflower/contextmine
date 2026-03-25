/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// API URL for proxying - defaults to localhost for local dev
const apiUrl = process.env.VITE_API_URL || 'http://localhost:8000'
const codechartaUrl = process.env.VITE_CODECHARTA_URL || 'http://localhost:9000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      reportsDirectory: 'coverage',
      include: ['src/**/*.{ts,tsx}'],
      exclude: ['src/**/*.test.{ts,tsx}', 'src/test/**'],
    },
  },
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
      '/codecharta': {
        target: codechartaUrl,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/codecharta/, ''),
      },
    },
  },
})
