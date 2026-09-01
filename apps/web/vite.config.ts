/// <reference types="vitest" />
import { fileURLToPath } from 'node:url'

import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

import { resolveCodechartaUrl } from './config/codecharta.ts'

const repositoryRoot = fileURLToPath(new URL('../..', import.meta.url))

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, repositoryRoot, ['CODECHARTA_', 'VITE_'])
  const apiUrl = env.VITE_API_URL || 'http://localhost:8000'
  const codechartaUrl = resolveCodechartaUrl(env)

  return {
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
  }
})
