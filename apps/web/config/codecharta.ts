export function resolveCodechartaUrl(env: Record<string, string | undefined>): string {
  return env.VITE_CODECHARTA_URL || `http://localhost:${env.CODECHARTA_PORT || '9001'}`
}
