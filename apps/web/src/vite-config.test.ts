import { describe, expect, it } from 'vitest'

import { resolveCodechartaUrl } from '../config/codecharta'

describe('resolveCodechartaUrl', () => {
  it('uses the Docker Compose default port', () => {
    expect(resolveCodechartaUrl({})).toBe('http://localhost:9001')
  })

  it('uses the configured Docker Compose host port', () => {
    expect(resolveCodechartaUrl({ CODECHARTA_PORT: '9002' })).toBe('http://localhost:9002')
  })

  it('prefers an explicit proxy target', () => {
    expect(
      resolveCodechartaUrl({
        CODECHARTA_PORT: '9002',
        VITE_CODECHARTA_URL: 'https://codecharta.example.com',
      }),
    ).toBe('https://codecharta.example.com')
  })
})
