import { describe, expect, it, vi, beforeEach } from 'vitest'

// Mock the Grafana Faro modules before importing
vi.mock('@grafana/faro-web-sdk', () => ({
  getWebInstrumentations: vi.fn().mockReturnValue([]),
  initializeFaro: vi.fn().mockReturnValue({ api: {} }),
}))

vi.mock('@grafana/faro-web-tracing', () => ({
  TracingInstrumentation: vi.fn().mockImplementation(() => ({})),
}))

describe('faro module', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  it('exports getFaro function', async () => {
    const faro = await import('./faro')
    expect(typeof faro.getFaro).toBe('function')
  })

  it('exports initFaro function', async () => {
    const faro = await import('./faro')
    expect(typeof faro.initFaro).toBe('function')
  })

  it('getFaro returns null initially', async () => {
    const faro = await import('./faro')
    expect(faro.getFaro()).toBeNull()
  })

  it('initFaro returns null when faroUrl is not configured', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ faroUrl: null, version: '1.0.0' }), { status: 200 }),
    )

    const faro = await import('./faro')
    const result = await faro.initFaro()
    expect(result).toBeNull()

    vi.mocked(globalThis.fetch).mockRestore()
  })

  it('initFaro returns null and handles fetch failure gracefully', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('Network error'))

    const faro = await import('./faro')
    const result = await faro.initFaro()
    // When fetch fails, config defaults to { faroUrl: null, version: '0.0.0' }
    // so faro won't be initialized
    expect(result).toBeNull()

    vi.mocked(globalThis.fetch).mockRestore()
  })
})
