import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError, apiData } from './client'

afterEach(() => vi.unstubAllGlobals())

describe('API client', () => {
  it.each([
    [401, 'AuthenticationError'],
    [403, 'AuthorizationError'],
  ])('turns HTTP %i into a typed authorization error', async (status, name) => {
    const response = new Response(JSON.stringify({ detail: 'denied' }), { status })
    await expect(apiData(Promise.resolve({ response, error: { detail: 'denied' } }))).rejects.toMatchObject<ApiError>({
      name,
      status,
      detail: { detail: 'denied' },
    })
  })

  it('uses browser credentials and forwards AbortSignal cancellation', async () => {
    const fetchMock = vi.fn((request: Request) => new Promise<Response>((_resolve, reject) => {
      expect(request.credentials).toBe('include')
      request.signal.addEventListener('abort', () => reject(request.signal.reason), { once: true })
    }))
    vi.stubGlobal('fetch', fetchMock)
    vi.resetModules()
    const { api } = await import('./client')
    const controller = new AbortController()
    const request = api.GET('/api/health', { signal: controller.signal })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledOnce())
    controller.abort(new DOMException('cancelled', 'AbortError'))
    await expect(request).rejects.toMatchObject({ name: 'AbortError' })
  })
})
