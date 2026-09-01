import createClient from 'openapi-fetch'

import type { paths } from './schema'

export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown

  constructor(response: Response, detail: unknown) {
    super(`API request failed with status ${response.status}`)
    this.name = response.status === 401
      ? 'AuthenticationError'
      : response.status === 403
        ? 'AuthorizationError'
        : 'ApiError'
    this.status = response.status
    this.detail = detail
  }
}

export const api = createClient<paths>({
  baseUrl: globalThis.location?.origin ?? '',
  credentials: 'include',
  headers: { Accept: 'application/json' },
})

interface ApiResult<T, E> {
  data?: T
  error?: E
  response: Response
}

export async function apiData<T, E>(request: Promise<ApiResult<T, E>>): Promise<T> {
  const result = await request
  if (!result.response.ok || result.data === undefined) {
    throw new ApiError(result.response, result.error)
  }
  return result.data
}

export function apiErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof ApiError && error.detail && typeof error.detail === 'object') {
    const detail = 'detail' in error.detail ? error.detail.detail : undefined
    if (typeof detail === 'string') return detail
  }
  return error instanceof Error ? error.message : fallback
}
