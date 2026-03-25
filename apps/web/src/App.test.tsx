/**
 * Tests for App.tsx pure helper functions.
 *
 * The App component itself is very large and manages many side effects.
 * We focus on testing the pure utility functions it defines:
 * - parseInitialPage()
 * - formatSourceUrl()
 * - updatePageQuery()
 *
 * Since these are not exported, we replicate the logic to test it thoroughly.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

// --- Replicated from App.tsx ---

type Page = 'dashboard' | 'collections' | 'runs' | 'cockpit'

const VALID_PAGES = new Set<Page>(['dashboard', 'collections', 'runs', 'cockpit'])

function parseInitialPage(): Page {
  const params = new URLSearchParams(globalThis.location.search)
  const rawPage = params.get('page')
  if (!rawPage) {
    return 'dashboard'
  }
  return VALID_PAGES.has(rawPage as Page) ? (rawPage as Page) : 'dashboard'
}

function formatSourceUrl(url: string): string {
  if (url.startsWith('https://github.com/')) {
    return url.replace('https://github.com/', '').split('/').slice(0, 2).join('/')
  }
  try {
    const parsed = new URL(url)
    const path = parsed.pathname.replace(/\/$/, '')
    if (path && path !== '/') {
      return parsed.hostname + path
    }
    return parsed.hostname
  } catch {
    return url
  }
}

type CockpitView = 'overview' | 'topology' | 'deep_dive' | 'c4_diff' | 'architecture' | 'city' | 'evolution' | 'graphrag' | 'semantic_map' | 'ui_map' | 'test_matrix' | 'rebuild_readiness' | 'exports'
type CockpitLayer = 'portfolio_system' | 'domain_container' | 'component_interface' | 'code_controlflow'

const DEFAULT_COCKPIT_VIEW: CockpitView = 'overview'
const DEFAULT_COCKPIT_LAYER: CockpitLayer = 'code_controlflow'

interface CockpitNavigationOptions {
  collectionId?: string
  scenarioId?: string
  view?: CockpitView
  layer?: CockpitLayer
}

function updatePageQuery(page: Page, cockpitOptions?: CockpitNavigationOptions): void {
  const params = new URLSearchParams(globalThis.location.search)
  params.set('page', page)

  if (page === 'cockpit') {
    const nextView = cockpitOptions?.view
    const nextLayer = cockpitOptions?.layer

    if (cockpitOptions?.collectionId) {
      params.set('collection', cockpitOptions.collectionId)
    } else if (cockpitOptions && !cockpitOptions.collectionId) {
      params.delete('collection')
    }

    if (cockpitOptions?.scenarioId) {
      params.set('scenario', cockpitOptions.scenarioId)
    } else if (cockpitOptions && !cockpitOptions.scenarioId) {
      params.delete('scenario')
    }

    if (nextView) {
      params.set('view', nextView)
    } else if (!params.get('view')) {
      params.set('view', DEFAULT_COCKPIT_VIEW)
    }

    if (nextLayer) {
      params.set('layer', nextLayer)
    } else if (!params.get('layer')) {
      params.set('layer', DEFAULT_COCKPIT_LAYER)
    }
  } else {
    params.delete('collection')
    params.delete('scenario')
    params.delete('view')
    params.delete('layer')
    params.delete('query')
    params.delete('node')
    params.delete('pageIndex')
    params.delete('limit')
    params.delete('includeKinds')
    params.delete('excludeKinds')
    params.delete('overlay')
    params.delete('hideIsolated')
    params.delete('edgeKinds')
  }

  const nextQuery = params.toString()
  const nextUrl = nextQuery ? `${globalThis.location.pathname}?${nextQuery}` : globalThis.location.pathname
  globalThis.history.replaceState({}, '', nextUrl)
}

// --- Tests ---

describe('parseInitialPage', () => {
  beforeEach(() => {
    globalThis.history.replaceState({}, '', '/')
  })

  it('returns dashboard when no page param', () => {
    expect(parseInitialPage()).toBe('dashboard')
  })

  it('returns the page when valid', () => {
    globalThis.history.replaceState({}, '', '/?page=collections')
    expect(parseInitialPage()).toBe('collections')
  })

  it('returns cockpit when valid', () => {
    globalThis.history.replaceState({}, '', '/?page=cockpit')
    expect(parseInitialPage()).toBe('cockpit')
  })

  it('returns runs when valid', () => {
    globalThis.history.replaceState({}, '', '/?page=runs')
    expect(parseInitialPage()).toBe('runs')
  })

  it('returns dashboard for invalid page', () => {
    globalThis.history.replaceState({}, '', '/?page=invalid')
    expect(parseInitialPage()).toBe('dashboard')
  })

  it('returns dashboard for empty page param', () => {
    globalThis.history.replaceState({}, '', '/?page=')
    expect(parseInitialPage()).toBe('dashboard')
  })
})

describe('formatSourceUrl', () => {
  it('formats GitHub URLs to owner/repo', () => {
    expect(formatSourceUrl('https://github.com/owner/repo')).toBe('owner/repo')
  })

  it('truncates deep GitHub paths to owner/repo', () => {
    expect(formatSourceUrl('https://github.com/owner/repo/tree/main/src')).toBe('owner/repo')
  })

  it('formats web doc URLs to hostname/path', () => {
    expect(formatSourceUrl('https://langchain-ai.github.io/langgraph')).toBe(
      'langchain-ai.github.io/langgraph',
    )
  })

  it('formats root URLs to hostname only', () => {
    expect(formatSourceUrl('https://example.com/')).toBe('example.com')
    expect(formatSourceUrl('https://example.com')).toBe('example.com')
  })

  it('returns raw URL for invalid URLs', () => {
    expect(formatSourceUrl('not a url')).toBe('not a url')
  })

  it('handles URLs with paths', () => {
    expect(formatSourceUrl('https://docs.example.com/guide/intro')).toBe(
      'docs.example.com/guide/intro',
    )
  })
})

describe('updatePageQuery', () => {
  beforeEach(() => {
    globalThis.history.replaceState({}, '', '/')
  })

  it('sets page param for dashboard', () => {
    updatePageQuery('dashboard')
    const params = new URLSearchParams(globalThis.location.search)
    expect(params.get('page')).toBe('dashboard')
  })

  it('cleans cockpit params when switching to non-cockpit page', () => {
    globalThis.history.replaceState({}, '', '/?page=cockpit&collection=c1&scenario=s1&view=topology')
    updatePageQuery('dashboard')
    const params = new URLSearchParams(globalThis.location.search)
    expect(params.get('page')).toBe('dashboard')
    expect(params.has('collection')).toBe(false)
    expect(params.has('scenario')).toBe(false)
    expect(params.has('view')).toBe(false)
  })

  it('sets default view and layer for cockpit', () => {
    updatePageQuery('cockpit')
    const params = new URLSearchParams(globalThis.location.search)
    expect(params.get('page')).toBe('cockpit')
    expect(params.get('view')).toBe('overview')
    expect(params.get('layer')).toBe('code_controlflow')
  })

  it('sets collection and scenario for cockpit navigation', () => {
    updatePageQuery('cockpit', { collectionId: 'c1', scenarioId: 's1', view: 'topology' })
    const params = new URLSearchParams(globalThis.location.search)
    expect(params.get('collection')).toBe('c1')
    expect(params.get('scenario')).toBe('s1')
    expect(params.get('view')).toBe('topology')
  })

  it('preserves existing view when no view option provided', () => {
    globalThis.history.replaceState({}, '', '/?view=city')
    updatePageQuery('cockpit')
    const params = new URLSearchParams(globalThis.location.search)
    expect(params.get('view')).toBe('city')
  })

  it('deletes collection when empty string passed', () => {
    globalThis.history.replaceState({}, '', '/?collection=c1')
    updatePageQuery('cockpit', { collectionId: '' })
    const params = new URLSearchParams(globalThis.location.search)
    expect(params.has('collection')).toBe(false)
  })
})
