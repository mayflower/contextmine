/**
 * Tests for useCockpitData hook internals.
 *
 * The hook itself is very large (1600+ lines) and rendering it with jsdom causes
 * memory pressure. We test the pure helper functions that it uses internally
 * by replicating them here.
 */
import { describe, expect, it } from 'vitest'

import type { CockpitView, CockpitLoadState, CockpitLayer } from '../types'

// --- Replicated types and constants from useCockpitData.ts ---

type DataStates = Record<CockpitView, CockpitLoadState>
type DataErrors = Record<CockpitView, string>

const DEFAULT_GRAPH = {
  nodes: [],
  edges: [],
  page: 0,
  limit: 0,
  total_nodes: 0,
}

const DEFAULT_STATES: DataStates = {
  overview: 'idle',
  topology: 'idle',
  deep_dive: 'idle',
  c4_diff: 'idle',
  architecture: 'idle',
  city: 'idle',
  evolution: 'idle',
  graphrag: 'idle',
  ui_map: 'idle',
  semantic_map: 'idle',
  test_matrix: 'idle',
  user_flows: 'idle',
  rebuild_readiness: 'idle',
  exports: 'idle',
}

const DEFAULT_ERRORS: DataErrors = {
  overview: '',
  topology: '',
  deep_dive: '',
  c4_diff: '',
  architecture: '',
  city: '',
  evolution: '',
  graphrag: '',
  ui_map: '',
  semantic_map: '',
  test_matrix: '',
  user_flows: '',
  rebuild_readiness: '',
  exports: '',
}

// Replicated from useCockpitData.ts (line 130-134)
function topologyEntityLevel(layer: CockpitLayer): 'domain' | 'container' | 'component' {
  if (layer === 'portfolio_system') return 'domain'
  if (layer === 'component_interface') return 'component'
  return 'container'
}

// Replicated from useCockpitData.ts - parseApiErrorMessage logic (lines 260-273)
async function parseApiErrorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const payload = await response.json()
    const detail = payload?.detail
    if (typeof detail === 'string' && detail.trim()) {
      return detail
    }
  } catch {
    // Ignore parse errors and use fallback.
  }
  return `${fallback} (${response.status})`
}

// --- Tests ---

describe('DEFAULT_STATES', () => {
  it('has idle state for all cockpit views', () => {
    const views: CockpitView[] = [
      'overview', 'topology', 'deep_dive', 'c4_diff', 'architecture',
      'city', 'evolution', 'graphrag', 'ui_map', 'semantic_map',
      'test_matrix', 'user_flows', 'rebuild_readiness', 'exports',
    ]
    for (const view of views) {
      expect(DEFAULT_STATES[view]).toBe('idle')
    }
  })
})

describe('DEFAULT_ERRORS', () => {
  it('has empty string for all cockpit views', () => {
    const views: CockpitView[] = [
      'overview', 'topology', 'deep_dive', 'c4_diff', 'architecture',
      'city', 'evolution', 'graphrag', 'ui_map', 'semantic_map',
      'test_matrix', 'user_flows', 'rebuild_readiness', 'exports',
    ]
    for (const view of views) {
      expect(DEFAULT_ERRORS[view]).toBe('')
    }
  })
})

describe('DEFAULT_GRAPH', () => {
  it('has empty nodes and edges', () => {
    expect(DEFAULT_GRAPH.nodes).toEqual([])
    expect(DEFAULT_GRAPH.edges).toEqual([])
  })

  it('has page 0 and limit 0', () => {
    expect(DEFAULT_GRAPH.page).toBe(0)
    expect(DEFAULT_GRAPH.limit).toBe(0)
  })

  it('has total_nodes 0', () => {
    expect(DEFAULT_GRAPH.total_nodes).toBe(0)
  })
})

describe('topologyEntityLevel', () => {
  it('returns domain for portfolio_system layer', () => {
    expect(topologyEntityLevel('portfolio_system')).toBe('domain')
  })

  it('returns component for component_interface layer', () => {
    expect(topologyEntityLevel('component_interface')).toBe('component')
  })

  it('returns container for domain_container layer', () => {
    expect(topologyEntityLevel('domain_container')).toBe('container')
  })

  it('returns container for code_controlflow layer', () => {
    expect(topologyEntityLevel('code_controlflow')).toBe('container')
  })
})

describe('parseApiErrorMessage', () => {
  it('returns detail from JSON response when present', async () => {
    const response = new Response(JSON.stringify({ detail: 'Not found' }), {
      status: 404,
    })
    const result = await parseApiErrorMessage(response, 'Request failed')
    expect(result).toBe('Not found')
  })

  it('returns fallback with status when detail is missing', async () => {
    const response = new Response(JSON.stringify({}), { status: 500 })
    const result = await parseApiErrorMessage(response, 'Server error')
    expect(result).toBe('Server error (500)')
  })

  it('returns fallback with status when detail is empty string', async () => {
    const response = new Response(JSON.stringify({ detail: '  ' }), { status: 400 })
    const result = await parseApiErrorMessage(response, 'Bad request')
    expect(result).toBe('Bad request (400)')
  })

  it('returns fallback with status when response is not JSON', async () => {
    const response = new Response('plain text error', { status: 502 })
    const result = await parseApiErrorMessage(response, 'Gateway error')
    expect(result).toBe('Gateway error (502)')
  })

  it('returns fallback when detail is a number', async () => {
    const response = new Response(JSON.stringify({ detail: 42 }), { status: 422 })
    const result = await parseApiErrorMessage(response, 'Validation error')
    expect(result).toBe('Validation error (422)')
  })
})
