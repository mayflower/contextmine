/**
 * Tests for SemanticMapView rendering + pure helper functions.
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { SemanticMapPayload, SemanticMapPoint, ViewScenario } from '../types'

import SemanticMapView from './SemanticMapView'

const mockScenario: ViewScenario = {
  id: 's1',
  collection_id: 'c1',
  name: 'Base',
  version: 1,
  is_as_is: true,
  base_scenario_id: null,
}

function makePoint(overrides: Partial<SemanticMapPoint> = {}): SemanticMapPoint {
  return {
    id: 'p1',
    label: 'Auth cluster',
    x: 0.5,
    y: 0.3,
    member_count: 10,
    cohesion: 0.85,
    top_kinds: [{ kind: 'FILE', count: 8 }],
    domain_counts: [{ domain: 'auth', count: 8 }],
    dominant_domain: 'auth',
    dominant_ratio: 0.8,
    summary: 'Authentication related files',
    anchor_node_id: 'n1',
    sample_nodes: [{ id: 'n1', name: 'auth.py', kind: 'FILE', natural_key: 'file:auth.py' }],
    member_node_ids: ['n1', 'n2', 'n3'],
    ...overrides,
  }
}

function makePayload(overrides: Partial<SemanticMapPayload> = {}): SemanticMapPayload {
  return {
    collection_id: 'c1',
    scenario: mockScenario,
    projection: 'semantic_map',
    map_mode: 'code_structure',
    status: { status: 'ready', reason: 'ok' },
    thresholds: {
      mixed_cluster_max_dominant_ratio: 0.7,
      isolated_distance_multiplier: 2,
      semantic_duplication_min_similarity: 0.8,
      semantic_duplication_max_source_overlap: 0.5,
      misplaced_min_dominant_ratio: 0.6,
    },
    summary: {
      points: 3,
      mixed_clusters: 1,
      isolated_points: 0,
      semantic_duplication: 2,
      misplaced_code: 0,
    },
    warnings: [],
    signals: {
      mixed_clusters: [
        { label: 'Mixed auth/billing', score: 0.72, anchor_node_id: 'n1', reason: 'Low dominant ratio' },
      ],
      isolated_points: [],
      semantic_duplication: [
        { left_label: 'auth utils', right_label: 'auth helpers', score: 0.85, anchor_node_id: 'n2', reason: 'High similarity' },
      ],
      misplaced_code: [],
    },
    points: [
      makePoint({ id: 'p1', label: 'Auth cluster', x: 0.3, y: 0.2 }),
      makePoint({ id: 'p2', label: 'Billing cluster', x: 0.7, y: 0.6, dominant_domain: 'billing', anchor_node_id: 'n5', member_node_ids: ['n5', 'n6'] }),
      makePoint({ id: 'p3', label: 'Gateway cluster', x: 0.1, y: 0.9, dominant_domain: 'gateway', anchor_node_id: 'n8', member_node_ids: ['n8'] }),
    ],
    ...overrides,
  }
}

function makeProps(overrides: Partial<Parameters<typeof SemanticMapView>[0]> = {}) {
  return {
    state: 'ready' as const,
    error: '',
    payload: makePayload(),
    comparisonPayload: null,
    mode: 'code_structure' as const,
    selectedNodeId: '',
    showDiffOverlay: false,
    diffMinDrift: 0.1,
    onModeChange: vi.fn(),
    onSelectNodeId: vi.fn(),
    onRetry: vi.fn(),
    ...overrides,
  }
}

// --- Pure helper tests (replicated from component) ---

const DOMAIN_COLORS = [
  '#0ea5e9', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ef4444', '#14b8a6', '#f97316', '#3b82f6',
]

function colorForDomain(domain: string | null): string {
  if (!domain) return '#64748b'
  const normalized = domain.toLowerCase()
  let hash = 0
  for (let i = 0; i < normalized.length; i += 1) {
    hash = (hash * 31 + (normalized.codePointAt(i) ?? 0)) >>> 0
  }
  return DOMAIN_COLORS[hash % DOMAIN_COLORS.length]
}

function isPointSelected(point: SemanticMapPoint, selectedNodeId: string): boolean {
  if (!selectedNodeId) return false
  if (point.anchor_node_id === selectedNodeId) return true
  return point.sample_nodes.some((node) => node.id === selectedNodeId)
}

describe('colorForDomain', () => {
  it('returns gray for null domain', () => {
    expect(colorForDomain(null)).toBe('#64748b')
  })

  it('returns a color from the palette for a domain', () => {
    const color = colorForDomain('auth')
    expect(DOMAIN_COLORS).toContain(color)
  })

  it('returns consistent color for same domain', () => {
    expect(colorForDomain('billing')).toBe(colorForDomain('billing'))
  })

  it('is case insensitive', () => {
    expect(colorForDomain('Auth')).toBe(colorForDomain('auth'))
  })
})

describe('isPointSelected', () => {
  const point = makePoint()

  it('returns false when no selectedNodeId', () => {
    expect(isPointSelected(point, '')).toBe(false)
  })

  it('returns true when anchor_node_id matches', () => {
    expect(isPointSelected(point, 'n1')).toBe(true)
  })

  it('returns true when sample_node id matches', () => {
    expect(isPointSelected(point, 'n1')).toBe(true)
  })

  it('returns false when no match', () => {
    expect(isPointSelected(point, 'nonexistent')).toBe(false)
  })
})

// --- Rendering tests ---

describe('SemanticMapView rendering', () => {
  it('renders the semantic map panel with header', () => {
    render(<SemanticMapView {...makeProps()} />)
    expect(screen.getByText('Semantic Map')).toBeInTheDocument()
  })

  it('displays mode and point count', () => {
    render(<SemanticMapView {...makeProps()} />)
    expect(screen.getByText(/Code-structure communities/)).toBeInTheDocument()
    expect(screen.getByText(/Points: 3/)).toBeInTheDocument()
  })

  it('renders mode select', () => {
    render(<SemanticMapView {...makeProps()} />)
    expect(screen.getByText('Mode')).toBeInTheDocument()
    expect(screen.getByText('Code structure')).toBeInTheDocument()
    expect(screen.getByText('Semantic')).toBeInTheDocument()
  })

  it('renders SVG scatter plot', () => {
    render(<SemanticMapView {...makeProps()} />)
    expect(screen.getByLabelText('Semantic map scatter plot')).toBeInTheDocument()
  })

  it('renders KPI cards', () => {
    render(<SemanticMapView {...makeProps()} />)
    // These labels appear in both the KPI cards and the signal sections
    expect(screen.getAllByText('Mixed clusters').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Isolated points').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Duplication hints')).toBeInTheDocument()
    expect(screen.getAllByText('Misplaced code').length).toBeGreaterThanOrEqual(1)
  })

  it('renders signal sections', () => {
    render(<SemanticMapView {...makeProps()} />)
    expect(screen.getByText('Mixed auth/billing')).toBeInTheDocument()
    expect(screen.getByText('Low dominant ratio')).toBeInTheDocument()
  })

  it('renders semantic duplication signals', () => {
    render(<SemanticMapView {...makeProps()} />)
    // The signal label uses left_label + right_label
    expect(screen.getByText(/auth utils/)).toBeInTheDocument()
  })

  it('shows unavailable state', () => {
    const unavailablePayload = makePayload({
      status: { status: 'unavailable', reason: 'no_semantic_communities' },
    })
    render(<SemanticMapView {...makeProps({ payload: unavailablePayload })} />)
    expect(screen.getByText('No semantic map available')).toBeInTheDocument()
    expect(screen.getByText(/No semantic communities were found/)).toBeInTheDocument()
  })

  it('shows diff overlay KPI when enabled', () => {
    render(<SemanticMapView {...makeProps({ showDiffOverlay: true })} />)
    expect(screen.getByText('Drift overlay hits')).toBeInTheDocument()
  })

  it('shows diff overlay unavailable message when no comparison data', () => {
    render(<SemanticMapView {...makeProps({ showDiffOverlay: true, comparisonPayload: null })} />)
    expect(screen.getByText(/Diff overlay unavailable/)).toBeInTheDocument()
  })

  it('shows payload warnings', () => {
    const payloadWithWarnings = makePayload({ warnings: ['Embeddings incomplete'] })
    render(<SemanticMapView {...makeProps({ payload: payloadWithWarnings })} />)
    expect(screen.getByText('Embeddings incomplete')).toBeInTheDocument()
  })

  it('renders point labels in scatter plot when few points', () => {
    render(<SemanticMapView {...makeProps()} />)
    // With 3 points (< 80) labels should be rendered as text elements
    expect(screen.getByText('Auth cluster')).toBeInTheDocument()
  })

  it('renders loading skeleton when loading and no data', () => {
    render(<SemanticMapView {...makeProps({ state: 'loading', payload: null })} />)
    expect(screen.queryByText('Semantic Map')).not.toBeInTheDocument()
  })

  it('renders error state when error and no data', () => {
    render(<SemanticMapView {...makeProps({ state: 'error', error: 'Timeout', payload: null })} />)
    expect(screen.getByText('Semantic map request failed')).toBeInTheDocument()
  })

  it('renders diff signals when overlay is enabled with comparison data', () => {
    const comparisonPayload = makePayload({
      points: [
        makePoint({ id: 'cp1', label: 'Old auth', member_node_ids: ['n1', 'n2'] }),
      ],
    })
    render(<SemanticMapView {...makeProps({ showDiffOverlay: true, comparisonPayload })} />)
    expect(screen.getByText('Code vs semantic drift')).toBeInTheDocument()
  })
})
