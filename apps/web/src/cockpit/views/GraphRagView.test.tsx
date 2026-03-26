/**
 * Tests for GraphRagView rendering + pure helper functions.
 */
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { TwinGraphResponse } from '../types'

vi.mock('cytoscape', () => ({
  default: vi.fn().mockReturnValue({
    on: vi.fn(),
    destroy: vi.fn(),
    fit: vi.fn(),
    layout: vi.fn().mockReturnValue({ run: vi.fn() }),
  }),
}))

vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn().mockResolvedValue({ svg: '<svg></svg>', bindFunctions: undefined }),
  },
}))

vi.mock('../utils/mermaidUtils', () => ({
  renderMermaidSvg: vi.fn(),
}))

import GraphRagView from './GraphRagView'

const emptyGraph: TwinGraphResponse = {
  nodes: [],
  edges: [],
  page: 0,
  limit: 1200,
  total_nodes: 0,
}

const populatedGraph: TwinGraphResponse = {
  nodes: [
    { id: 'n1', natural_key: 'file:main.py', kind: 'FILE', name: 'main.py', meta: { community_id: 'c1' } },
    { id: 'n2', natural_key: 'sym:authenticate', kind: 'SYMBOL', name: 'authenticate', meta: { community_id: 'c1' } },
    { id: 'n3', natural_key: 'rule:check_auth', kind: 'BUSINESS_RULE', name: 'check_auth', meta: { community_id: 'c2' } },
  ],
  edges: [
    { id: 'e1', source_node_id: 'n1', target_node_id: 'n2', kind: 'FILE_DEFINES', meta: {} },
    { id: 'e2', source_node_id: 'n2', target_node_id: 'n3', kind: 'VALIDATES', meta: {} },
  ],
  page: 0,
  limit: 1200,
  total_nodes: 3,
  projection: 'graphrag',
}

function makeProps(overrides: Partial<Parameters<typeof GraphRagView>[0]> = {}) {
  return {
    graph: populatedGraph,
    state: 'ready' as const,
    error: '',
    status: 'ready' as const,
    reason: 'ok' as const,
    selectedNodeId: '',
    communityMode: 'none' as const,
    communityId: '',
    communities: [],
    communitiesState: 'ready' as const,
    communitiesError: '',
    path: null,
    pathState: 'idle' as const,
    pathError: '',
    processes: [],
    processesState: 'ready' as const,
    processesError: '',
    processDetail: null,
    processDetailState: 'idle' as const,
    processDetailError: '',
    evidenceItems: [],
    evidenceTotal: 0,
    evidenceNodeName: '',
    evidenceState: 'idle' as const,
    evidenceError: '',
    onSelectNodeId: vi.fn(),
    onTracePath: vi.fn().mockResolvedValue(null),
    onLoadProcessDetail: vi.fn().mockResolvedValue(null),
    onRetry: vi.fn(),
    ...overrides,
  }
}

// --- Pure helper tests (replicated from component) ---

function colorForKind(kind: string): string {
  const normalized = kind.toLowerCase()
  if (normalized.includes('file')) return '#2563eb'
  if (normalized.includes('symbol')) return '#16a34a'
  if (normalized.includes('rule')) return '#d97706'
  if (normalized.includes('db')) return '#9333ea'
  if (normalized.includes('api')) return '#0f766e'
  if (normalized.includes('semantic')) return '#0ea5e9'
  return '#475569'
}

function pairKey(source: string, target: string): string {
  return `${source}->${target}`
}

function metaString(value: unknown): string {
  if (value == null) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number') return String(value)
  return ''
}

describe('colorForKind', () => {
  it('returns blue for FILE kind', () => {
    expect(colorForKind('FILE')).toBe('#2563eb')
  })

  it('returns green for SYMBOL kind', () => {
    expect(colorForKind('SYMBOL')).toBe('#16a34a')
  })

  it('returns amber for BUSINESS_RULE kind', () => {
    expect(colorForKind('BUSINESS_RULE')).toBe('#d97706')
  })

  it('returns purple for DB_TABLE kind', () => {
    expect(colorForKind('DB_TABLE')).toBe('#9333ea')
  })

  it('returns teal for API_ENDPOINT kind', () => {
    expect(colorForKind('API_ENDPOINT')).toBe('#0f766e')
  })

  it('returns sky for SEMANTIC_COMMUNITY kind', () => {
    expect(colorForKind('SEMANTIC_COMMUNITY')).toBe('#0ea5e9')
  })

  it('returns gray for unknown kind', () => {
    expect(colorForKind('UNKNOWN')).toBe('#475569')
  })
})

describe('pairKey', () => {
  it('creates directional pair key', () => {
    expect(pairKey('a', 'b')).toBe('a->b')
  })
})

describe('metaString', () => {
  it('returns empty string for null', () => {
    expect(metaString(null)).toBe('')
  })

  it('returns empty string for undefined', () => {
    expect(metaString(undefined)).toBe('')
  })

  it('returns string values as-is', () => {
    expect(metaString('hello')).toBe('hello')
  })

  it('converts numbers to string', () => {
    expect(metaString(42)).toBe('42')
  })

  it('returns empty string for objects', () => {
    expect(metaString({})).toBe('')
  })
})

// --- Rendering tests ---

describe('GraphRagView rendering', () => {
  it('renders the GraphRAG panel with header', () => {
    render(<GraphRagView {...makeProps()} />)
    expect(screen.getByText('GraphRAG graph')).toBeInTheDocument()
  })

  it('displays node and edge counts', () => {
    render(<GraphRagView {...makeProps()} />)
    expect(screen.getByText(/Nodes: 3/)).toBeInTheDocument()
    expect(screen.getByText(/Edges: 2/)).toBeInTheDocument()
  })

  it('renders toolbar buttons', () => {
    render(<GraphRagView {...makeProps()} />)
    expect(screen.getByText('Fit view')).toBeInTheDocument()
    expect(screen.getByText('Reset layout')).toBeInTheDocument()
    expect(screen.getByText('Hide labels')).toBeInTheDocument()
    expect(screen.getByText('Reset focus')).toBeInTheDocument()
  })

  it('toggles labels button text', async () => {
    render(<GraphRagView {...makeProps()} />)
    await userEvent.click(screen.getByText('Hide labels'))
    expect(screen.getByText('Show labels')).toBeInTheDocument()
  })

  it('renders graph canvas', () => {
    render(<GraphRagView {...makeProps()} />)
    expect(screen.getByLabelText('GraphRAG graph')).toBeInTheDocument()
  })

  it('renders trace path section', () => {
    render(<GraphRagView {...makeProps()} />)
    expect(screen.getByText('Trace path')).toBeInTheDocument()
    expect(screen.getByText('From')).toBeInTheDocument()
    expect(screen.getByText('To')).toBeInTheDocument()
    expect(screen.getByText('Max hops')).toBeInTheDocument()
  })

  it('renders trace button', () => {
    render(<GraphRagView {...makeProps()} />)
    expect(screen.getByText('Trace')).toBeInTheDocument()
  })

  it('shows Tracing text when path is loading', () => {
    render(<GraphRagView {...makeProps({ pathState: 'loading' })} />)
    expect(screen.getByText('Tracing\u2026')).toBeInTheDocument()
  })

  it('renders processes section', () => {
    render(<GraphRagView {...makeProps()} />)
    expect(screen.getByText('Processes')).toBeInTheDocument()
    expect(screen.getByText('0 detected')).toBeInTheDocument()
  })

  it('renders processes with cross-community and intra sections', () => {
    const processes = [
      { id: 'p1', label: 'Auth flow', process_type: 'cross_community' as const, step_count: 5, community_ids: ['c1', 'c2'], entry_node_id: 'n1', terminal_node_id: 'n3' },
      { id: 'p2', label: 'Cache flush', process_type: 'intra_community' as const, step_count: 3, community_ids: ['c1'], entry_node_id: 'n1', terminal_node_id: 'n2' },
    ]
    render(<GraphRagView {...makeProps({ processes })} />)
    expect(screen.getByText(/Cross-community/)).toBeInTheDocument()
    expect(screen.getByText(/Intra-community/)).toBeInTheDocument()
    expect(screen.getByText('Auth flow')).toBeInTheDocument()
    expect(screen.getByText('Cache flush')).toBeInTheDocument()
  })

  it('renders evidence section', () => {
    render(<GraphRagView {...makeProps()} />)
    expect(screen.getByText('Indexed text')).toBeInTheDocument()
  })

  it('shows evidence prompt when no node selected', () => {
    render(<GraphRagView {...makeProps()} />)
    expect(screen.getByText('Select a graph node to inspect indexed evidence text.')).toBeInTheDocument()
  })

  it('shows evidence items when node is selected and evidence exists', () => {
    const evidenceItems = [
      { evidence_id: 'ev1', file_path: 'src/auth.py', start_line: 10, end_line: 20, text: 'def authenticate():', text_source: 'snippet' as const },
    ]
    render(<GraphRagView {...makeProps({ selectedNodeId: 'n1', evidenceItems, evidenceTotal: 1, evidenceState: 'ready', evidenceNodeName: 'main.py' })} />)
    expect(screen.getByText('src/auth.py')).toBeInTheDocument()
    expect(screen.getByText('def authenticate():')).toBeInTheDocument()
  })

  it('shows no evidence message when node selected but no evidence found', () => {
    render(<GraphRagView {...makeProps({ selectedNodeId: 'n1', evidenceState: 'ready' })} />)
    expect(screen.getByText('No indexed evidence was found for this node.')).toBeInTheDocument()
  })

  it('shows evidence loading state', () => {
    render(<GraphRagView {...makeProps({ selectedNodeId: 'n1', evidenceState: 'loading' })} />)
    // Loading renders skeleton cards
    expect(screen.queryByText('No indexed evidence was found for this node.')).not.toBeInTheDocument()
  })

  it('shows evidence error state', () => {
    render(<GraphRagView {...makeProps({ selectedNodeId: 'n1', evidenceState: 'error', evidenceError: 'Evidence fetch failed' })} />)
    expect(screen.getByText('Evidence fetch failed')).toBeInTheDocument()
  })

  it('renders communities section', () => {
    render(<GraphRagView {...makeProps()} />)
    expect(screen.getByText('Communities')).toBeInTheDocument()
  })

  it('shows communities loading state', () => {
    render(<GraphRagView {...makeProps({ communitiesState: 'loading' })} />)
    expect(screen.getByText(/Loading communities/)).toBeInTheDocument()
  })

  it('shows community chips when communities exist', () => {
    const communities = [
      { id: 'c1', label: 'Auth domain', size: 5, cohesion: 0.8, top_kinds: [], sample_nodes: [{ id: 'n1', name: 'auth', kind: 'FILE', natural_key: 'f:auth' }] },
    ]
    render(<GraphRagView {...makeProps({ communities })} />)
    expect(screen.getByText('Auth domain')).toBeInTheDocument()
  })

  it('renders guided empty state when status is unavailable', () => {
    render(<GraphRagView {...makeProps({ status: 'unavailable', reason: 'no_knowledge_graph', graph: emptyGraph })} />)
    expect(screen.getByText('No knowledge graph available yet')).toBeInTheDocument()
    expect(screen.getByText(/Knowledge graph data has not been generated/)).toBeInTheDocument()
  })

  it('renders loading skeleton when loading with no data', () => {
    render(<GraphRagView {...makeProps({ state: 'loading', graph: emptyGraph })} />)
    expect(screen.queryByText('GraphRAG graph')).not.toBeInTheDocument()
  })

  it('renders error state when error with no data', () => {
    render(<GraphRagView {...makeProps({ state: 'error', error: 'Server error', graph: emptyGraph })} />)
    expect(screen.getByText('GraphRAG request failed')).toBeInTheDocument()
  })

  it('shows path nodes when path is traced', () => {
    const path = {
      collection_id: 'c1',
      scenario: { id: 's1', collection_id: 'c1', name: 'Base', version: 1, is_as_is: true, base_scenario_id: null },
      status: 'found' as const,
      from_node_id: 'n1',
      to_node_id: 'n3',
      max_hops: 6,
      path: {
        nodes: [
          { id: 'n1', natural_key: 'file:main.py', kind: 'FILE', name: 'main.py', meta: {} },
          { id: 'n3', natural_key: 'rule:check', kind: 'RULE', name: 'check_auth', meta: {} },
        ],
        edges: [{ id: 'pe1', source_node_id: 'n1', target_node_id: 'n3', kind: 'VALIDATES', meta: {} }],
        hops: 1,
      },
    }
    render(<GraphRagView {...makeProps({ path })} />)
    expect(screen.getByText('Status: found')).toBeInTheDocument()
    expect(screen.getByText(/1\. main\.py/)).toBeInTheDocument()
    expect(screen.getByText(/2\. check_auth/)).toBeInTheDocument()
  })
})
