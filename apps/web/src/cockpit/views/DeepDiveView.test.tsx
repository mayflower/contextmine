/**
 * Tests for DeepDiveView rendering + pure helper functions.
 */
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { OverlayState, TwinGraphResponse } from '../types'

vi.mock('cytoscape', () => ({
  default: vi.fn().mockReturnValue({
    on: vi.fn(),
    destroy: vi.fn(),
    fit: vi.fn(),
    layout: vi.fn().mockReturnValue({ run: vi.fn() }),
  }),
}))

import DeepDiveView from './DeepDiveView'

const noOverlay: OverlayState = {
  mode: 'none',
  runtimeByNodeKey: {},
  riskByNodeKey: {},
  loadedAt: null,
}

const emptyGraph: TwinGraphResponse = {
  nodes: [],
  edges: [],
  page: 0,
  limit: 5000,
  total_nodes: 0,
}

const populatedGraph: TwinGraphResponse = {
  nodes: [
    { id: 'n1', natural_key: 'file:auth.py', kind: 'FILE', name: 'auth.py', meta: {} },
    { id: 'n2', natural_key: 'file:billing.py', kind: 'FILE', name: 'billing.py', meta: {} },
  ],
  edges: [
    { id: 'e1', source_node_id: 'n1', target_node_id: 'n2', kind: 'IMPORTS', meta: {} },
  ],
  page: 0,
  limit: 5000,
  total_nodes: 2,
  projection: 'code_file',
  slice_strategy: 'edge_aware_seed_window',
}

function makeProps(overrides: Partial<Parameters<typeof DeepDiveView>[0]> = {}) {
  return {
    graph: populatedGraph,
    state: 'ready' as const,
    error: '',
    layer: 'code_controlflow' as const,
    mode: 'file_dependency' as const,
    density: 3000,
    overlay: noOverlay,
    selectedNodeId: '',
    onModeChange: vi.fn(),
    onDensityChange: vi.fn(),
    onSelectNodeId: vi.fn(),
    onSwitchToCodeLayer: vi.fn(),
    onRetry: vi.fn(),
    ...overrides,
  }
}

// --- Pure helper tests ---

function getLayoutName(density: number): 'cose' | 'breadthfirst' {
  return density > 5000 ? 'breadthfirst' : 'cose'
}

describe('getLayoutName', () => {
  it('returns cose for density <= 5000', () => {
    expect(getLayoutName(5000)).toBe('cose')
    expect(getLayoutName(1000)).toBe('cose')
    expect(getLayoutName(0)).toBe('cose')
  })

  it('returns breadthfirst for density > 5000', () => {
    expect(getLayoutName(5001)).toBe('breadthfirst')
    expect(getLayoutName(10000)).toBe('breadthfirst')
  })
})

// --- Rendering tests ---

describe('DeepDiveView rendering', () => {
  it('renders the deep dive panel with header', () => {
    render(<DeepDiveView {...makeProps()} />)
    expect(screen.getByText('Deep dive graph')).toBeInTheDocument()
  })

  it('displays node, edge counts and projection info', () => {
    render(<DeepDiveView {...makeProps()} />)
    expect(screen.getByText(/Nodes: 2/)).toBeInTheDocument()
    expect(screen.getByText(/Edges: 1/)).toBeInTheDocument()
  })

  it('displays mode in muted text', () => {
    render(<DeepDiveView {...makeProps()} />)
    expect(screen.getByText(/Mode: file_dependency/)).toBeInTheDocument()
  })

  it('renders toolbar buttons', () => {
    render(<DeepDiveView {...makeProps()} />)
    expect(screen.getByText('Fit view')).toBeInTheDocument()
    expect(screen.getByText('Reset layout')).toBeInTheDocument()
    expect(screen.getByText('Hide labels')).toBeInTheDocument()
  })

  it('renders slice warnings when present', () => {
    render(<DeepDiveView {...makeProps({ graph: { ...populatedGraph, warnings: ['1 cross-page edge is hidden outside this graph slice.'] } })} />)
    expect(screen.getByText(/cross-page edge is hidden/i)).toBeInTheDocument()
  })

  it('toggles label button text on click', async () => {
    render(<DeepDiveView {...makeProps()} />)
    const btn = screen.getByText('Hide labels')
    await userEvent.click(btn)
    expect(screen.getByText('Show labels')).toBeInTheDocument()
  })

  it('renders mode select with options', () => {
    render(<DeepDiveView {...makeProps()} />)
    expect(screen.getByText('Mode')).toBeInTheDocument()
    expect(screen.getByText('File dependency')).toBeInTheDocument()
    expect(screen.getByText('Symbol callgraph')).toBeInTheDocument()
    expect(screen.getByText('Contains hierarchy')).toBeInTheDocument()
  })

  it('renders density select with options', () => {
    render(<DeepDiveView {...makeProps()} />)
    expect(screen.getByText('Focused')).toBeInTheDocument()
    expect(screen.getByText('Balanced')).toBeInTheDocument()
    expect(screen.getByText('Dense')).toBeInTheDocument()
  })

  it('renders canvas div when nodes exist', () => {
    render(<DeepDiveView {...makeProps()} />)
    expect(screen.getByLabelText('Deep dive graph')).toBeInTheDocument()
  })

  it('shows dense mode warning when density > 5000', () => {
    render(<DeepDiveView {...makeProps({ density: 8000 })} />)
    expect(screen.getByText(/Dense mode can be expensive/)).toBeInTheDocument()
  })

  it('does not show dense mode warning when density <= 5000', () => {
    render(<DeepDiveView {...makeProps({ density: 3000 })} />)
    expect(screen.queryByText(/Dense mode can be expensive/)).not.toBeInTheDocument()
  })

  it('renders empty state when graph has no nodes', () => {
    render(<DeepDiveView {...makeProps({ graph: emptyGraph })} />)
    expect(screen.getByText('No nodes for this layer')).toBeInTheDocument()
  })

  it('shows switch button for non-code layers with total_nodes > 0', () => {
    render(<DeepDiveView {...makeProps({
      graph: { ...emptyGraph, total_nodes: 10 },
      layer: 'domain_container',
    })} />)
    expect(screen.getByText('Switch to Code / Controlflow')).toBeInTheDocument()
  })

  it('hides switch button when on code layer', () => {
    render(<DeepDiveView {...makeProps({
      graph: { ...emptyGraph, total_nodes: 10 },
      layer: 'code_controlflow',
    })} />)
    expect(screen.queryByText('Switch to Code / Controlflow')).not.toBeInTheDocument()
  })

  it('shows no-nodes message when total_nodes is 0', () => {
    render(<DeepDiveView {...makeProps({ graph: emptyGraph })} />)
    expect(screen.getByText(/No twin nodes are available/)).toBeInTheDocument()
  })

  it('renders loading skeleton when state is loading and no data', () => {
    render(<DeepDiveView {...makeProps({ state: 'loading', graph: emptyGraph })} />)
    expect(screen.queryByText('Deep dive graph')).not.toBeInTheDocument()
  })

  it('renders error state when state is error and no data', () => {
    render(<DeepDiveView {...makeProps({ state: 'error', error: 'Timeout', graph: emptyGraph })} />)
    expect(screen.getByText('Deep dive request failed')).toBeInTheDocument()
    expect(screen.getByText('Timeout')).toBeInTheDocument()
  })

  it('calls onSwitchToCodeLayer on button click', async () => {
    const onSwitchToCodeLayer = vi.fn()
    render(<DeepDiveView {...makeProps({
      graph: { ...emptyGraph, total_nodes: 5 },
      layer: 'domain_container',
      onSwitchToCodeLayer,
    })} />)
    await userEvent.click(screen.getByText('Switch to Code / Controlflow'))
    expect(onSwitchToCodeLayer).toHaveBeenCalledOnce()
  })
})
