/**
 * Tests for TopologyView rendering + pure helper functions.
 */
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, beforeEach } from 'vitest'

import type { OverlayState, TwinGraphResponse } from '../types'

vi.mock('reactflow', () => {
  const Background = () => <div data-testid="rf-background" />
  const Controls = () => <div data-testid="rf-controls" />
  const MiniMap = () => <div data-testid="rf-minimap" />
  const ReactFlowComponent = (props: Record<string, unknown>) => (
    <div data-testid="reactflow" aria-label={props['aria-label'] as string} />
  )
  return {
    default: ReactFlowComponent,
    ReactFlow: ReactFlowComponent,
    Background,
    Controls,
    MiniMap,
  }
})

vi.mock('../layout/layoutCore', () => ({
  runGridLayout: vi.fn().mockReturnValue({}),
  runElkLayout: vi.fn().mockResolvedValue({}),
}))

import TopologyView from './TopologyView'

// --- Shared test data ---

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
  limit: 1200,
  total_nodes: 0,
}

const populatedGraph: TwinGraphResponse = {
  nodes: [
    { id: 'n1', natural_key: 'svc:auth', kind: 'CONTAINER', name: 'auth-service', meta: {} },
    { id: 'n2', natural_key: 'svc:billing', kind: 'CONTAINER', name: 'billing-service', meta: {} },
    { id: 'n3', natural_key: 'svc:gateway', kind: 'CONTAINER', name: 'api-gateway', meta: {} },
  ],
  edges: [
    { id: 'e1', source_node_id: 'n1', target_node_id: 'n2', kind: 'CALLS', meta: {} },
    { id: 'e2', source_node_id: 'n3', target_node_id: 'n1', kind: 'CALLS', meta: {} },
  ],
  page: 0,
  limit: 1200,
  total_nodes: 3,
  projection: 'architecture',
}

function makeProps(overrides: Partial<Parameters<typeof TopologyView>[0]> = {}) {
  return {
    graph: populatedGraph,
    state: 'ready' as const,
    error: '',
    layer: 'code_controlflow' as const,
    density: 1200,
    layoutEngine: 'grid' as const,
    elkEnabled: true,
    overlay: noOverlay,
    selectedNodeId: '',
    onDensityChange: vi.fn(),
    onLayoutEngineChange: vi.fn(),
    onSwitchToCodeLayer: vi.fn(),
    onSelectNodeId: vi.fn(),
    onLayoutCompleted: vi.fn(),
    onRetry: vi.fn(),
    ...overrides,
  }
}

// --- Pure helper tests (replicated from component) ---

function overlayColorForNode(overlay: OverlayState, naturalKey: string, fallbackName: string): string {
  if (overlay.mode === 'runtime') {
    const runtime = overlay.runtimeByNodeKey[naturalKey] || overlay.runtimeByNodeKey[fallbackName]
    const errorRate = Number(runtime?.error_rate || 0)
    if (errorRate >= 0.1) return '#dc2626'
    if (errorRate >= 0.03) return '#f59e0b'
    return '#2563eb'
  }
  if (overlay.mode === 'risk') {
    const risk = overlay.riskByNodeKey[naturalKey] || overlay.riskByNodeKey[fallbackName]
    const severity = Number(risk?.severity_score || 0)
    if (severity >= 8) return '#b91c1c'
    if (severity >= 4) return '#d97706'
    return '#1d4ed8'
  }
  return '#1d4ed8'
}

describe('overlayColorForNode', () => {
  it('returns default blue when mode is none', () => {
    expect(overlayColorForNode(noOverlay, 'key', 'name')).toBe('#1d4ed8')
  })

  describe('runtime mode', () => {
    const runtimeOverlay: OverlayState = {
      mode: 'runtime',
      runtimeByNodeKey: {
        'service:auth': { service: 'auth', error_rate: 0.15 },
        'service:billing': { service: 'billing', error_rate: 0.05 },
        'service:healthy': { service: 'healthy', error_rate: 0.01 },
      },
      riskByNodeKey: {},
      loadedAt: null,
    }

    it('returns red for high error rate (>=0.1)', () => {
      expect(overlayColorForNode(runtimeOverlay, 'service:auth', '')).toBe('#dc2626')
    })

    it('returns amber for medium error rate (>=0.03)', () => {
      expect(overlayColorForNode(runtimeOverlay, 'service:billing', '')).toBe('#f59e0b')
    })

    it('returns blue for low error rate', () => {
      expect(overlayColorForNode(runtimeOverlay, 'service:healthy', '')).toBe('#2563eb')
    })

    it('uses fallback name when natural key not found', () => {
      expect(overlayColorForNode(runtimeOverlay, 'nonexistent', 'service:auth')).toBe('#dc2626')
    })

    it('returns blue when no runtime data exists', () => {
      expect(overlayColorForNode(runtimeOverlay, 'unknown', 'also_unknown')).toBe('#2563eb')
    })
  })

  describe('risk mode', () => {
    const riskOverlay: OverlayState = {
      mode: 'risk',
      runtimeByNodeKey: {},
      riskByNodeKey: {
        'api-gateway': { node: 'api-gateway', severity_score: 9 },
        'user-service': { node: 'user-service', severity_score: 5 },
        'static-assets': { node: 'static-assets', severity_score: 1 },
      },
      loadedAt: null,
    }

    it('returns dark red for high severity (>=8)', () => {
      expect(overlayColorForNode(riskOverlay, 'api-gateway', '')).toBe('#b91c1c')
    })

    it('returns amber for medium severity (>=4)', () => {
      expect(overlayColorForNode(riskOverlay, 'user-service', '')).toBe('#d97706')
    })

    it('returns blue for low severity', () => {
      expect(overlayColorForNode(riskOverlay, 'static-assets', '')).toBe('#1d4ed8')
    })
  })
})

// --- Rendering tests ---

describe('TopologyView rendering', () => {
  it('renders the topology panel with header', () => {
    render(<TopologyView {...makeProps()} />)
    expect(screen.getByText('Topology graph')).toBeInTheDocument()
  })

  it('displays node and edge counts', () => {
    render(<TopologyView {...makeProps()} />)
    expect(screen.getByText(/Nodes: 3/)).toBeInTheDocument()
    expect(screen.getByText(/Edges: 2/)).toBeInTheDocument()
  })

  it('renders ReactFlow when nodes exist', () => {
    render(<TopologyView {...makeProps()} />)
    expect(screen.getByTestId('reactflow')).toBeInTheDocument()
  })

  it('renders toolbar buttons', () => {
    render(<TopologyView {...makeProps()} />)
    expect(screen.getByText('Fit view')).toBeInTheDocument()
    expect(screen.getByText('Show labels')).toBeInTheDocument()
    expect(screen.getByText('Display options')).toBeInTheDocument()
  })

  it('toggles show labels button text on click', async () => {
    render(<TopologyView {...makeProps()} />)
    const btn = screen.getByText('Show labels')
    await userEvent.click(btn)
    expect(screen.getByText('Hide labels')).toBeInTheDocument()
  })

  it('shows display options panel when toggled', async () => {
    render(<TopologyView {...makeProps()} />)
    await userEvent.click(screen.getByText('Display options'))
    expect(screen.getByText('Density')).toBeInTheDocument()
    expect(screen.getByText('Layout')).toBeInTheDocument()
    expect(screen.getByText('Show mini map')).toBeInTheDocument()
  })

  it('renders empty state when graph has no nodes', () => {
    render(<TopologyView {...makeProps({ graph: emptyGraph })} />)
    expect(screen.getByText('No nodes for this layer')).toBeInTheDocument()
  })

  it('shows switch to code layer button for non-code layers in empty state', () => {
    render(<TopologyView {...makeProps({ graph: { ...emptyGraph, total_nodes: 5 }, layer: 'domain_container' })} />)
    expect(screen.getByText('Switch to Code / Controlflow')).toBeInTheDocument()
  })

  it('hides switch button when already on code layer in empty state', () => {
    render(<TopologyView {...makeProps({ graph: { ...emptyGraph, total_nodes: 5 }, layer: 'code_controlflow' })} />)
    expect(screen.queryByText('Switch to Code / Controlflow')).not.toBeInTheDocument()
  })

  it('shows sync message when total_nodes is 0', () => {
    render(<TopologyView {...makeProps({ graph: emptyGraph })} />)
    expect(screen.getByText(/Run sync again/)).toBeInTheDocument()
  })

  it('shows runtime overlay legend when overlay mode is runtime', () => {
    const runtimeOverlay: OverlayState = {
      mode: 'runtime',
      runtimeByNodeKey: {},
      riskByNodeKey: {},
      loadedAt: null,
    }
    render(<TopologyView {...makeProps({ overlay: runtimeOverlay })} />)
    expect(screen.getByText('Runtime health')).toBeInTheDocument()
  })

  it('shows risk overlay legend when overlay mode is risk', () => {
    const riskOverlay: OverlayState = {
      mode: 'risk',
      runtimeByNodeKey: {},
      riskByNodeKey: {},
      loadedAt: null,
    }
    render(<TopologyView {...makeProps({ overlay: riskOverlay })} />)
    expect(screen.getByText('Dependency risk')).toBeInTheDocument()
  })

  it('hides overlay legend when overlay mode is none', () => {
    render(<TopologyView {...makeProps()} />)
    expect(screen.queryByText('Runtime health')).not.toBeInTheDocument()
    expect(screen.queryByText('Dependency risk')).not.toBeInTheDocument()
  })

  it('renders loading skeleton when state is loading and no data', () => {
    render(<TopologyView {...makeProps({ state: 'loading', graph: emptyGraph })} />)
    expect(screen.queryByText('Topology graph')).not.toBeInTheDocument()
  })

  it('renders error state when state is error and no data', () => {
    render(<TopologyView {...makeProps({ state: 'error', error: 'Network error', graph: emptyGraph })} />)
    expect(screen.getByText('Topology request failed')).toBeInTheDocument()
    expect(screen.getByText('Network error')).toBeInTheDocument()
  })

  it('calls onSwitchToCodeLayer when switch button is clicked', async () => {
    const onSwitchToCodeLayer = vi.fn()
    render(<TopologyView {...makeProps({
      graph: { ...emptyGraph, total_nodes: 5 },
      layer: 'domain_container',
      onSwitchToCodeLayer,
    })} />)
    await userEvent.click(screen.getByText('Switch to Code / Controlflow'))
    expect(onSwitchToCodeLayer).toHaveBeenCalledOnce()
  })
})
