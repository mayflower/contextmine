import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import CockpitCommandBar from './CockpitCommandBar'
import type { SemanticMapThresholds } from '../types'

const defaultThresholds: SemanticMapThresholds = {
  mixed_cluster_max_dominant_ratio: 0.7,
  isolated_distance_multiplier: 2,
  semantic_duplication_min_similarity: 0.8,
  semantic_duplication_max_source_overlap: 0.5,
  misplaced_min_dominant_ratio: 0.6,
}

const noop = vi.fn()

function makeProps(overrides: Record<string, unknown> = {}) {
  return {
    collections: [{ id: 'c1', name: 'Project A' }],
    scenarios: [{ id: 's1', name: 'Base', version: 1, is_as_is: true }],
    collectionId: 'c1',
    scenarioId: 's1',
    layer: 'code_controlflow' as const,
    activeView: 'overview' as const,
    hotspotFilter: '',
    graphQuery: '',
    hideIsolated: false,
    graphPage: 0,
    graphLimit: 1200,
    includeKinds: [] as string[],
    excludeKinds: [] as string[],
    edgeKinds: [] as string[],
    overlayMode: 'none' as const,
    graphRagCommunityMode: 'none' as const,
    graphRagCommunityId: '',
    graphRagCommunities: [],
    semanticMapMode: 'code_structure' as const,
    semanticMapShowDiffOverlay: false,
    semanticMapDiffMinDrift: 0.1,
    semanticMapThresholds: defaultThresholds,
    c4View: 'container' as const,
    c4Scope: '',
    c4MaxNodes: 200,
    architectureSection: '',
    portsDirection: 'all' as const,
    portsContainer: '',
    driftBaselineScenarioId: '',
    driftScenarioOptions: [],
    availableNodeKinds: ['CONTAINER', 'COMPONENT'],
    availableEdgeKinds: ['CALLS', 'READS'],
    onCollectionChange: noop,
    onScenarioChange: noop,
    onLayerChange: noop,
    onFilterChange: noop,
    onGraphQueryChange: noop,
    onHideIsolatedChange: noop,
    onGraphPageChange: noop,
    onGraphLimitChange: noop,
    onIncludeKindsChange: noop,
    onExcludeKindsChange: noop,
    onEdgeKindsChange: noop,
    onOverlayModeChange: noop,
    onGraphRagCommunityModeChange: noop,
    onGraphRagCommunityIdChange: noop,
    onSemanticMapModeChange: noop,
    onSemanticMapShowDiffOverlayChange: noop,
    onSemanticMapDiffMinDriftChange: noop,
    onSemanticMapThresholdsChange: noop,
    onC4ViewChange: noop,
    onC4ScopeChange: noop,
    onC4MaxNodesChange: noop,
    onArchitectureSectionChange: noop,
    onPortsDirectionChange: noop,
    onPortsContainerChange: noop,
    onDriftBaselineScenarioIdChange: noop,
    onRefresh: noop,
    onLoadOverlayFile: noop,
    onOpenCollections: noop,
    onOpenRuns: noop,
    ...overrides,
  }
}

describe('CockpitCommandBar', () => {
  it('renders project and scenario selects', () => {
    render(<CockpitCommandBar {...makeProps()} />)
    expect(screen.getByText('Project')).toBeInTheDocument()
    expect(screen.getByText('Scenario')).toBeInTheDocument()
  })

  it('renders collections as options', () => {
    render(<CockpitCommandBar {...makeProps()} />)
    expect(screen.getByText('Project A')).toBeInTheDocument()
  })

  it('renders scenario options with version and AS-IS/TO-BE', () => {
    render(<CockpitCommandBar {...makeProps()} />)
    expect(screen.getByText(/Base \(v1\)/)).toBeInTheDocument()
  })

  it('renders Refresh view button', async () => {
    const onRefresh = vi.fn()
    render(<CockpitCommandBar {...makeProps({ onRefresh })} />)
    const btn = screen.getByRole('button', { name: 'Refresh view' })
    await userEvent.click(btn)
    expect(onRefresh).toHaveBeenCalledOnce()
  })

  it('renders Open Collections and Open Runs buttons', () => {
    render(<CockpitCommandBar {...makeProps()} />)
    expect(screen.getByRole('button', { name: 'Open Collections' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Open Runs' })).toBeInTheDocument()
  })

  it('shows hotspot filter on overview view', () => {
    render(<CockpitCommandBar {...makeProps({ activeView: 'overview' })} />)
    expect(screen.getByText('Hotspot filter')).toBeInTheDocument()
  })

  it('shows layer select on topology view', () => {
    render(<CockpitCommandBar {...makeProps({ activeView: 'topology' })} />)
    expect(screen.getByText('Layer')).toBeInTheDocument()
  })

  it('shows C4 view controls on c4_diff view', () => {
    render(<CockpitCommandBar {...makeProps({ activeView: 'c4_diff' })} />)
    expect(screen.getByText('C4 view')).toBeInTheDocument()
    expect(screen.getByText('C4 scope')).toBeInTheDocument()
  })

  it('shows architecture section and ports controls on architecture view', () => {
    render(<CockpitCommandBar {...makeProps({ activeView: 'architecture' })} />)
    expect(screen.getByText('arc42 section')).toBeInTheDocument()
    expect(screen.getByText('Ports direction')).toBeInTheDocument()
    expect(screen.getByText('Ports container')).toBeInTheDocument()
    expect(screen.getByText('Drift baseline')).toBeInTheDocument()
  })

  it('shows community mode controls on graphrag view', () => {
    render(<CockpitCommandBar {...makeProps({ activeView: 'graphrag' })} />)
    expect(screen.getByText('Community mode')).toBeInTheDocument()
    // When graphrag is active, showGraphControls is also true and comes first in
    // the secondary slot chain, so "Graph search" is shown instead of "Community filter"
    expect(screen.getByText('Graph search')).toBeInTheDocument()
  })

  it('shows map mode on semantic_map view', () => {
    render(<CockpitCommandBar {...makeProps({ activeView: 'semantic_map' })} />)
    expect(screen.getByText('Map mode')).toBeInTheDocument()
  })

  it('shows graph controls (advanced) on topology view', () => {
    render(<CockpitCommandBar {...makeProps({ activeView: 'topology' })} />)
    expect(screen.getByText('Advanced graph controls')).toBeInTheDocument()
  })

  it('has aria-label on the section', () => {
    render(<CockpitCommandBar {...makeProps()} />)
    expect(screen.getByLabelText('Cockpit controls')).toBeInTheDocument()
  })
})
