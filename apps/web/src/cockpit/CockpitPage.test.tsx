/**
 * Tests for CockpitPage rendering.
 *
 * We mock useCockpitData and useCockpitState to render the component without
 * actual API calls. This covers the JSX composition logic and early returns.
 */
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, beforeEach } from 'vitest'

// Mock all view components to avoid deep dependency trees
vi.mock('./views/TopologyView', () => ({
  default: () => <div data-testid="topology-view">TopologyView</div>,
}))
vi.mock('./views/DeepDiveView', () => ({
  default: () => <div data-testid="deep-dive-view">DeepDiveView</div>,
}))
vi.mock('./views/C4DiffView', () => ({
  default: () => <div data-testid="c4-diff-view">C4DiffView</div>,
}))
vi.mock('./views/ArchitectureView', () => ({
  default: () => <div data-testid="architecture-view">ArchitectureView</div>,
}))
vi.mock('./views/CityView', () => ({
  default: () => <div data-testid="city-view">CityView</div>,
}))
vi.mock('./views/EvolutionView', () => ({
  default: () => <div data-testid="evolution-view">EvolutionView</div>,
}))
vi.mock('./views/GraphRagView', () => ({
  default: () => <div data-testid="graphrag-view">GraphRagView</div>,
}))
vi.mock('./views/OverviewView', () => ({
  default: () => <div data-testid="overview-view">OverviewView</div>,
}))
vi.mock('./views/SemanticMapView', () => ({
  default: () => <div data-testid="semantic-map-view">SemanticMapView</div>,
}))
vi.mock('./views/TestMatrixView', () => ({
  default: () => <div data-testid="test-matrix-view">TestMatrixView</div>,
}))
vi.mock('./views/RebuildReadinessView', () => ({
  default: () => <div data-testid="rebuild-readiness-view">RebuildReadinessView</div>,
}))
vi.mock('./views/ExportsView', () => ({
  default: () => <div data-testid="exports-view">ExportsView</div>,
}))

vi.mock('../faro', () => ({
  getFaro: () => null,
}))

const mockSetCollectionId = vi.fn()
const mockSetScenarioId = vi.fn()
const mockSetLayer = vi.fn()
const mockSetView = vi.fn()
const mockSetHotspotFilter = vi.fn()
const mockSetGraphQuery = vi.fn()
const mockSetSelectedNodeId = vi.fn()
const mockSetGraphPage = vi.fn()
const mockSetGraphLimit = vi.fn()
const mockSetIncludeKinds = vi.fn()
const mockSetExcludeKinds = vi.fn()
const mockSetEdgeKinds = vi.fn()
const mockSetHideIsolated = vi.fn()
const mockSetOverlayMode = vi.fn()

const defaultSelection = {
  collectionId: 'c1',
  scenarioId: 's1',
  layer: 'code_controlflow' as const,
  view: 'overview' as const,
}

vi.mock('./hooks/useCockpitState', () => ({
  useCockpitState: () => ({
    selection: defaultSelection,
    hotspotFilter: '',
    graphQuery: '',
    selectedNodeId: '',
    graphPage: 0,
    graphLimit: 1200,
    includeKinds: [],
    excludeKinds: [],
    edgeKinds: [],
    hideIsolated: false,
    overlayMode: 'none' as const,
    setHotspotFilter: mockSetHotspotFilter,
    setGraphQuery: mockSetGraphQuery,
    setSelectedNodeId: mockSetSelectedNodeId,
    setGraphPage: mockSetGraphPage,
    setGraphLimit: mockSetGraphLimit,
    setIncludeKinds: mockSetIncludeKinds,
    setExcludeKinds: mockSetExcludeKinds,
    setEdgeKinds: mockSetEdgeKinds,
    setHideIsolated: mockSetHideIsolated,
    setOverlayMode: mockSetOverlayMode,
    setCollectionId: mockSetCollectionId,
    setScenarioId: mockSetScenarioId,
    setLayer: mockSetLayer,
    setView: mockSetView,
  }),
}))

const defaultGraph = {
  nodes: [],
  edges: [],
  page: 0,
  limit: 1200,
  total_nodes: 0,
}

vi.mock('./hooks/useCockpitData', () => ({
  useCockpitData: () => ({
    scenarios: [{ id: 's1', name: 'Base', version: 1, is_as_is: true }],
    scenariosState: 'ready' as const,
    city: null,
    graph: defaultGraph,
    mermaid: null,
    arc42: null,
    portsAdapters: null,
    arc42Drift: null,
    erm: null,
    architecturePanelErrors: { arc42: '', ports: '', drift: '', erm: '' },
    architectureActions: { reindexState: 'idle' as const, reindexMessage: '', regenerateState: 'idle' as const, regenerateMessage: '' },
    activeState: 'ready' as const,
    activeError: '',
    activeUpdatedAt: null,
    errors: { topology: '', deep_dive: '', c4_diff: '', architecture: '', city: '', evolution: '', graphrag: '', semantic_map: '', ui_map: '', test_matrix: '', user_flows: '', rebuild_readiness: '', exports: '', overview: '' },
    cityProjection: 'architecture' as const,
    setCityProjection: vi.fn(),
    cityEntityLevel: 'domain' as const,
    setCityEntityLevel: vi.fn(),
    cityEmbedUrl: '',
    investmentUtilization: null,
    knowledgeIslands: null,
    temporalCoupling: null,
    fitnessFunctions: null,
    evolutionPanelErrors: { investment: '', knowledge: '', coupling: '', fitness: '' },
    exportFormat: 'cc_json' as const,
    setExportFormat: vi.fn(),
    exportProjection: 'architecture' as const,
    setExportProjection: vi.fn(),
    exportContent: null,
    neighborhood: null,
    neighborhoodState: 'idle' as const,
    neighborhoodError: '',
    graphRagStatus: 'ready' as const,
    graphRagReason: 'ok' as const,
    graphRagEvidenceItems: [],
    graphRagEvidenceTotal: 0,
    graphRagEvidenceNodeName: '',
    graphRagEvidenceState: 'idle' as const,
    graphRagEvidenceError: '',
    graphRagCommunities: [],
    graphRagCommunitiesState: 'ready' as const,
    graphRagCommunitiesError: '',
    graphRagPath: null,
    graphRagPathState: 'idle' as const,
    graphRagPathError: '',
    graphRagProcesses: [],
    graphRagProcessesState: 'ready' as const,
    graphRagProcessesError: '',
    graphRagProcessDetail: null,
    graphRagProcessDetailState: 'idle' as const,
    graphRagProcessDetailError: '',
    semanticMap: null,
    semanticMapComparison: null,
    uiMapSummary: null,
    testMatrix: null,
    userFlows: null,
    rebuildReadiness: null,
    traceGraphRagPath: vi.fn().mockResolvedValue(null),
    loadGraphRagProcessDetail: vi.fn().mockResolvedValue(null),
    triggerCollectionReindex: vi.fn().mockResolvedValue(true),
    regenerateArc42: vi.fn().mockResolvedValue(true),
    generateExport: vi.fn().mockResolvedValue(null),
    refreshActiveView: vi.fn(),
  }),
}))

import CockpitPage from './CockpitPage'

const defaultCollections = [{ id: 'c1', name: 'ContextMine' }]

describe('CockpitPage rendering', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    defaultSelection.view = 'overview'
    defaultSelection.collectionId = 'c1'
  })

  it('renders the cockpit shell with header', () => {
    render(<CockpitPage collections={defaultCollections} />)
    expect(screen.getByText('Architecture Cockpit')).toBeInTheDocument()
  })

  it('renders onboarding when no collections', () => {
    render(<CockpitPage collections={[]} />)
    expect(screen.getByText('Start by creating a collection')).toBeInTheDocument()
    expect(screen.getByText('Go to Collections')).toBeInTheDocument()
    expect(screen.getByText('Go to Runs')).toBeInTheDocument()
  })

  it('renders overview view by default', () => {
    defaultSelection.view = 'overview'
    render(<CockpitPage collections={defaultCollections} />)
    expect(screen.getByTestId('overview-view')).toBeInTheDocument()
  })

  it('renders topology view when selected', () => {
    defaultSelection.view = 'topology'
    render(<CockpitPage collections={defaultCollections} />)
    expect(screen.getByTestId('topology-view')).toBeInTheDocument()
  })

  it('renders deep dive view when selected', () => {
    defaultSelection.view = 'deep_dive'
    render(<CockpitPage collections={defaultCollections} />)
    expect(screen.getByTestId('deep-dive-view')).toBeInTheDocument()
  })

  it('renders c4 diff view when selected', () => {
    defaultSelection.view = 'c4_diff'
    render(<CockpitPage collections={defaultCollections} />)
    expect(screen.getByTestId('c4-diff-view')).toBeInTheDocument()
  })

  it('renders architecture view when selected', () => {
    defaultSelection.view = 'architecture'
    render(<CockpitPage collections={defaultCollections} />)
    expect(screen.getByTestId('architecture-view')).toBeInTheDocument()
  })

  it('renders city view when selected', () => {
    defaultSelection.view = 'city'
    render(<CockpitPage collections={defaultCollections} />)
    expect(screen.getByTestId('city-view')).toBeInTheDocument()
  })

  it('renders evolution view when selected', () => {
    defaultSelection.view = 'evolution'
    render(<CockpitPage collections={defaultCollections} />)
    expect(screen.getByTestId('evolution-view')).toBeInTheDocument()
  })

  it('renders graphrag view when selected', () => {
    defaultSelection.view = 'graphrag'
    render(<CockpitPage collections={defaultCollections} />)
    expect(screen.getByTestId('graphrag-view')).toBeInTheDocument()
  })

  it('renders semantic map view when selected', () => {
    defaultSelection.view = 'semantic_map'
    render(<CockpitPage collections={defaultCollections} />)
    expect(screen.getByTestId('semantic-map-view')).toBeInTheDocument()
  })

  it('renders test matrix view when selected', () => {
    defaultSelection.view = 'test_matrix'
    render(<CockpitPage collections={defaultCollections} />)
    expect(screen.getByTestId('test-matrix-view')).toBeInTheDocument()
  })

  it('renders rebuild readiness view when selected', () => {
    defaultSelection.view = 'rebuild_readiness'
    render(<CockpitPage collections={defaultCollections} />)
    expect(screen.getByTestId('rebuild-readiness-view')).toBeInTheDocument()
  })

  it('renders exports view when selected', () => {
    defaultSelection.view = 'exports'
    render(<CockpitPage collections={defaultCollections} />)
    expect(screen.getByTestId('exports-view')).toBeInTheDocument()
  })

  it('calls onOpenCollections when Go to Collections is clicked (empty state)', async () => {
    const onOpenCollections = vi.fn()
    render(<CockpitPage collections={[]} onOpenCollections={onOpenCollections} />)
    await userEvent.click(screen.getByText('Go to Collections'))
    expect(onOpenCollections).toHaveBeenCalledOnce()
  })

  it('calls onOpenRuns when Go to Runs is clicked (empty state)', async () => {
    const onOpenRuns = vi.fn()
    render(<CockpitPage collections={[]} onOpenRuns={onOpenRuns} />)
    await userEvent.click(screen.getByText('Go to Runs'))
    expect(onOpenRuns).toHaveBeenCalledOnce()
  })
})
