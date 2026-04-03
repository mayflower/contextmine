import { useCallback, useEffect, useMemo, useState } from 'react'

/** Process a batch of Promise.allSettled results into values/errors. */
function processSettledResults<K extends string>(
  entries: ReadonlyArray<{ key: K; result: PromiseSettledResult<unknown>; setter: (v: never) => void; fallbackError: string }>,
): { errors: Record<K, string>; successCount: number } {
  const errors = {} as Record<K, string>
  let successCount = 0
  for (const { key, result, setter, fallbackError } of entries) {
    if (result.status === 'fulfilled') {
      setter(result.value as never)
      successCount += 1
      errors[key] = ''
    } else {
      errors[key] = result.reason instanceof Error ? result.reason.message : fallbackError
    }
  }
  return { errors, successCount }
}

import type {
  Arc42DriftPayload,
  Arc42ViewPayload,
  C4ViewMode,
  CityEntityLevel,
  CityProjection,
  CityPayload,
  FitnessFunctionsPayload,
  GraphRagCommunity,
  GraphRagCommunityMode,
  GraphRagCommunitiesPayload,
  GraphRagPathPayload,
  GraphRagProcessDetailPayload,
  GraphRagProcessSummary,
  GraphRagStatusReason,
  SemanticMapMode,
  SemanticMapPayload,
  SemanticMapThresholds,
  RebuildReadinessPayload,
  CockpitLayer,
  CockpitLoadState,
  CockpitProjection,
  CockpitSelection,
  CockpitView,
  DeepDiveMode,
  ExportFormat,
  GraphFilters,
  GraphRagEvidenceItem,
  GraphRagEvidencePayload,
  GraphRagPayload,
  GraphViewPayload,
  GraphNeighborhoodResponse,
  GraphPagingState,
  InvestmentUtilizationPayload,
  KnowledgeIslandsPayload,
  ErmViewPayload,
  PortsAdaptersPayload,
  TestMatrixPayload,
  UIMapPayload,
  UserFlowsPayload,
  MermaidPayload,
  ScenarioLite,
  TemporalCouplingPayload,
  PortsDirection,
  TwinGraphResponse,
} from '../types'

type DataStates = Record<CockpitView, CockpitLoadState>
type DataErrors = Record<CockpitView, string>
type DataUpdated = Partial<Record<CockpitView, string>>
type ArchitectureActionState = {
  reindexState: CockpitLoadState
  reindexMessage: string
  regenerateState: CockpitLoadState
  regenerateMessage: string
}

const DEFAULT_GRAPH: TwinGraphResponse = {
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

const DEFAULT_ARCHITECTURE_ACTION_STATE: ArchitectureActionState = {
  reindexState: 'idle',
  reindexMessage: '',
  regenerateState: 'idle',
  regenerateMessage: '',
}

interface UseCockpitDataArgs {
  selection: CockpitSelection
  behaviorGraphMode: 'ui_map' | 'user_flows'
  topologyLimit: number
  deepDiveLimit: number
  deepDiveMode: DeepDiveMode
  c4View: C4ViewMode
  c4Scope: string
  c4MaxNodes: number
  architectureSection: string
  portsDirection: PortsDirection
  portsContainer: string
  driftBaselineScenarioId: string
  graphFilters: GraphFilters
  graphPaging: GraphPagingState
  graphRagCommunityMode: GraphRagCommunityMode
  graphRagCommunityId: string
  semanticMapMode: SemanticMapMode
  semanticMapThresholdsByMode: Record<SemanticMapMode, SemanticMapThresholds>
  selectedNodeId: string
  onScenarioAutoSelect: (scenarioId: string) => void
  onViewError?: (view: CockpitView, message: string) => void
}

function topologyEntityLevel(layer: CockpitLayer): 'domain' | 'container' | 'component' {
  if (layer === 'portfolio_system') return 'domain'
  if (layer === 'component_interface') return 'component'
  return 'container'
}

export function useCockpitData({
  selection,
  behaviorGraphMode,
  topologyLimit,
  deepDiveLimit,
  deepDiveMode,
  c4View,
  c4Scope,
  c4MaxNodes,
  architectureSection,
  portsDirection,
  portsContainer,
  driftBaselineScenarioId,
  graphFilters,
  graphPaging,
  graphRagCommunityMode,
  graphRagCommunityId,
  semanticMapMode,
  semanticMapThresholdsByMode,
  selectedNodeId,
  onScenarioAutoSelect,
  onViewError,
}: UseCockpitDataArgs) {
  const [scenarios, setScenarios] = useState<ScenarioLite[]>([])
  const [scenariosState, setScenariosState] = useState<CockpitLoadState>('idle')
  const [city, setCity] = useState<CityPayload | null>(null)
  const [graph, setGraph] = useState<TwinGraphResponse>(DEFAULT_GRAPH)
  const [mermaid, setMermaid] = useState<MermaidPayload | null>(null)
  const [arc42, setArc42] = useState<Arc42ViewPayload | null>(null)
  const [portsAdapters, setPortsAdapters] = useState<PortsAdaptersPayload | null>(null)
  const [arc42Drift, setArc42Drift] = useState<Arc42DriftPayload | null>(null)
  const [erm, setErm] = useState<ErmViewPayload | null>(null)
  const [architecturePanelErrors, setArchitecturePanelErrors] = useState<{
    arc42: string
    ports: string
    drift: string
    erm: string
  }>({
    arc42: '',
    ports: '',
    drift: '',
    erm: '',
  })
  const [architectureActions, setArchitectureActions] = useState<ArchitectureActionState>(
    DEFAULT_ARCHITECTURE_ACTION_STATE,
  )
  const [states, setStates] = useState<DataStates>(DEFAULT_STATES)
  const [errors, setErrors] = useState<DataErrors>(DEFAULT_ERRORS)
  const [updatedAt, setUpdatedAt] = useState<DataUpdated>({})
  const [refreshNonce, setRefreshNonce] = useState(0)
  const [cityProjection, setCityProjection] = useState<CityProjection>('architecture')
  const [cityEntityLevel, setCityEntityLevel] = useState<CityEntityLevel>('container')
  const [cityEmbedUrl, setCityEmbedUrl] = useState('')
  const [investmentUtilization, setInvestmentUtilization] = useState<InvestmentUtilizationPayload | null>(null)
  const [knowledgeIslands, setKnowledgeIslands] = useState<KnowledgeIslandsPayload | null>(null)
  const [temporalCoupling, setTemporalCoupling] = useState<TemporalCouplingPayload | null>(null)
  const [fitnessFunctions, setFitnessFunctions] = useState<FitnessFunctionsPayload | null>(null)
  const [evolutionPanelErrors, setEvolutionPanelErrors] = useState<{
    investment: string
    knowledge: string
    coupling: string
    fitness: string
  }>({
    investment: '',
    knowledge: '',
    coupling: '',
    fitness: '',
  })
  const [exportFormat, setExportFormat] = useState<ExportFormat>('cc_json')
  const [exportProjection, setExportProjection] = useState<CockpitProjection>('architecture')
  const [exportContent, setExportContent] = useState('')
  const [neighborhood, setNeighborhood] = useState<TwinGraphResponse>(DEFAULT_GRAPH)
  const [neighborhoodState, setNeighborhoodState] = useState<CockpitLoadState>('idle')
  const [neighborhoodError, setNeighborhoodError] = useState('')
  const [graphRagStatus, setGraphRagStatus] = useState<'ready' | 'unavailable'>('ready')
  const [graphRagReason, setGraphRagReason] = useState<GraphRagStatusReason>('ok')
  const [graphRagEvidenceItems, setGraphRagEvidenceItems] = useState<GraphRagEvidenceItem[]>([])
  const [graphRagEvidenceTotal, setGraphRagEvidenceTotal] = useState(0)
  const [graphRagEvidenceNodeName, setGraphRagEvidenceNodeName] = useState('')
  const [graphRagEvidenceState, setGraphRagEvidenceState] = useState<CockpitLoadState>('idle')
  const [graphRagEvidenceError, setGraphRagEvidenceError] = useState('')
  const [graphRagCommunities, setGraphRagCommunities] = useState<GraphRagCommunity[]>([])
  const [graphRagCommunitiesState, setGraphRagCommunitiesState] = useState<CockpitLoadState>('idle')
  const [graphRagCommunitiesError, setGraphRagCommunitiesError] = useState('')
  const [graphRagPath, setGraphRagPath] = useState<GraphRagPathPayload | null>(null)
  const [graphRagPathState, setGraphRagPathState] = useState<CockpitLoadState>('idle')
  const [graphRagPathError, setGraphRagPathError] = useState('')
  const [graphRagProcesses, setGraphRagProcesses] = useState<GraphRagProcessSummary[]>([])
  const [graphRagProcessesState, setGraphRagProcessesState] = useState<CockpitLoadState>('idle')
  const [graphRagProcessesError, setGraphRagProcessesError] = useState('')
  const [graphRagProcessDetail, setGraphRagProcessDetail] = useState<GraphRagProcessDetailPayload | null>(null)
  const [graphRagProcessDetailState, setGraphRagProcessDetailState] = useState<CockpitLoadState>('idle')
  const [graphRagProcessDetailError, setGraphRagProcessDetailError] = useState('')
  const [uiMapSummary, setUiMapSummary] = useState<UIMapPayload['summary'] | null>(null)
  const [uiMapGraph, setUiMapGraph] = useState<TwinGraphResponse>(DEFAULT_GRAPH)
  const [userFlowsGraph, setUserFlowsGraph] = useState<TwinGraphResponse>(DEFAULT_GRAPH)
  const [semanticMap, setSemanticMap] = useState<SemanticMapPayload | null>(null)
  const [semanticMapComparison, setSemanticMapComparison] = useState<SemanticMapPayload | null>(null)
  const [testMatrix, setTestMatrix] = useState<TestMatrixPayload | null>(null)
  const [userFlows, setUserFlows] = useState<UserFlowsPayload | null>(null)
  const [rebuildReadiness, setRebuildReadiness] = useState<RebuildReadinessPayload | null>(null)

  const setViewState = useCallback((view: CockpitView, nextState: CockpitLoadState) => {
    setStates((prev) => ({ ...prev, [view]: nextState }))
  }, [])

  const setViewError = useCallback(
    (view: CockpitView, message: string) => {
      setErrors((prev) => ({ ...prev, [view]: message }))
      if (message && onViewError) {
        onViewError(view, message)
      }
    },
    [onViewError],
  )

  const markUpdated = useCallback((view: CockpitView) => {
    setUpdatedAt((prev) => ({ ...prev, [view]: new Date().toISOString() }))
  }, [])

  const refreshActiveView = useCallback(() => {
    setRefreshNonce((prev) => prev + 1)
  }, [])

  const parseApiErrorMessage = useCallback(
    async (response: Response, fallback: string): Promise<string> => {
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
    },
    [],
  )

  useEffect(() => {
    setArchitectureActions(DEFAULT_ARCHITECTURE_ACTION_STATE)
  }, [selection.collectionId, selection.scenarioId])

  useEffect(() => {
    if (!selection.collectionId) {
      setScenarios([])
      setScenariosState('empty')
      return
    }

    const controller = new AbortController()

    const run = async () => {
      setScenariosState('loading')
      try {
        const response = await fetch(`/api/twin/scenarios?collection_id=${selection.collectionId}`, {
          credentials: 'include',
          signal: controller.signal,
        })

        if (!response.ok) {
          throw new Error(`Could not load scenarios (${response.status})`)
        }

        const payload = await response.json()
        const nextScenarios: ScenarioLite[] = payload.scenarios || []
        setScenarios(nextScenarios)

        if (nextScenarios.length === 0) {
          setScenariosState('empty')
          return
        }

        const selectedScenario = nextScenarios.find((scenario) => scenario.id === selection.scenarioId)
        if (!selectedScenario) {
          const asIs = nextScenarios.find((scenario) => scenario.is_as_is)
          onScenarioAutoSelect(asIs?.id || nextScenarios[0].id)
        }

        setScenariosState('ready')
      } catch (error) {
        if (controller.signal.aborted) {
          return
        }
        setScenariosState('error')
        setScenarios([])
        setViewError(selection.view, error instanceof Error ? error.message : 'Could not load scenarios')
      }
    }

    run()

    return () => {
      controller.abort()
    }
  }, [selection.collectionId, selection.scenarioId, selection.view, onScenarioAutoSelect, setViewError])

  useEffect(() => {
    const { collectionId, scenarioId, view, layer } = selection
    if (!collectionId || !scenarioId) {
      setViewState(view, 'empty')
      return
    }

    if (view === 'exports') {
      setViewState('exports', 'ready')
      return
    }

    const controller = new AbortController()

    const fetchJson = async <T>(url: string, label: string): Promise<T> => {
      const response = await fetch(url, {
        credentials: 'include',
        signal: controller.signal,
      })
      if (!response.ok) {
        throw new Error(`Could not load ${label} (${response.status})`)
      }
      return (await response.json()) as T
    }

    const fetchOverview = async () => {
      const payload = await fetchJson<CityPayload>(
        `/api/twin/collections/${collectionId}/views/city?scenario_id=${scenarioId}&hotspots_limit=60`,
        'overview',
      )
      setCity(payload)
      setViewState('overview', 'ready')
      markUpdated('overview')
    }

    const fetchTopologyOrDeepDive = async () => {
      const endpoint = view === 'topology' ? 'topology' : 'deep-dive'
      const fallbackLimit = view === 'topology' ? topologyLimit : deepDiveLimit
      const limit = graphPaging.limit > 0 ? graphPaging.limit : fallbackLimit
      const query = new URLSearchParams({
        scenario_id: scenarioId,
        layer,
        limit: String(limit),
        page: String(graphPaging.page),
      })

      if (graphFilters.includeKinds.length > 0) {
        query.set('include_kinds', graphFilters.includeKinds.join(','))
      }
      if (graphFilters.excludeKinds.length > 0) {
        query.set('exclude_kinds', graphFilters.excludeKinds.join(','))
      }

      if (view === 'topology') {
        query.set('projection', 'architecture')
        query.set('entity_level', topologyEntityLevel(layer))
      } else if (deepDiveMode === 'file_dependency') {
        query.set('projection', 'code_file')
        query.set('entity_level', 'file')
      } else if (deepDiveMode === 'symbol_callgraph') {
        query.set('projection', 'code_symbol')
        query.set('entity_level', 'symbol')
        query.set('mode', 'symbol_callgraph')
      } else {
        query.set('projection', 'code_symbol')
        query.set('entity_level', 'symbol')
        query.set('mode', 'contains_hierarchy')
      }

      const payload = await fetchJson<GraphViewPayload>(
        `/api/twin/collections/${collectionId}/views/${endpoint}?${query.toString()}`,
        endpoint,
      )
      const g = payload.graph || DEFAULT_GRAPH
      setGraph({
        ...g,
        projection: payload.projection ?? g.projection,
        entity_level: payload.entity_level ?? g.entity_level,
        grouping_strategy: payload.grouping_strategy ?? g.grouping_strategy,
        excluded_kinds: payload.excluded_kinds ?? g.excluded_kinds,
      })
      setViewState(view, 'ready')
      markUpdated(view)
    }

    const fetchUiMap = async () => {
      const limit = graphPaging.limit > 0 ? graphPaging.limit : topologyLimit
      const query = new URLSearchParams({
        scenario_id: scenarioId,
        limit: String(limit),
        page: String(graphPaging.page),
      })

      const loadEndpoint = async <T>(endpoint: 'ui-map' | 'user-flows'): Promise<T> => {
        return fetchJson<T>(
          `/api/twin/collections/${collectionId}/views/${endpoint}?${query.toString()}`,
          endpoint,
        )
      }

      const [uiResult, flowsResult] = await Promise.allSettled([
        loadEndpoint<UIMapPayload>('ui-map'),
        loadEndpoint<UserFlowsPayload>('user-flows'),
      ])

      let successCount = 0
      const viewErrors: string[] = []
      let nextUiGraph = DEFAULT_GRAPH
      let nextFlowsGraph = DEFAULT_GRAPH

      if (uiResult.status === 'fulfilled') {
        setUiMapSummary(uiResult.value.summary)
        nextUiGraph = {
          ...(uiResult.value.graph || DEFAULT_GRAPH),
          projection: 'code_symbol',
          entity_level: uiResult.value.entity_level,
        }
        setUiMapGraph(nextUiGraph)
        successCount += 1
      } else {
        setUiMapSummary(null)
        setUiMapGraph(DEFAULT_GRAPH)
        viewErrors.push(uiResult.reason instanceof Error ? uiResult.reason.message : 'Could not load ui-map')
      }

      if (flowsResult.status === 'fulfilled') {
        setUserFlows(flowsResult.value)
        nextFlowsGraph = {
          ...(flowsResult.value.graph || DEFAULT_GRAPH),
          projection: 'code_symbol',
          entity_level: flowsResult.value.entity_level,
        }
        setUserFlowsGraph(nextFlowsGraph)
        successCount += 1
      } else {
        setUserFlows(null)
        setUserFlowsGraph(DEFAULT_GRAPH)
        viewErrors.push(
          flowsResult.reason instanceof Error
            ? flowsResult.reason.message
            : 'Could not load user-flows',
        )
      }

      setGraph(nextUiGraph.total_nodes > 0 ? nextUiGraph : nextFlowsGraph)
      setViewError('ui_map', viewErrors.length > 0 ? viewErrors.join(' • ') : '')
      if (successCount > 0) {
        setViewState('ui_map', 'ready')
        markUpdated('ui_map')
      } else {
        setViewState('ui_map', 'error')
      }
    }

    const fetchTestMatrix = async () => {
      const limit = graphPaging.limit > 0 ? graphPaging.limit : topologyLimit
      const query = new URLSearchParams({
        scenario_id: scenarioId,
        limit: String(limit),
        page: String(graphPaging.page),
      })
      const payload = await fetchJson<TestMatrixPayload>(
        `/api/twin/collections/${collectionId}/views/test-matrix?${query.toString()}`,
        'test-matrix',
      )
      setTestMatrix(payload)
      setGraph({
        ...(payload.graph || DEFAULT_GRAPH),
        projection: 'code_symbol',
        entity_level: payload.entity_level,
      })
      setViewState('test_matrix', 'ready')
      markUpdated('test_matrix')
    }

    const fetchSemanticMap = async () => {
      const limit = graphPaging.limit > 0 ? graphPaging.limit : topologyLimit
      const buildMapQuery = (mode: SemanticMapMode) => {
        const thresholds = semanticMapThresholdsByMode[mode]
        const q = new URLSearchParams({
          scenario_id: scenarioId,
          map_mode: mode,
          limit: String(limit),
          page: String(graphPaging.page),
          mixed_cluster_max_dominant_ratio: String(thresholds.mixed_cluster_max_dominant_ratio),
          isolated_distance_multiplier: String(thresholds.isolated_distance_multiplier),
          semantic_duplication_min_similarity: String(thresholds.semantic_duplication_min_similarity),
          semantic_duplication_max_source_overlap: String(thresholds.semantic_duplication_max_source_overlap),
          misplaced_min_dominant_ratio: String(thresholds.misplaced_min_dominant_ratio),
        })
        if (graphFilters.includeKinds.length > 0) q.set('include_kinds', graphFilters.includeKinds.join(','))
        if (graphFilters.excludeKinds.length > 0) q.set('exclude_kinds', graphFilters.excludeKinds.join(','))
        if (graphFilters.edgeKinds.length > 0) q.set('edge_kinds', graphFilters.edgeKinds.join(','))
        return q
      }

      const comparisonMode: SemanticMapMode = semanticMapMode === 'semantic' ? 'code_structure' : 'semantic'
      const [activeMapResult, comparisonMapResult] = await Promise.allSettled([
        fetchJson<SemanticMapPayload>(
          `/api/twin/collections/${collectionId}/views/semantic-map?${buildMapQuery(semanticMapMode).toString()}`,
          'semantic map',
        ),
        fetchJson<SemanticMapPayload>(
          `/api/twin/collections/${collectionId}/views/semantic-map?${buildMapQuery(comparisonMode).toString()}`,
          'semantic map comparison',
        ),
      ])

      if (activeMapResult.status !== 'fulfilled') {
        throw activeMapResult.reason
      }

      setSemanticMap(activeMapResult.value)
      setSemanticMapComparison(comparisonMapResult.status === 'fulfilled' ? comparisonMapResult.value : null)
      setGraph(DEFAULT_GRAPH)
      setViewState('semantic_map', 'ready')
      markUpdated('semantic_map')
    }

    const fetchRebuildReadiness = async () => {
      const payload = await fetchJson<RebuildReadinessPayload>(
        `/api/twin/collections/${collectionId}/views/rebuild-readiness?scenario_id=${encodeURIComponent(scenarioId)}`,
        'rebuild-readiness',
      )
      setRebuildReadiness(payload)
      setViewState('rebuild_readiness', 'ready')
      markUpdated('rebuild_readiness')
    }

    const fetchArchitecture = async () => {
      const section = architectureSection.trim()
      const normalizedContainer = portsContainer.trim()
      const baselineScenarioId = driftBaselineScenarioId.trim()

      const arc42Query = new URLSearchParams({ scenario_id: scenarioId })
      if (section) arc42Query.set('section', section)

      const portsQuery = new URLSearchParams({ scenario_id: scenarioId })
      if (portsDirection !== 'all') portsQuery.set('direction', portsDirection)
      if (normalizedContainer) portsQuery.set('container', normalizedContainer)

      const driftQuery = new URLSearchParams({ scenario_id: scenarioId })
      if (baselineScenarioId) driftQuery.set('baseline_scenario_id', baselineScenarioId)

      const [arc42Result, portsResult, driftResult, ermResult] = await Promise.allSettled([
        fetchJson<Arc42ViewPayload>(`/api/twin/collections/${collectionId}/views/arc42?${arc42Query.toString()}`, 'arc42 view'),
        fetchJson<PortsAdaptersPayload>(`/api/twin/collections/${collectionId}/views/ports-adapters?${portsQuery.toString()}`, 'ports/adapters view'),
        fetchJson<Arc42DriftPayload>(`/api/twin/collections/${collectionId}/views/arc42/drift?${driftQuery.toString()}`, 'arc42 drift view'),
        fetchJson<ErmViewPayload>(`/api/twin/collections/${collectionId}/views/erm?scenario_id=${encodeURIComponent(scenarioId)}`, 'erm view'),
      ])

      const { errors: nextErrors, successCount } = processSettledResults([
        { key: 'arc42' as const, result: arc42Result, setter: setArc42 as (v: never) => void, fallbackError: 'Could not load arc42 view' },
        { key: 'ports' as const, result: portsResult, setter: setPortsAdapters as (v: never) => void, fallbackError: 'Could not load ports/adapters view' },
        { key: 'drift' as const, result: driftResult, setter: setArc42Drift as (v: never) => void, fallbackError: 'Could not load drift view' },
        { key: 'erm' as const, result: ermResult, setter: setErm as (v: never) => void, fallbackError: 'Could not load erm view' },
      ])
      setArchitecturePanelErrors(nextErrors)
      const allErrors = Object.values(nextErrors).filter(Boolean)
      setViewError('architecture', allErrors.length > 0 ? allErrors.join(' • ') : '')
      if (successCount > 0) { setViewState('architecture', 'ready'); markUpdated('architecture') }
      else { setViewState('architecture', 'error') }
    }

    const fetchEvolution = async () => {
      const [investmentResult, knowledgeResult, couplingResult, fitnessResult] =
        await Promise.allSettled([
          fetchJson<InvestmentUtilizationPayload>(`/api/twin/collections/${collectionId}/views/evolution/investment-utilization?scenario_id=${encodeURIComponent(scenarioId)}&entity_level=container&window_days=365`, 'evolution investment/utilization'),
          fetchJson<KnowledgeIslandsPayload>(`/api/twin/collections/${collectionId}/views/evolution/knowledge-islands?scenario_id=${encodeURIComponent(scenarioId)}&entity_level=container&window_days=365&ownership_threshold=0.7`, 'evolution knowledge islands'),
          fetchJson<TemporalCouplingPayload>(`/api/twin/collections/${collectionId}/views/evolution/temporal-coupling?scenario_id=${encodeURIComponent(scenarioId)}&entity_level=component&window_days=365&min_jaccard=0.2&max_edges=300`, 'evolution temporal coupling'),
          fetchJson<FitnessFunctionsPayload>(`/api/twin/collections/${collectionId}/views/evolution/fitness-functions?scenario_id=${encodeURIComponent(scenarioId)}&window_days=365&include_resolved=false`, 'evolution fitness functions'),
        ])

      const { errors: nextErrors, successCount } = processSettledResults([
        { key: 'investment' as const, result: investmentResult, setter: setInvestmentUtilization as (v: never) => void, fallbackError: 'Could not load investment/utilization panel' },
        { key: 'knowledge' as const, result: knowledgeResult, setter: setKnowledgeIslands as (v: never) => void, fallbackError: 'Could not load knowledge islands panel' },
        { key: 'coupling' as const, result: couplingResult, setter: setTemporalCoupling as (v: never) => void, fallbackError: 'Could not load temporal coupling panel' },
        { key: 'fitness' as const, result: fitnessResult, setter: setFitnessFunctions as (v: never) => void, fallbackError: 'Could not load fitness functions panel' },
      ])
      setEvolutionPanelErrors(nextErrors)
      const allErrors = Object.values(nextErrors).filter(Boolean)
      setViewError('evolution', allErrors.length > 0 ? allErrors.join(' • ') : '')
      if (successCount > 0) { setViewState('evolution', 'ready'); markUpdated('evolution') }
      else { setViewState('evolution', 'error') }
    }

    const fetchGraphRagCommunities = async () => {
      setGraphRagCommunitiesState('loading')
      setGraphRagCommunitiesError('')
      try {
        const communitiesPayload = await fetchJson<GraphRagCommunitiesPayload>(
          `/api/twin/collections/${collectionId}/views/graphrag/communities?scenario_id=${encodeURIComponent(scenarioId)}&limit=500`,
          'communities',
        )
        setGraphRagCommunities(communitiesPayload.items || [])
        setGraphRagCommunitiesState((communitiesPayload.items || []).length > 0 ? 'ready' : 'empty')
      } catch (error) {
        if (!controller.signal.aborted) {
          setGraphRagCommunitiesState('error')
          setGraphRagCommunitiesError(error instanceof Error ? error.message : 'Could not load communities')
        }
      }
    }

    const fetchGraphRagProcesses = async () => {
      setGraphRagProcessesState('loading')
      setGraphRagProcessesError('')
      try {
        const processesPayload = await fetchJson<{ items?: GraphRagProcessSummary[] }>(
          `/api/twin/collections/${collectionId}/views/graphrag/processes?scenario_id=${encodeURIComponent(scenarioId)}`,
          'processes',
        )
        setGraphRagProcesses(processesPayload.items || [])
        setGraphRagProcessesState((processesPayload.items || []).length > 0 ? 'ready' : 'empty')
      } catch (error) {
        if (!controller.signal.aborted) {
          setGraphRagProcessesState('error')
          setGraphRagProcessesError(error instanceof Error ? error.message : 'Could not load processes')
        }
      }
    }

    const fetchGraphRag = async () => {
      const query = new URLSearchParams({
        scenario_id: scenarioId,
        limit: String(graphPaging.limit > 0 ? graphPaging.limit : topologyLimit),
        page: String(graphPaging.page),
        community_mode: graphRagCommunityMode,
      })
      const normalizedCommunityId = graphRagCommunityId.trim()
      if (normalizedCommunityId) query.set('community_id', normalizedCommunityId)
      if (graphFilters.includeKinds.length > 0) query.set('include_kinds', graphFilters.includeKinds.join(','))
      if (graphFilters.excludeKinds.length > 0) query.set('exclude_kinds', graphFilters.excludeKinds.join(','))
      if (graphFilters.edgeKinds.length > 0) query.set('edge_kinds', graphFilters.edgeKinds.join(','))

      const payload = await fetchJson<GraphRagPayload>(
        `/api/twin/collections/${collectionId}/views/graphrag?${query.toString()}`,
        'graphrag view',
      )
      setGraph({
        ...(payload.graph || DEFAULT_GRAPH),
        projection: payload.projection,
        entity_level: payload.entity_level,
      })
      setGraphRagStatus(payload.status?.status || 'ready')
      setGraphRagReason(payload.status?.reason || 'ok')

      await fetchGraphRagCommunities()
      await fetchGraphRagProcesses()
      setViewState('graphrag', 'ready')
      markUpdated('graphrag')
    }

    const fetchC4Diff = async () => {
      const query = new URLSearchParams({
        scenario_id: scenarioId,
        compare_with_base: 'true',
        c4_view: c4View,
        max_nodes: String(Math.max(10, c4MaxNodes || 120)),
      })
      const normalizedScope = c4Scope.trim()
      if (normalizedScope) query.set('c4_scope', normalizedScope)

      const payload = await fetchJson<MermaidPayload>(
        `/api/twin/collections/${collectionId}/views/mermaid?${query.toString()}`,
        'C4 view',
      )
      setMermaid(payload)
      setViewState('c4_diff', 'ready')
      markUpdated('c4_diff')
    }

    const fetchCity = async () => {
      const cityProjectionValue: CityProjection = cityProjection
      const entityLevel = cityProjectionValue === 'architecture' ? cityEntityLevel : 'file'

      const exportResponse = await fetch(`/api/twin/scenarios/${scenarioId}/exports`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        signal: controller.signal,
        body: JSON.stringify({ format: 'cc_json', projection: cityProjectionValue, entity_level: entityLevel }),
      })
      if (!exportResponse.ok) {
        throw new Error(`Could not generate city map (${exportResponse.status})`)
      }

      const exportData = await exportResponse.json()
      const exportId = exportData.id || exportData.exports?.[0]?.id
      if (!exportId) {
        throw new Error('Missing export id from city export response')
      }

      const rawPath = `/api/twin/scenarios/${scenarioId}/exports/${exportId}/raw`
      const params = new URLSearchParams({ file: rawPath, area: 'loc', height: 'coupling', color: 'complexity', mode: 'Single' })
      setCityEmbedUrl(`/codecharta/index.html?${params.toString()}`)
      setViewState('city', 'ready')
      markUpdated('city')
    }

    const viewHandlers: Record<string, (() => Promise<void>) | undefined> = {
      overview: fetchOverview,
      topology: fetchTopologyOrDeepDive,
      deep_dive: fetchTopologyOrDeepDive,
      ui_map: fetchUiMap,
      test_matrix: fetchTestMatrix,
      semantic_map: fetchSemanticMap,
      rebuild_readiness: fetchRebuildReadiness,
      architecture: fetchArchitecture,
      evolution: fetchEvolution,
      graphrag: fetchGraphRag,
      c4_diff: fetchC4Diff,
      city: fetchCity,
    }

    const run = async () => {
      setViewState(view, 'loading')
      setViewError(view, '')
      if (view === 'city') {
        setCityEmbedUrl('')
      }

      try {
        const handler = viewHandlers[view]
        if (handler) {
          await handler()
        }
      } catch (error) {
        if (controller.signal.aborted) {
          return
        }
        const message = error instanceof Error ? error.message : 'Unexpected Cockpit request error'
        setViewError(view, message)
        setViewState(view, 'error')
      }
    }

    run()

    return () => {
      controller.abort()
    }
  }, [
    selection,
    topologyLimit,
    deepDiveLimit,
    deepDiveMode,
    c4View,
    c4Scope,
    c4MaxNodes,
    architectureSection,
    portsDirection,
    portsContainer,
    driftBaselineScenarioId,
    graphFilters.excludeKinds,
    graphFilters.edgeKinds,
    graphFilters.includeKinds,
    graphPaging.limit,
    graphPaging.page,
    graphRagCommunityMode,
    graphRagCommunityId,
    semanticMapMode,
    semanticMapThresholdsByMode.code_structure.mixed_cluster_max_dominant_ratio,
    semanticMapThresholdsByMode.code_structure.isolated_distance_multiplier,
    semanticMapThresholdsByMode.code_structure.semantic_duplication_min_similarity,
    semanticMapThresholdsByMode.code_structure.semantic_duplication_max_source_overlap,
    semanticMapThresholdsByMode.code_structure.misplaced_min_dominant_ratio,
    semanticMapThresholdsByMode.semantic.mixed_cluster_max_dominant_ratio,
    semanticMapThresholdsByMode.semantic.isolated_distance_multiplier,
    semanticMapThresholdsByMode.semantic.semantic_duplication_min_similarity,
    semanticMapThresholdsByMode.semantic.semantic_duplication_max_source_overlap,
    semanticMapThresholdsByMode.semantic.misplaced_min_dominant_ratio,
    semanticMapThresholdsByMode,
    cityProjection,
    cityEntityLevel,
    refreshNonce,
    markUpdated,
    setViewError,
    setViewState,
  ])

  useEffect(() => {
    if (selection.view !== 'ui_map') {
      return
    }
    let preferred: TwinGraphResponse
    if (behaviorGraphMode === 'user_flows') {
      preferred = userFlowsGraph.total_nodes > 0 ? userFlowsGraph : uiMapGraph
    } else {
      preferred = uiMapGraph.total_nodes > 0 ? uiMapGraph : userFlowsGraph
    }
    setGraph(preferred)
  }, [selection.view, behaviorGraphMode, uiMapGraph, userFlowsGraph])

  useEffect(() => {
    const { collectionId, scenarioId, view } = selection
    if (!collectionId || !scenarioId || view !== 'semantic_map' || !selectedNodeId) {
      return
    }

    const controller = new AbortController()
    const run = async () => {
      try {
        const limit = graphPaging.limit > 0 ? graphPaging.limit : topologyLimit
        const query = new URLSearchParams({
          scenario_id: scenarioId,
          community_mode: 'color',
          limit: String(limit),
          page: String(graphPaging.page),
        })
        if (graphFilters.includeKinds.length > 0) {
          query.set('include_kinds', graphFilters.includeKinds.join(','))
        }
        if (graphFilters.excludeKinds.length > 0) {
          query.set('exclude_kinds', graphFilters.excludeKinds.join(','))
        }
        if (graphFilters.edgeKinds.length > 0) {
          query.set('edge_kinds', graphFilters.edgeKinds.join(','))
        }

        const response = await fetch(
          `/api/twin/collections/${collectionId}/views/graphrag?${query.toString()}`,
          {
            credentials: 'include',
            signal: controller.signal,
          },
        )
        if (!response.ok) {
          return
        }
        const payload: GraphRagPayload = await response.json()
        if (controller.signal.aborted) {
          return
        }
        setGraph({
          ...(payload.graph || DEFAULT_GRAPH),
          projection: payload.projection,
          entity_level: payload.entity_level,
        })
      } catch {
        // Keep semantic map usable even if optional graph context cannot be loaded.
      }
    }

    run()
    return () => {
      controller.abort()
    }
  }, [
    selection,
    selectedNodeId,
    graphFilters.includeKinds,
    graphFilters.excludeKinds,
    graphFilters.edgeKinds,
    graphPaging.limit,
    graphPaging.page,
    topologyLimit,
    refreshNonce,
  ])

  useEffect(() => {
    const { collectionId, scenarioId, view } = selection
    if (!collectionId || !scenarioId || view !== 'graphrag') {
      setGraphRagEvidenceItems([])
      setGraphRagEvidenceTotal(0)
      setGraphRagEvidenceNodeName('')
      setGraphRagEvidenceState('idle')
      setGraphRagEvidenceError('')
      return
    }
    if (!selectedNodeId) {
      setGraphRagEvidenceItems([])
      setGraphRagEvidenceTotal(0)
      setGraphRagEvidenceNodeName('')
      setGraphRagEvidenceState('empty')
      setGraphRagEvidenceError('')
      return
    }

    const controller = new AbortController()
    const run = async () => {
      setGraphRagEvidenceState('loading')
      setGraphRagEvidenceError('')
      try {
        const query = new URLSearchParams({
          scenario_id: scenarioId,
          node_id: selectedNodeId,
          limit: '50',
        })
        const response = await fetch(
          `/api/twin/collections/${collectionId}/views/graphrag/evidence?${query.toString()}`,
          {
            credentials: 'include',
            signal: controller.signal,
          },
        )
        if (!response.ok) {
          throw new Error(`Could not load node evidence (${response.status})`)
        }
        const payload: GraphRagEvidencePayload = await response.json()
        setGraphRagEvidenceItems(payload.items || [])
        setGraphRagEvidenceTotal(payload.total || 0)
        setGraphRagEvidenceNodeName(payload.node_name || '')
        setGraphRagEvidenceState((payload.items || []).length > 0 ? 'ready' : 'empty')
      } catch (error) {
        if (controller.signal.aborted) {
          return
        }
        setGraphRagEvidenceState('error')
        setGraphRagEvidenceError(
          error instanceof Error ? error.message : 'Could not load node evidence',
        )
      }
    }

    run()
    return () => {
      controller.abort()
    }
  }, [selection, selectedNodeId, refreshNonce])

  useEffect(() => {
    if (selection.view === 'graphrag') {
      return
    }
    setGraphRagCommunities([])
    setGraphRagCommunitiesState('idle')
    setGraphRagCommunitiesError('')
    setGraphRagPath(null)
    setGraphRagPathState('idle')
    setGraphRagPathError('')
    setGraphRagProcesses([])
    setGraphRagProcessesState('idle')
    setGraphRagProcessesError('')
    setGraphRagProcessDetail(null)
    setGraphRagProcessDetailState('idle')
    setGraphRagProcessDetailError('')
  }, [selection.view])

  useEffect(() => {
    const { scenarioId, view } = selection
    if (!scenarioId || !selectedNodeId || (view !== 'topology' && view !== 'deep_dive')) {
      setNeighborhood(DEFAULT_GRAPH)
      setNeighborhoodState('idle')
      setNeighborhoodError('')
      return
    }

    const controller = new AbortController()
    const run = async () => {
      setNeighborhoodState('loading')
      setNeighborhoodError('')
      try {
        let projection: string
        if (view === 'topology') {
          projection = 'architecture'
        } else if (deepDiveMode === 'file_dependency') {
          projection = 'code_file'
        } else {
          projection = 'code_symbol'
        }
        const response = await fetch(
          `/api/twin/scenarios/${scenarioId}/graph/neighborhood?node_id=${encodeURIComponent(selectedNodeId)}&projection=${projection}&hops=1&limit=200`,
          {
            credentials: 'include',
            signal: controller.signal,
          },
        )
        if (!response.ok) {
          throw new Error(`Could not load neighborhood (${response.status})`)
        }
        const payload: GraphNeighborhoodResponse = await response.json()
        setNeighborhood(payload.graph || DEFAULT_GRAPH)
        setNeighborhoodState('ready')
      } catch (error) {
        if (controller.signal.aborted) {
          return
        }
        const message = error instanceof Error ? error.message : 'Could not load neighborhood'
        setNeighborhoodState('error')
        setNeighborhoodError(message)
      }
    }

    run()
    return () => {
      controller.abort()
    }
  }, [selection, selectedNodeId, deepDiveMode])

  const triggerCollectionReindex = useCallback(async () => {
    const collectionId = selection.collectionId
    if (!collectionId) {
      setArchitectureActions((prev) => ({
        ...prev,
        reindexState: 'error',
        reindexMessage: 'Select a project before starting reindexing.',
      }))
      return false
    }

    setArchitectureActions((prev) => ({
      ...prev,
      reindexState: 'loading',
      reindexMessage: 'Reindexing started...',
    }))

    try {
      const response = await fetch(`/api/twin/collections/${collectionId}/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ force: true }),
      })
      if (!response.ok) {
        const detail = await parseApiErrorMessage(response, 'Could not start reindexing')
        throw new Error(detail)
      }

      const payload = await response.json()
      const created = Number(payload?.created || 0)
      const skipped = Number(payload?.skipped || 0)
      let message: string
      if (created > 0) {
        message = `Reindexing queued for ${created} source(s).`
      } else if (skipped > 0) {
        message = `No new source revisions queued (${skipped} unchanged).`
      } else {
        message = 'Reindexing request accepted.'
      }
      setArchitectureActions((prev) => ({
        ...prev,
        reindexState: 'ready',
        reindexMessage: message,
      }))
      refreshActiveView()
      return true
    } catch (error) {
      setArchitectureActions((prev) => ({
        ...prev,
        reindexState: 'error',
        reindexMessage: error instanceof Error ? error.message : 'Could not start reindexing.',
      }))
      return false
    }
  }, [parseApiErrorMessage, refreshActiveView, selection.collectionId])

  const regenerateArc42 = useCallback(async () => {
    const { collectionId, scenarioId } = selection
    if (!collectionId || !scenarioId) {
      setArchitectureActions((prev) => ({
        ...prev,
        regenerateState: 'error',
        regenerateMessage: 'Select project and scenario before regenerating arc42.',
      }))
      return false
    }

    setArchitectureActions((prev) => ({
      ...prev,
      regenerateState: 'loading',
      regenerateMessage: 'Generating arc42...',
    }))

    try {
      const query = new URLSearchParams({
        scenario_id: scenarioId,
        regenerate: 'true',
      })
      const section = architectureSection.trim()
      if (section) {
        query.set('section', section)
      }
      const response = await fetch(
        `/api/twin/collections/${collectionId}/views/arc42?${query.toString()}`,
        {
          credentials: 'include',
        },
      )
      if (!response.ok) {
        const detail = await parseApiErrorMessage(response, 'Could not regenerate arc42')
        throw new Error(detail)
      }
      const payload: Arc42ViewPayload = await response.json()
      setArc42(payload)
      setArchitecturePanelErrors((prev) => ({ ...prev, arc42: '' }))
      setArchitectureActions((prev) => ({
        ...prev,
        regenerateState: 'ready',
        regenerateMessage: 'arc42 regenerated successfully.',
      }))
      refreshActiveView()
      return true
    } catch (error) {
      setArchitectureActions((prev) => ({
        ...prev,
        regenerateState: 'error',
        regenerateMessage: error instanceof Error ? error.message : 'Could not regenerate arc42.',
      }))
      return false
    }
  }, [
    architectureSection,
    parseApiErrorMessage,
    refreshActiveView,
    selection.collectionId,
    selection.scenarioId,
  ])

  const generateExport = useCallback(async () => {
    const scenarioId = selection.scenarioId
    if (!scenarioId) {
      setViewError('exports', 'Select a scenario before generating an export.')
      setViewState('exports', 'error')
      return null
    }

    setViewError('exports', '')
    setViewState('exports', 'loading')

    try {
      let entityLevel: string
      if (exportProjection === 'architecture') {
        entityLevel = topologyEntityLevel(selection.layer)
      } else if (exportProjection === 'code_file') {
        entityLevel = 'file'
      } else {
        entityLevel = 'symbol'
      }

      const exportResponse = await fetch(`/api/twin/scenarios/${scenarioId}/exports`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          format: exportFormat,
          projection: exportProjection,
          entity_level: entityLevel,
        }),
      })

      if (!exportResponse.ok) {
        throw new Error(`Could not generate export (${exportResponse.status})`)
      }

      const exportData = await exportResponse.json()
      const exportId = exportData.id || exportData.exports?.[0]?.id
      if (!exportId) {
        throw new Error('Missing export id from API response')
      }

      const artifactResponse = await fetch(`/api/twin/scenarios/${scenarioId}/exports/${exportId}`, {
        credentials: 'include',
      })

      if (!artifactResponse.ok) {
        throw new Error(`Could not fetch export artifact (${artifactResponse.status})`)
      }

      const artifact = await artifactResponse.json()
      const content = artifact.content || ''
      setExportContent(content)
      setViewState('exports', 'ready')
      markUpdated('exports')
      return {
        content,
        name: artifact.name || `${exportFormat}.txt`,
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Export generation failed'
      setViewError('exports', message)
      setViewState('exports', 'error')
      return null
    }
  }, [
    exportFormat,
    exportProjection,
    markUpdated,
    selection.layer,
    selection.scenarioId,
    setViewError,
    setViewState,
  ])

  const traceGraphRagPath = useCallback(
    async (fromNodeId: string, toNodeId: string, maxHops: number) => {
      const { collectionId, scenarioId } = selection
      if (!collectionId || !scenarioId) {
        setGraphRagPathState('error')
        setGraphRagPathError('Select a project and scenario before tracing a path.')
        return null
      }

      const normalizedFrom = fromNodeId.trim()
      const normalizedTo = toNodeId.trim()
      if (!normalizedFrom || !normalizedTo) {
        setGraphRagPathState('error')
        setGraphRagPathError('Both from/to node IDs are required.')
        return null
      }

      setGraphRagPathState('loading')
      setGraphRagPathError('')

      try {
        const query = new URLSearchParams({
          scenario_id: scenarioId,
          from_node_id: normalizedFrom,
          to_node_id: normalizedTo,
          max_hops: String(Math.max(1, Math.min(20, maxHops))),
        })
        const response = await fetch(
          `/api/twin/collections/${collectionId}/views/graphrag/path?${query.toString()}`,
          {
            credentials: 'include',
          },
        )
        if (!response.ok) {
          throw new Error(`Could not trace path (${response.status})`)
        }
        const payload: GraphRagPathPayload = await response.json()
        setGraphRagPath(payload)
        setGraphRagPathState(payload.status === 'found' ? 'ready' : 'empty')
        return payload
      } catch (error) {
        setGraphRagPathState('error')
        setGraphRagPathError(error instanceof Error ? error.message : 'Could not trace path')
        return null
      }
    },
    [selection],
  )

  const loadGraphRagProcessDetail = useCallback(
    async (processId: string) => {
      const { collectionId, scenarioId } = selection
      if (!collectionId || !scenarioId || !processId) {
        setGraphRagProcessDetailState('error')
        setGraphRagProcessDetailError('Missing process context.')
        return null
      }
      setGraphRagProcessDetailState('loading')
      setGraphRagProcessDetailError('')
      try {
        const response = await fetch(
          `/api/twin/collections/${collectionId}/views/graphrag/processes/${encodeURIComponent(processId)}?scenario_id=${encodeURIComponent(scenarioId)}`,
          {
            credentials: 'include',
          },
        )
        if (!response.ok) {
          throw new Error(`Could not load process detail (${response.status})`)
        }
        const payload: GraphRagProcessDetailPayload = await response.json()
        setGraphRagProcessDetail(payload)
        setGraphRagProcessDetailState('ready')
        return payload
      } catch (error) {
        setGraphRagProcessDetailState('error')
        setGraphRagProcessDetailError(
          error instanceof Error ? error.message : 'Could not load process detail',
        )
        return null
      }
    },
    [selection],
  )

  const activeState = useMemo(() => states[selection.view], [states, selection.view])
  const activeError = useMemo(() => errors[selection.view], [errors, selection.view])
  const activeUpdatedAt = useMemo(
    () => updatedAt[selection.view] ?? null,
    [updatedAt, selection.view],
  )

  return {
    scenarios,
    scenariosState,
    city,
    graph,
    mermaid,
    arc42,
    portsAdapters,
    arc42Drift,
    erm,
    architecturePanelErrors,
    architectureActions,
    states,
    errors,
    activeState,
    activeError,
    activeUpdatedAt,
    cityProjection,
    setCityProjection,
    cityEntityLevel,
    setCityEntityLevel,
    cityEmbedUrl,
    setCityEmbedUrl,
    investmentUtilization,
    knowledgeIslands,
    temporalCoupling,
    fitnessFunctions,
    evolutionPanelErrors,
    exportFormat,
    setExportFormat,
    exportProjection,
    setExportProjection,
    exportContent,
    setExportContent,
    neighborhood,
    neighborhoodState,
    neighborhoodError,
    graphRagStatus,
    graphRagReason,
    graphRagEvidenceItems,
    graphRagEvidenceTotal,
    graphRagEvidenceNodeName,
    graphRagEvidenceState,
    graphRagEvidenceError,
    graphRagCommunities,
    graphRagCommunitiesState,
    graphRagCommunitiesError,
    graphRagPath,
    graphRagPathState,
    graphRagPathError,
    graphRagProcesses,
    graphRagProcessesState,
    graphRagProcessesError,
    graphRagProcessDetail,
    graphRagProcessDetailState,
    graphRagProcessDetailError,
    uiMapSummary,
    uiMapGraph,
    userFlowsGraph,
    semanticMap,
    semanticMapComparison,
    testMatrix,
    userFlows,
    rebuildReadiness,
    traceGraphRagPath,
    loadGraphRagProcessDetail,
    triggerCollectionReindex,
    regenerateArc42,
    generateExport,
    refreshActiveView,
  }
}
