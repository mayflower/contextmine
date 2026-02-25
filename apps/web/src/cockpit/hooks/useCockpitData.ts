import { useCallback, useEffect, useMemo, useState } from 'react'

import type {
  Arc42DriftPayload,
  Arc42ViewPayload,
  C4ViewMode,
  CityEntityLevel,
  CityProjection,
  CityPayload,
  GraphRagCommunity,
  GraphRagCommunityMode,
  GraphRagCommunitiesPayload,
  GraphRagPathPayload,
  GraphRagProcessDetailPayload,
  GraphRagProcessSummary,
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
  GraphNeighborhoodResponse,
  GraphPagingState,
  ErmViewPayload,
  PortsAdaptersPayload,
  TestMatrixPayload,
  UIMapPayload,
  UserFlowsPayload,
  MermaidPayload,
  ScenarioLite,
  TwinGraphResponse,
} from '../types'

type DataStates = Record<CockpitView, CockpitLoadState>
type DataErrors = Record<CockpitView, string>
type DataUpdated = Partial<Record<CockpitView, string>>

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
  graphrag: '',
  ui_map: '',
  semantic_map: '',
  test_matrix: '',
  user_flows: '',
  rebuild_readiness: '',
  exports: '',
}

interface UseCockpitDataArgs {
  selection: CockpitSelection
  topologyLimit: number
  deepDiveLimit: number
  deepDiveMode: DeepDiveMode
  c4View: C4ViewMode
  c4Scope: string
  c4MaxNodes: number
  architectureSection: string
  portsDirection: 'all' | 'inbound' | 'outbound'
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
  const [states, setStates] = useState<DataStates>(DEFAULT_STATES)
  const [errors, setErrors] = useState<DataErrors>(DEFAULT_ERRORS)
  const [updatedAt, setUpdatedAt] = useState<DataUpdated>({})
  const [refreshNonce, setRefreshNonce] = useState(0)
  const [cityProjection, setCityProjection] = useState<CityProjection>('architecture')
  const [cityEntityLevel, setCityEntityLevel] = useState<CityEntityLevel>('container')
  const [cityEmbedUrl, setCityEmbedUrl] = useState('')
  const [exportFormat, setExportFormat] = useState<ExportFormat>('cc_json')
  const [exportProjection, setExportProjection] = useState<CockpitProjection>('architecture')
  const [exportContent, setExportContent] = useState('')
  const [neighborhood, setNeighborhood] = useState<TwinGraphResponse>(DEFAULT_GRAPH)
  const [neighborhoodState, setNeighborhoodState] = useState<CockpitLoadState>('idle')
  const [neighborhoodError, setNeighborhoodError] = useState('')
  const [graphRagStatus, setGraphRagStatus] = useState<'ready' | 'unavailable'>('ready')
  const [graphRagReason, setGraphRagReason] = useState<'ok' | 'no_knowledge_graph'>('ok')
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

    const run = async () => {
      setViewState(view, 'loading')
      setViewError(view, '')
      if (view === 'city') {
        setCityEmbedUrl('')
      }

      try {
        if (view === 'overview') {
          const response = await fetch(
            `/api/twin/collections/${collectionId}/views/city?scenario_id=${scenarioId}&hotspots_limit=60`,
            {
              credentials: 'include',
              signal: controller.signal,
            },
          )
          if (!response.ok) {
            throw new Error(`Could not load overview (${response.status})`)
          }
          const payload: CityPayload = await response.json()
          setCity(payload)
          setViewState('overview', 'ready')
          markUpdated('overview')
          return
        }

        if (view === 'topology' || view === 'deep_dive') {
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
          } else {
            if (deepDiveMode === 'file_dependency') {
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
          }

          const response = await fetch(
            `/api/twin/collections/${collectionId}/views/${endpoint}?${query.toString()}`,
            {
              credentials: 'include',
              signal: controller.signal,
            },
          )
          if (!response.ok) {
            throw new Error(`Could not load ${endpoint} (${response.status})`)
          }
          const payload = await response.json()
          setGraph({
            ...(payload.graph || DEFAULT_GRAPH),
            projection: payload.projection ?? payload.graph?.projection,
            entity_level: payload.entity_level ?? payload.graph?.entity_level,
            grouping_strategy: payload.grouping_strategy ?? payload.graph?.grouping_strategy,
            excluded_kinds: payload.excluded_kinds ?? payload.graph?.excluded_kinds,
          })
          setViewState(view, 'ready')
          markUpdated(view)
          return
        }

        if (view === 'ui_map' || view === 'test_matrix' || view === 'user_flows') {
          const endpoint =
            view === 'ui_map' ? 'ui-map' : view === 'test_matrix' ? 'test-matrix' : 'user-flows'
          const limit = graphPaging.limit > 0 ? graphPaging.limit : topologyLimit
          const query = new URLSearchParams({
            scenario_id: scenarioId,
            limit: String(limit),
            page: String(graphPaging.page),
          })
          const response = await fetch(
            `/api/twin/collections/${collectionId}/views/${endpoint}?${query.toString()}`,
            {
              credentials: 'include',
              signal: controller.signal,
            },
          )
          if (!response.ok) {
            throw new Error(`Could not load ${endpoint} (${response.status})`)
          }
          if (view === 'ui_map') {
            const payload: UIMapPayload = await response.json()
            setUiMapSummary(payload.summary)
            setGraph({
              ...(payload.graph || DEFAULT_GRAPH),
              projection: 'code_symbol',
              entity_level: payload.entity_level,
            })
          } else if (view === 'test_matrix') {
            const payload: TestMatrixPayload = await response.json()
            setTestMatrix(payload)
            setGraph({
              ...(payload.graph || DEFAULT_GRAPH),
              projection: 'code_symbol',
              entity_level: payload.entity_level,
            })
          } else {
            const payload: UserFlowsPayload = await response.json()
            setUserFlows(payload)
            setGraph({
              ...(payload.graph || DEFAULT_GRAPH),
              projection: 'code_symbol',
              entity_level: payload.entity_level,
            })
          }
          setViewState(view, 'ready')
          markUpdated(view)
          return
        }

        if (view === 'semantic_map') {
          const limit = graphPaging.limit > 0 ? graphPaging.limit : topologyLimit
          const buildMapQuery = (mode: SemanticMapMode) => {
            const thresholds = semanticMapThresholdsByMode[mode]
            const query = new URLSearchParams({
              scenario_id: scenarioId,
              map_mode: mode,
              limit: String(limit),
              page: String(graphPaging.page),
              mixed_cluster_max_dominant_ratio: String(
                thresholds.mixed_cluster_max_dominant_ratio,
              ),
              isolated_distance_multiplier: String(thresholds.isolated_distance_multiplier),
              semantic_duplication_min_similarity: String(
                thresholds.semantic_duplication_min_similarity,
              ),
              semantic_duplication_max_source_overlap: String(
                thresholds.semantic_duplication_max_source_overlap,
              ),
              misplaced_min_dominant_ratio: String(
                thresholds.misplaced_min_dominant_ratio,
              ),
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
            return query
          }

          const loadMap = async (mode: SemanticMapMode): Promise<SemanticMapPayload> => {
            const response = await fetch(
              `/api/twin/collections/${collectionId}/views/semantic-map?${buildMapQuery(mode).toString()}`,
              {
                credentials: 'include',
                signal: controller.signal,
              },
            )
            if (!response.ok) {
              throw new Error(`Could not load semantic map (${response.status})`)
            }
            return (await response.json()) as SemanticMapPayload
          }

          const comparisonMode: SemanticMapMode =
            semanticMapMode === 'semantic' ? 'code_structure' : 'semantic'
          const [activeMapResult, comparisonMapResult] = await Promise.allSettled([
            loadMap(semanticMapMode),
            loadMap(comparisonMode),
          ])

          if (activeMapResult.status !== 'fulfilled') {
            throw activeMapResult.reason
          }

          setSemanticMap(activeMapResult.value)
          if (comparisonMapResult.status === 'fulfilled') {
            setSemanticMapComparison(comparisonMapResult.value)
          } else {
            setSemanticMapComparison(null)
          }

          const graphQuery = new URLSearchParams({
            scenario_id: scenarioId,
            community_mode: 'color',
            limit: String(limit),
            page: String(graphPaging.page),
          })
          if (graphFilters.includeKinds.length > 0) {
            graphQuery.set('include_kinds', graphFilters.includeKinds.join(','))
          }
          if (graphFilters.excludeKinds.length > 0) {
            graphQuery.set('exclude_kinds', graphFilters.excludeKinds.join(','))
          }
          if (graphFilters.edgeKinds.length > 0) {
            graphQuery.set('edge_kinds', graphFilters.edgeKinds.join(','))
          }

          const graphResponse = await fetch(
            `/api/twin/collections/${collectionId}/views/graphrag?${graphQuery.toString()}`,
            {
              credentials: 'include',
              signal: controller.signal,
            },
          )
          if (!graphResponse.ok) {
            throw new Error(`Could not load semantic map graph (${graphResponse.status})`)
          }
          const graphPayload: GraphRagPayload = await graphResponse.json()
          setGraph({
            ...(graphPayload.graph || DEFAULT_GRAPH),
            projection: graphPayload.projection,
            entity_level: graphPayload.entity_level,
          })

          setViewState('semantic_map', 'ready')
          markUpdated('semantic_map')
          return
        }

        if (view === 'rebuild_readiness') {
          const response = await fetch(
            `/api/twin/collections/${collectionId}/views/rebuild-readiness?scenario_id=${encodeURIComponent(scenarioId)}`,
            {
              credentials: 'include',
              signal: controller.signal,
            },
          )
          if (!response.ok) {
            throw new Error(`Could not load rebuild-readiness (${response.status})`)
          }
          const payload: RebuildReadinessPayload = await response.json()
          setRebuildReadiness(payload)
          setViewState('rebuild_readiness', 'ready')
          markUpdated('rebuild_readiness')
          return
        }

        if (view === 'architecture') {
          const section = architectureSection.trim()
          const normalizedContainer = portsContainer.trim()
          const baselineScenarioId = driftBaselineScenarioId.trim()

          const arc42Query = new URLSearchParams({ scenario_id: scenarioId })
          if (section) arc42Query.set('section', section)

          const portsQuery = new URLSearchParams({ scenario_id: scenarioId })
          if (portsDirection !== 'all') {
            portsQuery.set('direction', portsDirection)
          }
          if (normalizedContainer) {
            portsQuery.set('container', normalizedContainer)
          }

          const driftQuery = new URLSearchParams({ scenario_id: scenarioId })
          if (baselineScenarioId) {
            driftQuery.set('baseline_scenario_id', baselineScenarioId)
          }

          const loadPayload = async <T>(url: string, label: string): Promise<T> => {
            const response = await fetch(url, {
              credentials: 'include',
              signal: controller.signal,
            })
            if (!response.ok) {
              throw new Error(`Could not load ${label} (${response.status})`)
            }
            return (await response.json()) as T
          }

          const [arc42Result, portsResult, driftResult, ermResult] = await Promise.allSettled([
            loadPayload<Arc42ViewPayload>(
              `/api/twin/collections/${collectionId}/views/arc42?${arc42Query.toString()}`,
              'arc42 view',
            ),
            loadPayload<PortsAdaptersPayload>(
              `/api/twin/collections/${collectionId}/views/ports-adapters?${portsQuery.toString()}`,
              'ports/adapters view',
            ),
            loadPayload<Arc42DriftPayload>(
              `/api/twin/collections/${collectionId}/views/arc42/drift?${driftQuery.toString()}`,
              'arc42 drift view',
            ),
            loadPayload<ErmViewPayload>(
              `/api/twin/collections/${collectionId}/views/erm?scenario_id=${encodeURIComponent(scenarioId)}`,
              'erm view',
            ),
          ])

          let successCount = 0
          const nextErrors = { arc42: '', ports: '', drift: '', erm: '' }
          if (arc42Result.status === 'fulfilled') {
            setArc42(arc42Result.value)
            successCount += 1
          } else {
            nextErrors.arc42 =
              arc42Result.reason instanceof Error ? arc42Result.reason.message : 'Could not load arc42 view'
          }

          if (portsResult.status === 'fulfilled') {
            setPortsAdapters(portsResult.value)
            successCount += 1
          } else {
            nextErrors.ports =
              portsResult.reason instanceof Error
                ? portsResult.reason.message
                : 'Could not load ports/adapters view'
          }

          if (driftResult.status === 'fulfilled') {
            setArc42Drift(driftResult.value)
            successCount += 1
          } else {
            nextErrors.drift =
              driftResult.reason instanceof Error
                ? driftResult.reason.message
                : 'Could not load drift view'
          }

          if (ermResult.status === 'fulfilled') {
            setErm(ermResult.value)
            successCount += 1
          } else {
            nextErrors.erm =
              ermResult.reason instanceof Error
                ? ermResult.reason.message
                : 'Could not load erm view'
          }

          setArchitecturePanelErrors(nextErrors)
          const allErrors = [nextErrors.arc42, nextErrors.ports, nextErrors.drift, nextErrors.erm].filter(Boolean)
          if (allErrors.length > 0) {
            setViewError('architecture', allErrors.join(' • '))
          } else {
            setViewError('architecture', '')
          }

          if (successCount > 0) {
            setViewState('architecture', 'ready')
            markUpdated('architecture')
          } else {
            setViewState('architecture', 'error')
          }
          return
        }

        if (view === 'graphrag') {
          const query = new URLSearchParams({
            scenario_id: scenarioId,
            limit: String(graphPaging.limit > 0 ? graphPaging.limit : topologyLimit),
            page: String(graphPaging.page),
            community_mode: graphRagCommunityMode,
          })
          const normalizedCommunityId = graphRagCommunityId.trim()
          if (normalizedCommunityId) {
            query.set('community_id', normalizedCommunityId)
          }
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
            throw new Error(`Could not load graphrag view (${response.status})`)
          }
          const payload: GraphRagPayload = await response.json()
          setGraph({
            ...(payload.graph || DEFAULT_GRAPH),
            projection: payload.projection,
            entity_level: payload.entity_level,
          })
          setGraphRagStatus(payload.status?.status || 'ready')
          setGraphRagReason(payload.status?.reason || 'ok')

          setGraphRagCommunitiesState('loading')
          setGraphRagCommunitiesError('')
          try {
            const communitiesResponse = await fetch(
              `/api/twin/collections/${collectionId}/views/graphrag/communities?scenario_id=${encodeURIComponent(scenarioId)}&limit=500`,
              {
                credentials: 'include',
                signal: controller.signal,
              },
            )
            if (!communitiesResponse.ok) {
              throw new Error(`Could not load communities (${communitiesResponse.status})`)
            }
            const communitiesPayload: GraphRagCommunitiesPayload = await communitiesResponse.json()
            setGraphRagCommunities(communitiesPayload.items || [])
            setGraphRagCommunitiesState((communitiesPayload.items || []).length > 0 ? 'ready' : 'empty')
          } catch (error) {
            if (!controller.signal.aborted) {
              setGraphRagCommunitiesState('error')
              setGraphRagCommunitiesError(
                error instanceof Error ? error.message : 'Could not load communities',
              )
            }
          }

          setGraphRagProcessesState('loading')
          setGraphRagProcessesError('')
          try {
            const processesResponse = await fetch(
              `/api/twin/collections/${collectionId}/views/graphrag/processes?scenario_id=${encodeURIComponent(scenarioId)}`,
              {
                credentials: 'include',
                signal: controller.signal,
              },
            )
            if (!processesResponse.ok) {
              throw new Error(`Could not load processes (${processesResponse.status})`)
            }
            const processesPayload = await processesResponse.json()
            setGraphRagProcesses(processesPayload.items || [])
            setGraphRagProcessesState((processesPayload.items || []).length > 0 ? 'ready' : 'empty')
          } catch (error) {
            if (!controller.signal.aborted) {
              setGraphRagProcessesState('error')
              setGraphRagProcessesError(
                error instanceof Error ? error.message : 'Could not load processes',
              )
            }
          }
          setViewState('graphrag', 'ready')
          markUpdated('graphrag')
          return
        }

        if (view === 'c4_diff') {
          const query = new URLSearchParams({
            scenario_id: scenarioId,
            compare_with_base: 'true',
            c4_view: c4View,
            max_nodes: String(Math.max(10, c4MaxNodes || 120)),
          })
          const normalizedScope = c4Scope.trim()
          if (normalizedScope) {
            query.set('c4_scope', normalizedScope)
          }
          const response = await fetch(
            `/api/twin/collections/${collectionId}/views/mermaid?${query.toString()}`,
            {
              credentials: 'include',
              signal: controller.signal,
            },
          )
          if (!response.ok) {
            throw new Error(`Could not load C4 view (${response.status})`)
          }
          const payload: MermaidPayload = await response.json()
          setMermaid(payload)
          setViewState('c4_diff', 'ready')
          markUpdated('c4_diff')
          return
        }

        if (view === 'city') {
          const cityProjectionValue: CityProjection = cityProjection
          const entityLevel =
            cityProjectionValue === 'architecture'
              ? cityEntityLevel
              : 'file'

          const exportResponse = await fetch(`/api/twin/scenarios/${scenarioId}/exports`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            signal: controller.signal,
            body: JSON.stringify({
              format: 'cc_json',
              projection: cityProjectionValue,
              entity_level: entityLevel,
            }),
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
          const params = new URLSearchParams({
            file: rawPath,
            area: 'loc',
            height: 'coupling',
            color: 'complexity',
            mode: 'Single',
          })
          setCityEmbedUrl(`/codecharta/index.html?${params.toString()}`)
          setViewState('city', 'ready')
          markUpdated('city')
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
    cityProjection,
    cityEntityLevel,
    refreshNonce,
    markUpdated,
    setViewError,
    setViewState,
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
        const projection =
          view === 'topology'
            ? 'architecture'
            : deepDiveMode === 'file_dependency'
              ? 'code_file'
              : 'code_symbol'
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
      const exportResponse = await fetch(`/api/twin/scenarios/${scenarioId}/exports`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          format: exportFormat,
          projection: exportProjection,
          entity_level:
            exportProjection === 'architecture'
              ? topologyEntityLevel(selection.layer)
              : exportProjection === 'code_file'
                ? 'file'
                : 'symbol',
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
    semanticMap,
    semanticMapComparison,
    testMatrix,
    userFlows,
    rebuildReadiness,
    traceGraphRagPath,
    loadGraphRagProcessDetail,
    generateExport,
    refreshActiveView,
  }
}
