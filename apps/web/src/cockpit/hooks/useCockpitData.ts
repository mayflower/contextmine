import { useCallback, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api, apiData, apiErrorMessage } from '../../api/client'
import { buildCityEmbedUrl } from '../utils/cityEmbed'
import type {
  Arc42DriftPayload,
  Arc42ViewPayload,
  C4ViewMode,
  CityEntityLevel,
  CityPayload,
  CityProjection,
  CockpitLayer,
  CockpitLoadState,
  CockpitSelection,
  CockpitView,
  DeepDiveMode,
  ErmViewPayload,
  ExportFormat,
  ExportProjection,
  FitnessFunctionsPayload,
  GraphFilters,
  GraphPagingState,
  GraphRagCommunity,
  GraphRagCommunityMode,
  GraphRagEvidenceItem,
  GraphRagProcessSummary,
  GraphRagStatusReason,
  GraphViewPayload,
  InvestmentUtilizationPayload,
  KnowledgeIslandsPayload,
  MermaidPayload,
  PortsAdaptersPayload,
  PortsDirection,
  RebuildReadinessPayload,
  SemanticMapMode,
  SemanticMapPayload,
  SemanticMapThresholds,
  TemporalCouplingPayload,
  TestMatrixPayload,
  TwinGraphResponse,
  UIMapPayload,
  UserFlowsPayload,
} from '../types'

type DataStates = Record<CockpitView, CockpitLoadState>
type DataErrors = Record<CockpitView, string>

const DEFAULT_GRAPH: TwinGraphResponse = {
  nodes: [],
  edges: [],
  page: 0,
  limit: 0,
  total_nodes: 0,
  warnings: [],
}

const EMPTY_STATES: DataStates = {
  overview: 'idle', topology: 'idle', deep_dive: 'idle', c4_diff: 'idle', architecture: 'idle',
  city: 'idle', evolution: 'idle', graphrag: 'idle', ui_map: 'idle', semantic_map: 'idle',
  test_matrix: 'idle', user_flows: 'idle', rebuild_readiness: 'idle', exports: 'idle',
}

const EMPTY_ERRORS: DataErrors = {
  overview: '', topology: '', deep_dive: '', c4_diff: '', architecture: '', city: '', evolution: '',
  graphrag: '', ui_map: '', semantic_map: '', test_matrix: '', user_flows: '', rebuild_readiness: '', exports: '',
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

interface ViewData {
  city?: CityPayload
  graph?: TwinGraphResponse
  mermaid?: MermaidPayload
  arc42?: Arc42ViewPayload
  portsAdapters?: PortsAdaptersPayload
  arc42Drift?: Arc42DriftPayload
  erm?: ErmViewPayload
  architecturePanelErrors?: { arc42: string; ports: string; drift: string; erm: string }
  cityEmbedUrl?: string
  investmentUtilization?: InvestmentUtilizationPayload
  knowledgeIslands?: KnowledgeIslandsPayload
  temporalCoupling?: TemporalCouplingPayload
  fitnessFunctions?: FitnessFunctionsPayload
  evolutionPanelErrors?: { investment: string; knowledge: string; coupling: string; fitness: string }
  graphRagStatus?: 'ready' | 'unavailable'
  graphRagReason?: GraphRagStatusReason
  graphRagCommunities?: GraphRagCommunity[]
  graphRagCommunitiesError?: string
  graphRagProcesses?: GraphRagProcessSummary[]
  graphRagProcessesError?: string
  uiMapSummary?: UIMapPayload['summary']
  uiMapGraph?: TwinGraphResponse
  userFlowsGraph?: TwinGraphResponse
  semanticMap?: SemanticMapPayload
  semanticMapComparison?: SemanticMapPayload
  testMatrix?: TestMatrixPayload
  userFlows?: UserFlowsPayload
  rebuildReadiness?: RebuildReadinessPayload
}

function topologyEntityLevel(layer: CockpitLayer): 'domain' | 'container' | 'component' {
  if (layer === 'portfolio_system') return 'domain'
  if (layer === 'component_interface') return 'component'
  return 'container'
}

function errorText(result: PromiseSettledResult<unknown>, fallback: string): string {
  return result.status === 'rejected' ? apiErrorMessage(result.reason, fallback) : ''
}

function graphFromView(payload: GraphViewPayload): TwinGraphResponse {
  const graph = payload.graph || DEFAULT_GRAPH
  return {
    ...graph,
    projection: payload.projection ?? graph.projection,
    entity_level: payload.entity_level ?? graph.entity_level,
    grouping_strategy: payload.grouping_strategy ?? graph.grouping_strategy,
    excluded_kinds: payload.excluded_kinds ?? graph.excluded_kinds,
    warnings: payload.warnings ?? graph.warnings ?? [],
    provenance: payload.provenance ?? graph.provenance,
  }
}

export function useCockpitData(args: UseCockpitDataArgs) {
  const {
    selection, behaviorGraphMode, topologyLimit, deepDiveLimit, deepDiveMode, c4View, c4Scope,
    c4MaxNodes, architectureSection, portsDirection, portsContainer, driftBaselineScenarioId,
    graphFilters, graphPaging, graphRagCommunityMode, graphRagCommunityId, semanticMapMode,
    semanticMapThresholdsByMode, selectedNodeId, onScenarioAutoSelect, onViewError,
  } = args
  const queryClient = useQueryClient()
  const [cityProjection, setCityProjection] = useState<CityProjection>('architecture')
  const [cityEntityLevel, setCityEntityLevel] = useState<CityEntityLevel>('container')
  const [exportFormat, setExportFormat] = useState<ExportFormat>('cc_json')
  const [exportProjection, setExportProjection] = useState<ExportProjection>('architecture')
  const [exportContent, setExportContent] = useState('')

  const scenariosQuery = useQuery({
    queryKey: ['twin', 'scenarios', selection.collectionId],
    enabled: Boolean(selection.collectionId),
    queryFn: ({ signal }) => apiData(api.GET('/api/twin/scenarios', {
      params: { query: { collection_id: selection.collectionId } }, signal,
    })),
  })
  const scenarios = useMemo(() => scenariosQuery.data?.scenarios ?? [], [scenariosQuery.data])

  useEffect(() => {
    if (scenarios.length === 0 || scenarios.some((scenario) => scenario.id === selection.scenarioId)) return
    onScenarioAutoSelect(scenarios.find((scenario) => scenario.is_as_is)?.id ?? scenarios[0].id)
  }, [onScenarioAutoSelect, scenarios, selection.scenarioId])

  const activeQuery = useQuery({
    queryKey: ['cockpit', selection.collectionId, selection.scenarioId, selection.view, {
      layer: selection.layer, topologyLimit, deepDiveLimit, deepDiveMode, c4View, c4Scope, c4MaxNodes,
      architectureSection, portsDirection, portsContainer, driftBaselineScenarioId, graphFilters,
      graphPaging, graphRagCommunityMode, graphRagCommunityId, semanticMapMode,
      semanticThresholds: semanticMapThresholdsByMode, cityProjection, cityEntityLevel,
    }],
    enabled: Boolean(selection.collectionId && selection.scenarioId && selection.view !== 'exports'),
    queryFn: async ({ signal }): Promise<ViewData> => {
      const collectionId = selection.collectionId
      const scenarioId = selection.scenarioId
      const path = { collection_id: collectionId }
      const page = graphPaging.page
      const limit = graphPaging.limit > 0 ? graphPaging.limit : topologyLimit

      if (selection.view === 'overview') {
        const city = await apiData(api.GET('/api/twin/collections/{collection_id}/views/city', {
          params: { path, query: { scenario_id: scenarioId, hotspots_limit: 60 } }, signal,
        }))
        return { city }
      }

      if (selection.view === 'topology' || selection.view === 'deep_dive') {
        const view = selection.view
        const viewLimit = graphPaging.limit > 0 ? graphPaging.limit : view === 'topology' ? topologyLimit : deepDiveLimit
        const common = {
          scenario_id: scenarioId, layer: selection.layer, limit: viewLimit, page,
          include_kinds: graphFilters.includeKinds.join(',') || null,
          exclude_kinds: graphFilters.excludeKinds.join(',') || null,
        }
        let payload: GraphViewPayload
        if (view === 'topology') {
          payload = await apiData(api.GET('/api/twin/collections/{collection_id}/views/topology', {
            params: { path, query: { ...common, projection: 'architecture', entity_level: topologyEntityLevel(selection.layer) } }, signal,
          }))
        } else {
          const projection = deepDiveMode === 'file_dependency' ? 'code_file' : 'code_symbol'
          const entityLevel = deepDiveMode === 'file_dependency' ? 'file' : 'symbol'
          payload = await apiData(api.GET('/api/twin/collections/{collection_id}/views/deep-dive', {
            params: { path, query: { ...common, projection, entity_level: entityLevel, mode: deepDiveMode } }, signal,
          }))
        }
        return { graph: graphFromView(payload) }
      }

      if (selection.view === 'ui_map') {
        const query = { scenario_id: scenarioId, page, limit }
        const [ui, flows] = await Promise.allSettled([
          apiData(api.GET('/api/twin/collections/{collection_id}/views/ui-map', { params: { path, query }, signal })),
          apiData(api.GET('/api/twin/collections/{collection_id}/views/user-flows', { params: { path, query }, signal })),
        ])
        if (ui.status === 'rejected' && flows.status === 'rejected') throw ui.reason
        const uiPayload = ui.status === 'fulfilled' ? ui.value : undefined
        const flowsPayload = flows.status === 'fulfilled' ? flows.value : undefined
        const uiGraph = uiPayload ? { ...uiPayload.graph, projection: 'code_symbol' as const, entity_level: uiPayload.entity_level } : DEFAULT_GRAPH
        const flowsGraph = flowsPayload ? { ...flowsPayload.graph, projection: 'code_symbol' as const, entity_level: flowsPayload.entity_level } : DEFAULT_GRAPH
        return {
          uiMapSummary: uiPayload?.summary, uiMapGraph: uiGraph, userFlows: flowsPayload,
          userFlowsGraph: flowsGraph,
          graph: behaviorGraphMode === 'user_flows'
            ? (flowsGraph.total_nodes > 0 ? flowsGraph : uiGraph)
            : (uiGraph.total_nodes > 0 ? uiGraph : flowsGraph),
        }
      }

      if (selection.view === 'test_matrix') {
        const testMatrix = await apiData(api.GET('/api/twin/collections/{collection_id}/views/test-matrix', {
          params: { path, query: { scenario_id: scenarioId, page, limit } }, signal,
        }))
        return { testMatrix, graph: { ...testMatrix.graph, projection: 'code_symbol', entity_level: testMatrix.entity_level } }
      }

      if (selection.view === 'semantic_map') {
        const buildQuery = (mode: SemanticMapMode) => {
          const thresholds = semanticMapThresholdsByMode[mode]
          return {
            scenario_id: scenarioId, map_mode: mode, page, limit,
            include_kinds: graphFilters.includeKinds.join(',') || null,
            exclude_kinds: graphFilters.excludeKinds.join(',') || null,
            edge_kinds: graphFilters.edgeKinds.join(',') || null,
            ...thresholds,
          }
        }
        const comparisonMode = semanticMapMode === 'semantic' ? 'code_structure' : 'semantic'
        const [current, comparison, context] = await Promise.allSettled([
          apiData(api.GET('/api/twin/collections/{collection_id}/views/semantic-map', { params: { path, query: buildQuery(semanticMapMode) }, signal })),
          apiData(api.GET('/api/twin/collections/{collection_id}/views/semantic-map', { params: { path, query: buildQuery(comparisonMode) }, signal })),
          apiData(api.GET('/api/twin/collections/{collection_id}/views/graphrag', {
            params: { path, query: { scenario_id: scenarioId, community_mode: 'color', page, limit,
              include_kinds: graphFilters.includeKinds.join(',') || null,
              exclude_kinds: graphFilters.excludeKinds.join(',') || null,
              edge_kinds: graphFilters.edgeKinds.join(',') || null } }, signal,
          })),
        ])
        if (current.status === 'rejected') throw current.reason
        return {
          semanticMap: current.value,
          semanticMapComparison: comparison.status === 'fulfilled' ? comparison.value : undefined,
          graph: context.status === 'fulfilled' ? { ...context.value.graph, projection: context.value.projection, entity_level: context.value.entity_level } : DEFAULT_GRAPH,
        }
      }

      if (selection.view === 'rebuild_readiness') {
        return { rebuildReadiness: await apiData(api.GET('/api/twin/collections/{collection_id}/views/rebuild-readiness', {
          params: { path, query: { scenario_id: scenarioId } }, signal,
        })) }
      }

      if (selection.view === 'architecture') {
        const section = architectureSection.trim() || null
        const direction = portsDirection === 'all' ? null : portsDirection
        const container = portsContainer.trim() || null
        const baseline = driftBaselineScenarioId.trim() || null
        const results = await Promise.allSettled([
          apiData(api.GET('/api/twin/collections/{collection_id}/views/arc42', { params: { path, query: { scenario_id: scenarioId, section } }, signal })),
          apiData(api.GET('/api/twin/collections/{collection_id}/views/ports-adapters', { params: { path, query: { scenario_id: scenarioId, direction, container } }, signal })),
          apiData(api.GET('/api/twin/collections/{collection_id}/views/arc42/drift', { params: { path, query: { scenario_id: scenarioId, baseline_scenario_id: baseline } }, signal })),
          apiData(api.GET('/api/twin/collections/{collection_id}/views/erm', { params: { path, query: { scenario_id: scenarioId } }, signal })),
        ])
        if (results.every((result) => result.status === 'rejected')) throw (results[0] as PromiseRejectedResult).reason
        return {
          arc42: results[0].status === 'fulfilled' ? results[0].value : undefined,
          portsAdapters: results[1].status === 'fulfilled' ? results[1].value : undefined,
          arc42Drift: results[2].status === 'fulfilled' ? results[2].value : undefined,
          erm: results[3].status === 'fulfilled' ? results[3].value : undefined,
          architecturePanelErrors: {
            arc42: errorText(results[0], 'Could not load arc42 view'), ports: errorText(results[1], 'Could not load ports/adapters view'),
            drift: errorText(results[2], 'Could not load drift view'), erm: errorText(results[3], 'Could not load ERM view'),
          },
        }
      }

      if (selection.view === 'evolution') {
        const results = await Promise.allSettled([
          apiData(api.GET('/api/twin/collections/{collection_id}/views/evolution/investment-utilization', { params: { path, query: { scenario_id: scenarioId, entity_level: 'container', window_days: 365 } }, signal })),
          apiData(api.GET('/api/twin/collections/{collection_id}/views/evolution/knowledge-islands', { params: { path, query: { scenario_id: scenarioId, entity_level: 'container', window_days: 365, ownership_threshold: 0.7 } }, signal })),
          apiData(api.GET('/api/twin/collections/{collection_id}/views/evolution/temporal-coupling', { params: { path, query: { scenario_id: scenarioId, entity_level: 'component', window_days: 365, min_jaccard: 0.2, max_edges: 300 } }, signal })),
          apiData(api.GET('/api/twin/collections/{collection_id}/views/evolution/fitness-functions', { params: { path, query: { scenario_id: scenarioId, window_days: 365, include_resolved: false } }, signal })),
        ])
        if (results.every((result) => result.status === 'rejected')) throw (results[0] as PromiseRejectedResult).reason
        return {
          investmentUtilization: results[0].status === 'fulfilled' ? results[0].value : undefined,
          knowledgeIslands: results[1].status === 'fulfilled' ? results[1].value : undefined,
          temporalCoupling: results[2].status === 'fulfilled' ? results[2].value : undefined,
          fitnessFunctions: results[3].status === 'fulfilled' ? results[3].value : undefined,
          evolutionPanelErrors: {
            investment: errorText(results[0], 'Could not load investment/utilization'), knowledge: errorText(results[1], 'Could not load knowledge islands'),
            coupling: errorText(results[2], 'Could not load temporal coupling'), fitness: errorText(results[3], 'Could not load fitness functions'),
          },
        }
      }

      if (selection.view === 'graphrag') {
        const query = { scenario_id: scenarioId, page, limit, community_mode: graphRagCommunityMode,
          community_id: graphRagCommunityId.trim() || null, include_kinds: graphFilters.includeKinds.join(',') || null,
          exclude_kinds: graphFilters.excludeKinds.join(',') || null, edge_kinds: graphFilters.edgeKinds.join(',') || null }
        const [main, communities, processes] = await Promise.allSettled([
          apiData(api.GET('/api/twin/collections/{collection_id}/views/graphrag', { params: { path, query }, signal })),
          apiData(api.GET('/api/twin/collections/{collection_id}/views/graphrag/communities', { params: { path, query: { scenario_id: scenarioId, limit: 500 } }, signal })),
          apiData(api.GET('/api/twin/collections/{collection_id}/views/graphrag/processes', { params: { path, query: { scenario_id: scenarioId } }, signal })),
        ])
        if (main.status === 'rejected') throw main.reason
        return {
          graph: { ...main.value.graph, projection: main.value.projection, entity_level: main.value.entity_level },
          graphRagStatus: main.value.status?.status ?? 'ready', graphRagReason: main.value.status?.reason ?? 'ok',
          graphRagCommunities: communities.status === 'fulfilled' ? communities.value.items : [],
          graphRagCommunitiesError: errorText(communities, 'Could not load communities'),
          graphRagProcesses: processes.status === 'fulfilled' ? processes.value.items : [],
          graphRagProcessesError: errorText(processes, 'Could not load processes'),
        }
      }

      if (selection.view === 'c4_diff') {
        return { mermaid: await apiData(api.GET('/api/twin/collections/{collection_id}/views/mermaid', {
          params: { path, query: { scenario_id: scenarioId, compare_with_base: true, c4_view: c4View,
            c4_scope: c4Scope.trim() || null, max_nodes: Math.max(10, c4MaxNodes || 120) } }, signal,
        })) }
      }

      if (selection.view === 'city') {
        const entityLevel = cityProjection === 'architecture' ? cityEntityLevel : 'file'
        const created = await apiData(api.POST('/api/twin/scenarios/{scenario_id}/exports', {
          params: { path: { scenario_id: scenarioId } }, body: { format: 'cc_json', projection: cityProjection, entity_level: entityLevel }, signal,
        }))
        const exportId = created.id ?? created.exports?.[0]?.id
        if (!exportId) throw new Error('Missing export id from city export response')
        const rawPath = `/api/twin/scenarios/${scenarioId}/exports/${exportId}/raw`
        return { cityEmbedUrl: buildCityEmbedUrl(rawPath) }
      }

      return {}
    },
  })

  const evidenceQuery = useQuery({
    queryKey: ['cockpit', selection.collectionId, selection.scenarioId, 'graphrag-evidence', selectedNodeId],
    enabled: selection.view === 'graphrag' && Boolean(selection.collectionId && selection.scenarioId && selectedNodeId),
    queryFn: ({ signal }) => apiData(api.GET('/api/twin/collections/{collection_id}/views/graphrag/evidence', {
      params: { path: { collection_id: selection.collectionId }, query: { scenario_id: selection.scenarioId, node_id: selectedNodeId, limit: 50 } }, signal,
    })),
  })

  const neighborhoodQuery = useQuery({
    queryKey: ['cockpit', selection.scenarioId, 'neighborhood', selectedNodeId, selection.view, deepDiveMode],
    enabled: Boolean(selection.scenarioId && selectedNodeId && (selection.view === 'topology' || selection.view === 'deep_dive')),
    queryFn: ({ signal }) => {
      const projection = selection.view === 'topology' ? 'architecture' : deepDiveMode === 'file_dependency' ? 'code_file' : 'code_symbol'
      return apiData(api.GET('/api/twin/scenarios/{scenario_id}/graph/neighborhood', {
        params: { path: { scenario_id: selection.scenarioId }, query: { node_id: selectedNodeId, projection, hops: 1, limit: 200 } }, signal,
      }))
    },
  })

  const reindex = useMutation({
    mutationFn: async () => apiData(api.POST('/api/twin/collections/{collection_id}/refresh', {
      params: { path: { collection_id: selection.collectionId } }, body: { force: true },
    })),
    onSuccess: async () => queryClient.invalidateQueries({ queryKey: ['cockpit', selection.collectionId] }),
  })
  const regenerate = useMutation({
    mutationFn: () => apiData(api.GET('/api/twin/collections/{collection_id}/views/arc42', {
      params: { path: { collection_id: selection.collectionId }, query: { scenario_id: selection.scenarioId, regenerate: true, section: architectureSection.trim() || null } },
    })),
    onSuccess: async () => queryClient.invalidateQueries({ queryKey: ['cockpit', selection.collectionId, selection.scenarioId, 'architecture'] }),
  })
  const exportMutation = useMutation({
    mutationFn: async () => {
      const entityLevel = exportProjection === 'architecture' ? topologyEntityLevel(selection.layer) : exportProjection === 'code_file' ? 'file' : 'symbol'
      const created = await apiData(api.POST('/api/twin/scenarios/{scenario_id}/exports', {
        params: { path: { scenario_id: selection.scenarioId } }, body: { format: exportFormat, projection: exportProjection, entity_level: entityLevel },
      }))
      const exportId = created.id ?? created.exports?.[0]?.id
      if (!exportId) throw new Error('Missing export id from API response')
      const artifact = await apiData(api.GET('/api/twin/scenarios/{scenario_id}/exports/{export_id}', {
        params: { path: { scenario_id: selection.scenarioId, export_id: exportId } },
      }))
      return { content: artifact.content ?? '', name: artifact.name ?? `${exportFormat}.txt` }
    },
    onSuccess: (artifact) => setExportContent(artifact.content),
  })
  const pathMutation = useMutation({
    mutationFn: ({ from, to, maxHops }: { from: string; to: string; maxHops: number }) => apiData(api.GET('/api/twin/collections/{collection_id}/views/graphrag/path', {
      params: { path: { collection_id: selection.collectionId }, query: { scenario_id: selection.scenarioId, from_node_id: from, to_node_id: to, max_hops: Math.max(1, Math.min(20, maxHops)) } },
    })),
  })
  const processMutation = useMutation({
    mutationFn: (processId: string) => apiData(api.GET('/api/twin/collections/{collection_id}/views/graphrag/processes/{process_id}', {
      params: { path: { collection_id: selection.collectionId, process_id: processId }, query: { scenario_id: selection.scenarioId } },
    })),
  })

  const refreshActiveView = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ['cockpit', selection.collectionId, selection.scenarioId] })
  }, [queryClient, selection.collectionId, selection.scenarioId])

  const triggerCollectionReindex = useCallback(async () => {
    if (!selection.collectionId) return false
    try { await reindex.mutateAsync(); return true } catch { return false }
  }, [reindex, selection.collectionId])
  const regenerateArc42 = useCallback(async () => {
    if (!selection.collectionId || !selection.scenarioId) return false
    try { await regenerate.mutateAsync(); return true } catch { return false }
  }, [regenerate, selection.collectionId, selection.scenarioId])
  const generateExport = useCallback(async () => {
    if (!selection.scenarioId) return null
    try { return await exportMutation.mutateAsync() } catch { return null }
  }, [exportMutation, selection.scenarioId])
  const traceGraphRagPath = useCallback(async (fromNodeId: string, toNodeId: string, maxHops: number) => {
    const from = fromNodeId.trim(); const to = toNodeId.trim()
    if (!selection.collectionId || !selection.scenarioId || !from || !to) return null
    try { return await pathMutation.mutateAsync({ from, to, maxHops }) } catch { return null }
  }, [pathMutation, selection.collectionId, selection.scenarioId])
  const loadGraphRagProcessDetail = useCallback(async (processId: string) => {
    if (!selection.collectionId || !selection.scenarioId || !processId) return null
    try { return await processMutation.mutateAsync(processId) } catch { return null }
  }, [processMutation, selection.collectionId, selection.scenarioId])

  const activeState: CockpitLoadState = selection.view === 'exports'
    ? (selection.scenarioId ? (exportMutation.isPending ? 'loading' : exportMutation.isError ? 'error' : 'ready') : 'empty')
    : !selection.collectionId || !selection.scenarioId ? 'empty'
      : activeQuery.isPending ? 'loading' : activeQuery.isError ? 'error' : 'ready'
  const activeError = activeQuery.isError ? apiErrorMessage(activeQuery.error, 'Unexpected Cockpit request error')
    : exportMutation.isError ? apiErrorMessage(exportMutation.error, 'Export generation failed') : ''
  const states = useMemo(() => ({ ...EMPTY_STATES, [selection.view]: activeState }), [activeState, selection.view])
  const errors = useMemo(() => ({ ...EMPTY_ERRORS, [selection.view]: activeError }), [activeError, selection.view])

  useEffect(() => {
    if (activeError) onViewError?.(selection.view, activeError)
  }, [activeError, onViewError, selection.view])

  const data = activeQuery.data ?? {}
  const graph = data.graph ?? DEFAULT_GRAPH
  const evidence = evidenceQuery.data
  const architectureActions = {
    reindexState: (reindex.isPending ? 'loading' : reindex.isError ? 'error' : reindex.isSuccess ? 'ready' : 'idle') as CockpitLoadState,
    reindexMessage: reindex.isPending ? 'Reindexing started...' : reindex.isError ? apiErrorMessage(reindex.error, 'Could not start reindexing.')
      : reindex.data ? reindex.data.created ? `Reindexing queued for ${reindex.data.created} source(s).` : reindex.data.skipped ? `No new source revisions queued (${reindex.data.skipped} unchanged).` : 'Reindexing request accepted.' : '',
    regenerateState: (regenerate.isPending ? 'loading' : regenerate.isError ? 'error' : regenerate.isSuccess ? 'ready' : 'idle') as CockpitLoadState,
    regenerateMessage: regenerate.isPending ? 'Generating arc42...' : regenerate.isError ? apiErrorMessage(regenerate.error, 'Could not regenerate arc42.') : regenerate.isSuccess ? 'arc42 regenerated successfully.' : '',
  }

  return {
    scenarios,
    scenariosState: (!selection.collectionId ? 'empty' : scenariosQuery.isPending ? 'loading' : scenariosQuery.isError ? 'error' : scenarios.length ? 'ready' : 'empty') as CockpitLoadState,
    city: data.city ?? null,
    graph,
    mermaid: data.mermaid ?? null,
    arc42: data.arc42 ?? null,
    portsAdapters: data.portsAdapters ?? null,
    arc42Drift: data.arc42Drift ?? null,
    erm: data.erm ?? null,
    architecturePanelErrors: data.architecturePanelErrors ?? { arc42: '', ports: '', drift: '', erm: '' },
    architectureActions,
    states,
    errors,
    activeState,
    activeError,
    activeUpdatedAt: activeQuery.dataUpdatedAt ? new Date(activeQuery.dataUpdatedAt).toISOString() : null,
    cityProjection, setCityProjection, cityEntityLevel, setCityEntityLevel,
    cityEmbedUrl: data.cityEmbedUrl ?? '',
    investmentUtilization: data.investmentUtilization ?? null,
    knowledgeIslands: data.knowledgeIslands ?? null,
    temporalCoupling: data.temporalCoupling ?? null,
    fitnessFunctions: data.fitnessFunctions ?? null,
    evolutionPanelErrors: data.evolutionPanelErrors ?? { investment: '', knowledge: '', coupling: '', fitness: '' },
    exportFormat, setExportFormat, exportProjection, setExportProjection, exportContent, setExportContent,
    neighborhood: neighborhoodQuery.data?.graph ?? DEFAULT_GRAPH,
    neighborhoodState: (!neighborhoodQuery.isEnabled ? 'idle' : neighborhoodQuery.isPending ? 'loading' : neighborhoodQuery.isError ? 'error' : 'ready') as CockpitLoadState,
    neighborhoodError: neighborhoodQuery.isError ? apiErrorMessage(neighborhoodQuery.error, 'Could not load neighborhood') : '',
    graphRagStatus: data.graphRagStatus ?? 'ready',
    graphRagReason: data.graphRagReason ?? 'ok',
    graphRagEvidenceItems: evidence?.items ?? [] as GraphRagEvidenceItem[],
    graphRagEvidenceTotal: evidence?.total ?? 0,
    graphRagEvidenceNodeName: evidence?.node_name ?? '',
    graphRagEvidenceState: (!evidenceQuery.isEnabled ? (selection.view === 'graphrag' ? 'empty' : 'idle') : evidenceQuery.isPending ? 'loading' : evidenceQuery.isError ? 'error' : evidence?.items.length ? 'ready' : 'empty') as CockpitLoadState,
    graphRagEvidenceError: evidenceQuery.isError ? apiErrorMessage(evidenceQuery.error, 'Could not load node evidence') : '',
    graphRagCommunities: data.graphRagCommunities ?? [],
    graphRagCommunitiesState: (selection.view !== 'graphrag' ? 'idle' : activeQuery.isPending ? 'loading' : data.graphRagCommunities?.length ? 'ready' : data.graphRagCommunitiesError ? 'error' : 'empty') as CockpitLoadState,
    graphRagCommunitiesError: data.graphRagCommunitiesError ?? '',
    graphRagPath: pathMutation.data ?? null,
    graphRagPathState: (pathMutation.isPending ? 'loading' : pathMutation.isError ? 'error' : pathMutation.data ? pathMutation.data.status === 'found' ? 'ready' : 'empty' : 'idle') as CockpitLoadState,
    graphRagPathError: pathMutation.isError ? apiErrorMessage(pathMutation.error, 'Could not trace path') : '',
    graphRagProcesses: data.graphRagProcesses ?? [],
    graphRagProcessesState: (selection.view !== 'graphrag' ? 'idle' : activeQuery.isPending ? 'loading' : data.graphRagProcesses?.length ? 'ready' : data.graphRagProcessesError ? 'error' : 'empty') as CockpitLoadState,
    graphRagProcessesError: data.graphRagProcessesError ?? '',
    graphRagProcessDetail: processMutation.data ?? null,
    graphRagProcessDetailState: (processMutation.isPending ? 'loading' : processMutation.isError ? 'error' : processMutation.data ? 'ready' : 'idle') as CockpitLoadState,
    graphRagProcessDetailError: processMutation.isError ? apiErrorMessage(processMutation.error, 'Could not load process detail') : '',
    uiMapSummary: data.uiMapSummary ?? null,
    uiMapGraph: data.uiMapGraph ?? DEFAULT_GRAPH,
    userFlowsGraph: data.userFlowsGraph ?? DEFAULT_GRAPH,
    semanticMap: data.semanticMap ?? null,
    semanticMapComparison: data.semanticMapComparison ?? null,
    testMatrix: data.testMatrix ?? null,
    userFlows: data.userFlows ?? null,
    rebuildReadiness: data.rebuildReadiness ?? null,
    traceGraphRagPath, loadGraphRagProcessDetail, triggerCollectionReindex, regenerateArc42,
    generateExport, refreshActiveView,
  }
}
