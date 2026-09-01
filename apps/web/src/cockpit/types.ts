import type { components } from '../api/schema'

type ApiSchemas = components['schemas']

export type CockpitLayer =
  | 'portfolio_system'
  | 'domain_container'
  | 'component_interface'
  | 'code_controlflow'

export type CockpitView =
  | 'overview'
  | 'topology'
  | 'deep_dive'
  | 'c4_diff'
  | 'architecture'
  | 'city'
  | 'evolution'
  | 'graphrag'
  | 'semantic_map'
  | 'ui_map'
  | 'test_matrix'
  | 'user_flows'
  | 'rebuild_readiness'
  | 'exports'

export type CockpitLoadState = 'idle' | 'loading' | 'ready' | 'empty' | 'error'
export type ExportFormat = 'lpg_jsonl' | 'cc_json' | 'cx2' | 'jgf' | 'mermaid_c4' | 'twin_manifest'
export type CockpitProjection = 'architecture' | 'code_file' | 'code_symbol' | 'graphrag'
export type ExportProjection = Exclude<CockpitProjection, 'graphrag'>
export type CityProjection = 'architecture' | 'code_file'
export type CityEntityLevel = 'domain' | 'container' | 'component'
export type TopologyEntityLevel = 'domain' | 'container' | 'component'
export type DeepDiveMode = 'file_dependency' | 'symbol_callgraph' | 'contains_hierarchy'
export type LayoutEngine = 'grid' | 'elk_layered' | 'elk_force_like'
export type OverlayMode = 'none' | 'runtime' | 'risk'
export type PortsDirection = 'all' | 'inbound' | 'outbound'
export type C4ViewMode = 'context' | 'container' | 'component' | 'code' | 'deployment'
export type GraphRagCommunityMode = 'none' | 'color' | 'focus'
export type SemanticMapMode = 'code_structure' | 'semantic'

export interface CockpitSelection {
  collectionId: string
  scenarioId: string
  layer: CockpitLayer
  view: CockpitView
}

export interface GraphFilters {
  query: string
  hideIsolated: boolean
  edgeKinds: string[]
  includeKinds: string[]
  excludeKinds: string[]
}

export interface GraphPagingState {
  page: number
  limit: number
}

export interface NodeInspectorState {
  nodeId: string
}

export interface CollectionLite {
  id: string
  name: string
}

export type ViewScenario = ApiSchemas['ViewScenario']
export type ScenarioLite = Pick<ViewScenario, 'id' | 'name' | 'version' | 'is_as_is'>
export type TwinGraphNode = ApiSchemas['TwinGraphNode']
export type TwinGraphEdge = ApiSchemas['TwinGraphEdge']
export type TwinGraphResponse = ApiSchemas['TwinGraphResponse']
export type GraphNeighborhoodResponse = ApiSchemas['GraphNeighborhoodResponse']
export type GraphViewPayload = ApiSchemas['GraphViewResponse']
export type UIMapPayload = ApiSchemas['UIMapResponse']
export type TestMatrixRow = ApiSchemas['TestMatrixRow']
export type TestMatrixPayload = ApiSchemas['TestMatrixResponse']
export type UserFlowStep = ApiSchemas['UserFlowStep']
export type UserFlowItem = ApiSchemas['UserFlowItem']
export type UserFlowsPayload = ApiSchemas['UserFlowsResponse']
export type RebuildReadinessCriticalNode = ApiSchemas['RebuildReadinessCriticalNode']
export type RebuildReadinessPayload = ApiSchemas['RebuildReadinessResponse']
export type GraphRagStatusReason = ApiSchemas['GraphRagStatus']['reason']
export type GraphRagStatus = ApiSchemas['GraphRagStatus']
export type GraphRagPayload = ApiSchemas['GraphRagResponse']
export type SemanticMapStatus = ApiSchemas['SemanticMapStatus']
export type SemanticMapPoint = ApiSchemas['SemanticMapPoint']
export type SemanticMapSignal = ApiSchemas['SemanticMapSignal']
export type SemanticMapPayload = ApiSchemas['SemanticMapResponse']
export type SemanticMapThresholds = ApiSchemas['SemanticMapThresholds']
export type GraphRagCommunityKindCount = ApiSchemas['KindCount']
export type GraphRagCommunityNodePreview = ApiSchemas['GraphRagCommunityNodePreview']
export type GraphRagCommunity = ApiSchemas['GraphRagCommunity']
export type GraphRagCommunitiesPayload = ApiSchemas['GraphRagCommunitiesResponse']
export type GraphRagPathNode = TwinGraphNode
export type GraphRagPathEdge = TwinGraphEdge
export type GraphRagPathPayload = ApiSchemas['GraphRagPathResponse']
export type GraphRagProcessSummary = ApiSchemas['GraphRagProcessSummary']
export type GraphRagProcessesPayload = ApiSchemas['GraphRagProcessesResponse']
export type GraphRagProcessStep = ApiSchemas['GraphRagProcessStep']
export type GraphRagProcessEdge = TwinGraphEdge
export type GraphRagProcessDetailPayload = ApiSchemas['GraphRagProcessDetailResponse']
export type GraphRagEvidenceItem = ApiSchemas['GraphRagEvidenceItem']
export type GraphRagEvidencePayload = ApiSchemas['GraphRagEvidenceResponse']
export type CityHotspot = ApiSchemas['CityHotspot']
export type MetricsStatus = ApiSchemas['MetricsStatus']
export type CityPayload = ApiSchemas['CityResponse']
export type InvestmentUtilizationItem = ApiSchemas['InvestmentUtilizationItem']
export type InvestmentUtilizationPayload = ApiSchemas['InvestmentUtilizationResponse']
export type KnowledgeIslandEntity = ApiSchemas['KnowledgeIslandEntity']
export type KnowledgeIslandFileRisk = ApiSchemas['KnowledgeIslandFileRisk']
export type KnowledgeIslandsPayload = ApiSchemas['KnowledgeIslandsResponse']
export type TemporalCouplingNode = ApiSchemas['TemporalCouplingNode']
export type TemporalCouplingEdge = ApiSchemas['TemporalCouplingEdge']
export type TemporalCouplingPayload = ApiSchemas['TemporalCouplingResponse']
export type SeverityLevel = ApiSchemas['FitnessViolationItem']['severity']
export type FitnessRuleSummary = ApiSchemas['FitnessRuleSummary']
export type FitnessViolationItem = ApiSchemas['FitnessViolationItem']
export type FitnessFunctionsPayload = ApiSchemas['FitnessFunctionsResponse']
export type MermaidPayload = ApiSchemas['MermaidResponse']
export type Arc42Payload = ApiSchemas['Arc42Payload']
export type Arc42ViewPayload = ApiSchemas['Arc42ViewResponse']
export type PortAdapterEvidenceRef = ApiSchemas['PortAdapterEvidenceRef']
export type PortAdapterItem = ApiSchemas['PortAdapterItem']
export type PortsAdaptersPayload = ApiSchemas['PortsAdaptersResponse']
export type Arc42DriftDelta = ApiSchemas['Arc42DriftDelta']
export type Arc42DriftPayload = ApiSchemas['Arc42DriftResponse']
export type ErmColumnItem = ApiSchemas['ErmColumnItem']
export type ErmTableItem = ApiSchemas['ErmTableItem']
export type ErmForeignKeyItem = ApiSchemas['ErmForeignKeyItem']
export type ErmViewPayload = ApiSchemas['ErmResponse']

export interface RuntimeOverlayMetric {
  service: string
  latency_p95?: number
  error_rate?: number
}

export interface RiskOverlayMetric {
  node: string
  vuln_count?: number
  severity_score?: number
}

export interface OverlayState {
  mode: OverlayMode
  runtimeByNodeKey: Record<string, RuntimeOverlayMetric>
  riskByNodeKey: Record<string, RiskOverlayMetric>
  loadedAt: string | null
}

export interface CockpitToast {
  id: number
  kind: 'success' | 'error' | 'info'
  message: string
}

export const DEFAULT_LAYER: CockpitLayer = 'code_controlflow'
export const DEFAULT_VIEW: CockpitView = 'overview'

export const COCKPIT_VIEWS: Array<{ key: CockpitView; label: string }> = [
  { key: 'overview', label: 'Overview' }, { key: 'topology', label: 'Topology' },
  { key: 'deep_dive', label: 'Deep Dive' }, { key: 'c4_diff', label: 'C4 Diff' },
  { key: 'architecture', label: 'Architecture' }, { key: 'city', label: 'City' },
  { key: 'evolution', label: 'Evolution' }, { key: 'graphrag', label: 'GraphRAG' },
  { key: 'semantic_map', label: 'Semantic Map' }, { key: 'ui_map', label: 'UI & Flows' },
  { key: 'test_matrix', label: 'Test Matrix' }, { key: 'rebuild_readiness', label: 'Rebuild Readiness' },
  { key: 'exports', label: 'Exports' },
]

export const COCKPIT_LAYERS: Array<{ key: CockpitLayer; label: string }> = [
  { key: 'code_controlflow', label: 'Code / Controlflow' },
  { key: 'portfolio_system', label: 'Portfolio / System' },
  { key: 'domain_container', label: 'Domain / Container' },
  { key: 'component_interface', label: 'Component / Interface' },
]

const LAYER_LABEL_MAP = Object.fromEntries(
  COCKPIT_LAYERS.map((layer) => [layer.key, layer.label]),
) as Record<CockpitLayer, string>

export function layerLabel(layer: CockpitLayer): string {
  return LAYER_LABEL_MAP[layer] ?? layer
}

export const EXPORT_FORMATS: Array<{ key: ExportFormat; label: string; extension: string }> = [
  { key: 'cc_json', label: 'CodeCharta (cc.json)', extension: 'cc.json' },
  { key: 'cx2', label: 'CX2', extension: 'cx2.json' },
  { key: 'jgf', label: 'JGF', extension: 'jgf.json' },
  { key: 'lpg_jsonl', label: 'LPG JSONL', extension: 'lpg.jsonl' },
  { key: 'mermaid_c4', label: 'Mermaid C4', extension: 'mmd' },
  { key: 'twin_manifest', label: 'Twin Manifest', extension: 'json' },
]
