export type CockpitLayer =
  | 'portfolio_system'
  | 'domain_container'
  | 'component_interface'
  | 'code_controlflow'

export type CockpitView = 'overview' | 'topology' | 'deep_dive' | 'c4_diff' | 'city' | 'graphrag' | 'exports'

export type CockpitLoadState = 'idle' | 'loading' | 'ready' | 'empty' | 'error'

export type ExportFormat = 'lpg_jsonl' | 'cc_json' | 'cx2' | 'jgf' | 'mermaid_c4'
export type CockpitProjection = 'architecture' | 'code_file' | 'code_symbol' | 'graphrag'
export type CityProjection = 'architecture' | 'code_file'
export type CityEntityLevel = 'domain' | 'container' | 'component'
export type TopologyEntityLevel = 'domain' | 'container' | 'component'
export type DeepDiveMode = 'file_dependency' | 'symbol_callgraph' | 'contains_hierarchy'
export type LayoutEngine = 'grid' | 'elk_layered' | 'elk_force_like'
export type OverlayMode = 'none' | 'runtime' | 'risk'

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

export interface ScenarioLite {
  id: string
  name: string
  version: number
  is_as_is: boolean
}

export interface ViewScenario {
  id: string
  collection_id: string
  name: string
  version: number
  is_as_is: boolean
  base_scenario_id: string | null
}

export interface TwinGraphNode {
  id: string
  natural_key: string
  kind: string
  name: string
  meta: Record<string, unknown>
}

export interface TwinGraphEdge {
  id: string
  source_node_id: string
  target_node_id: string
  kind: string
  meta: Record<string, unknown>
}

export interface TwinGraphResponse {
  nodes: TwinGraphNode[]
  edges: TwinGraphEdge[]
  page: number
  limit: number
  total_nodes: number
  projection?: CockpitProjection
  entity_level?: string
  grouping_strategy?: 'explicit' | 'heuristic' | 'mixed'
  excluded_kinds?: string[]
}

export interface GraphNeighborhoodResponse {
  scenario_id: string
  node_id: string
  hops: number
  projection: CockpitProjection
  graph: TwinGraphResponse
}

export interface GraphViewPayload {
  collection_id: string
  scenario: ViewScenario
  layer: CockpitLayer | null
  projection?: CockpitProjection
  entity_level?: string
  grouping_strategy?: 'explicit' | 'heuristic' | 'mixed'
  excluded_kinds?: string[]
  graph: TwinGraphResponse
}

export interface GraphRagStatus {
  status: 'ready' | 'unavailable'
  reason: 'ok' | 'no_knowledge_graph'
}

export interface GraphRagPayload {
  collection_id: string
  scenario: ViewScenario
  projection: 'graphrag'
  entity_level: 'knowledge_node'
  status: GraphRagStatus
  graph: TwinGraphResponse
}

export interface GraphRagEvidenceItem {
  evidence_id: string
  file_path: string
  start_line: number
  end_line: number
  text: string
  text_source: 'snippet' | 'document_lines' | 'unavailable'
}

export interface GraphRagEvidencePayload {
  collection_id: string
  node_id: string
  node_name: string
  node_kind: string
  items: GraphRagEvidenceItem[]
  total: number
}

export interface CityHotspot {
  node_natural_key: string
  loc: number
  symbol_count: number
  coverage: number
  complexity: number
  coupling: number
}

export interface MetricsStatus {
  status: 'ready' | 'unavailable'
  reason: 'ok' | 'no_real_metrics' | 'awaiting_ci_coverage' | 'coverage_ingest_failed'
  strict_mode: boolean
}

export interface CityPayload {
  collection_id: string
  scenario: ViewScenario
  summary: {
    metric_nodes: number
    coverage_avg: number | null
    complexity_avg: number | null
    coupling_avg: number | null
  }
  metrics_status: MetricsStatus
  hotspots: CityHotspot[]
  cc_json: Record<string, unknown>
}

export interface MermaidPayload {
  collection_id: string
  scenario: ViewScenario
  mode: 'single' | 'compare'
  content?: string
  as_is?: string
  to_be?: string
  as_is_scenario_id?: string
}

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
  { key: 'overview', label: 'Overview' },
  { key: 'topology', label: 'Topology' },
  { key: 'deep_dive', label: 'Deep Dive' },
  { key: 'c4_diff', label: 'C4 Diff' },
  { key: 'city', label: 'City' },
  { key: 'graphrag', label: 'GraphRAG' },
  { key: 'exports', label: 'Exports' },
]

export const COCKPIT_LAYERS: Array<{ key: CockpitLayer; label: string }> = [
  { key: 'code_controlflow', label: 'Code / Controlflow' },
  { key: 'portfolio_system', label: 'Portfolio / System' },
  { key: 'domain_container', label: 'Domain / Container' },
  { key: 'component_interface', label: 'Component / Interface' },
]

export const EXPORT_FORMATS: Array<{ key: ExportFormat; label: string; extension: string }> = [
  { key: 'cc_json', label: 'CodeCharta (cc.json)', extension: 'cc.json' },
  { key: 'cx2', label: 'CX2', extension: 'cx2.json' },
  { key: 'jgf', label: 'JGF', extension: 'jgf.json' },
  { key: 'lpg_jsonl', label: 'LPG JSONL', extension: 'lpg.jsonl' },
  { key: 'mermaid_c4', label: 'Mermaid C4', extension: 'mmd' },
]
