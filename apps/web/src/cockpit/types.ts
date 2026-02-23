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
  | 'graphrag'
  | 'ui_map'
  | 'test_matrix'
  | 'user_flows'
  | 'rebuild_readiness'
  | 'exports'

export type CockpitLoadState = 'idle' | 'loading' | 'ready' | 'empty' | 'error'

export type ExportFormat =
  | 'lpg_jsonl'
  | 'cc_json'
  | 'cx2'
  | 'jgf'
  | 'mermaid_c4'
  | 'twin_manifest'
export type CockpitProjection = 'architecture' | 'code_file' | 'code_symbol' | 'graphrag'
export type CityProjection = 'architecture' | 'code_file'
export type CityEntityLevel = 'domain' | 'container' | 'component'
export type TopologyEntityLevel = 'domain' | 'container' | 'component'
export type DeepDiveMode = 'file_dependency' | 'symbol_callgraph' | 'contains_hierarchy'
export type LayoutEngine = 'grid' | 'elk_layered' | 'elk_force_like'
export type OverlayMode = 'none' | 'runtime' | 'risk'
export type C4ViewMode = 'context' | 'container' | 'component' | 'code' | 'deployment'
export type GraphRagCommunityMode = 'none' | 'color' | 'focus'

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

export interface UIMapPayload {
  collection_id: string
  scenario: ViewScenario
  projection: 'ui_map'
  entity_level: 'ui'
  summary: {
    routes: number
    views: number
    components: number
    contracts: number
    trace_edges: number
  }
  warnings: string[]
  graph: TwinGraphResponse
}

export interface TestMatrixRow {
  test_case_id: string
  test_case_key: string
  test_case_name: string
  covers_symbols: string[]
  validates_rules: string[]
  fixtures: string[]
  verifies_flows: string[]
  evidence_ids: string[]
}

export interface TestMatrixPayload {
  collection_id: string
  scenario: ViewScenario
  projection: 'test_matrix'
  entity_level: 'test_case'
  summary: {
    test_cases: number
    test_suites: number
    test_fixtures: number
    matrix_rows: number
  }
  matrix: TestMatrixRow[]
  warnings: string[]
  graph: TwinGraphResponse
}

export interface UserFlowStep {
  step_id: string
  name: string
  order: number
  endpoint_hints: string[]
  calls_endpoints: string[]
  evidence_ids: string[]
}

export interface UserFlowItem {
  flow_id: string
  flow_key: string
  flow_name: string
  route_path: string
  steps: UserFlowStep[]
  verified_by_tests: string[]
  evidence_ids: string[]
}

export interface UserFlowsPayload {
  collection_id: string
  scenario: ViewScenario
  projection: 'user_flows'
  entity_level: 'user_flow'
  summary: {
    user_flows: number
    flow_steps: number
    flow_edges: number
  }
  flows: UserFlowItem[]
  warnings: string[]
  graph: TwinGraphResponse
}

export interface RebuildReadinessCriticalNode {
  node_id: string
  kind: string
  name: string
  confidence: number
  evidence_ids: string[]
}

export interface RebuildReadinessPayload {
  collection_id: string
  scenario: ViewScenario
  projection: 'rebuild_readiness'
  score: number
  summary: {
    interface_test_coverage: number
    flow_evidence_density: number
    ui_to_endpoint_traceability: number
    critical_inferred_only_count: number
    total_nodes: number
    total_edges: number
  }
  known_gaps: string[]
  critical_inferred_only: RebuildReadinessCriticalNode[]
  evidence_handles: Array<{
    kind: string
    ref: string
    node_id?: string
  }>
  behavioral_layers_status: string | null
  last_behavioral_materialized_at: string | null
  deep_warnings: string[]
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
  community_mode?: GraphRagCommunityMode
  community_id?: string | null
  status: GraphRagStatus
  graph: TwinGraphResponse
}

export interface GraphRagCommunityKindCount {
  kind: string
  count: number
}

export interface GraphRagCommunityNodePreview {
  id: string
  name: string
  kind: string
  natural_key: string
}

export interface GraphRagCommunity {
  id: string
  label: string
  size: number
  cohesion: number
  top_kinds: GraphRagCommunityKindCount[]
  sample_nodes: GraphRagCommunityNodePreview[]
}

export interface GraphRagCommunitiesPayload {
  collection_id: string
  scenario: ViewScenario
  items: GraphRagCommunity[]
  page: number
  limit: number
  total: number
}

export interface GraphRagPathNode {
  id: string
  natural_key: string
  kind: string
  name: string
  meta: Record<string, unknown>
}

export interface GraphRagPathEdge {
  id: string
  source_node_id: string
  target_node_id: string
  kind: string
  meta: Record<string, unknown>
}

export interface GraphRagPathPayload {
  collection_id: string
  scenario: ViewScenario
  status: 'found' | 'not_found' | 'truncated'
  from_node_id: string
  to_node_id: string
  max_hops: number
  path: {
    nodes: GraphRagPathNode[]
    edges: GraphRagPathEdge[]
    hops: number
  }
}

export interface GraphRagProcessSummary {
  id: string
  label: string
  process_type: 'intra_community' | 'cross_community'
  step_count: number
  community_ids: string[]
  entry_node_id: string
  terminal_node_id: string
}

export interface GraphRagProcessesPayload {
  collection_id: string
  scenario: ViewScenario
  items: GraphRagProcessSummary[]
  total: number
}

export interface GraphRagProcessStep {
  step: number
  node_id: string
  node_name: string
  node_kind: string
  node_natural_key: string
}

export interface GraphRagProcessEdge {
  id: string
  source_node_id: string
  target_node_id: string
  kind: string
  meta: Record<string, unknown>
}

export interface GraphRagProcessDetailPayload {
  collection_id: string
  scenario: ViewScenario
  process: GraphRagProcessSummary
  steps: GraphRagProcessStep[]
  edges: GraphRagProcessEdge[]
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
  change_frequency: number
  churn: number
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
    change_frequency_avg: number | null
    churn_avg: number | null
  }
  metrics_status: MetricsStatus
  hotspots: CityHotspot[]
  cc_json: Record<string, unknown>
}

export interface MermaidPayload {
  collection_id: string
  scenario: ViewScenario
  mode: 'single' | 'compare'
  c4_view?: C4ViewMode
  c4_scope?: string | null
  max_nodes?: number
  warnings?: string[]
  as_is_warnings?: string[]
  to_be_warnings?: string[]
  content?: string
  as_is?: string
  to_be?: string
  as_is_scenario_id?: string
}

export interface Arc42Payload {
  title: string
  generated_at: string
  sections: Record<string, string>
  markdown: string
  warnings: string[]
  confidence_summary: {
    total: number
    avg: number | null
    by_source: Record<string, { count: number; avg: number | null }>
  }
  section_coverage: Record<string, boolean>
}

export interface Arc42ViewPayload {
  collection_id: string
  scenario: ViewScenario
  artifact: {
    id: string
    name: string
    kind: string
    cached: boolean
  }
  section: string | null
  arc42: Arc42Payload
  facts_hash: string
  facts_count: number
  ports_adapters_count: number
  warnings: string[]
}

export interface PortAdapterEvidenceRef {
  kind: 'file' | 'node' | 'edge' | 'artifact'
  ref: string
  start_line?: number | null
  end_line?: number | null
}

export interface PortAdapterItem {
  fact_id: string
  direction: 'inbound' | 'outbound'
  port_name: string
  adapter_name: string | null
  container: string | null
  component: string | null
  protocol: string | null
  source: 'deterministic' | 'hybrid' | 'llm'
  confidence: number
  attributes: Record<string, unknown>
  evidence: PortAdapterEvidenceRef[]
}

export interface PortsAdaptersPayload {
  collection_id: string
  scenario: ViewScenario
  summary: {
    total: number
    inbound: number
    outbound: number
  }
  filters: {
    direction: string | null
    container: string | null
  }
  items: PortAdapterItem[]
  warnings: string[]
}

export interface Arc42DriftDelta {
  delta_type: 'added' | 'removed' | 'changed_confidence' | 'moved_component' | 'new_port' | 'removed_adapter'
  subject: string
  detail: string
  before?: Record<string, unknown> | null
  after?: Record<string, unknown> | null
  confidence: number
}

export interface Arc42DriftPayload {
  collection_id: string
  scenario: ViewScenario
  baseline_scenario: ViewScenario | null
  generated_at: string
  current_hash: string
  baseline_hash: string | null
  summary: {
    total: number
    by_type: Record<string, number>
    severity: 'low' | 'medium'
  }
  deltas: Arc42DriftDelta[]
  warnings: string[]
}

export interface ErmColumnItem {
  id: string
  natural_key: string
  name: string
  table: string | null
  type: string | null
  nullable: boolean
  primary_key: boolean
  foreign_key: string | null
}

export interface ErmTableItem {
  id: string
  natural_key: string
  name: string
  description: string | null
  column_count: number
  primary_keys: string[]
  columns: ErmColumnItem[]
}

export interface ErmForeignKeyItem {
  id: string
  fk_name: string | null
  source_table: string
  source_column: string
  target_table: string
  target_column: string
  source_column_node_id: string
  target_column_node_id: string
}

export interface ErmViewPayload {
  collection_id: string
  scenario: ViewScenario
  summary: {
    tables: number
    columns: number
    foreign_keys: number
    has_mermaid: boolean
  }
  tables: ErmTableItem[]
  foreign_keys: ErmForeignKeyItem[]
  mermaid: {
    artifact_id: string
    name: string
    content: string
    meta: Record<string, unknown>
  } | null
  warnings: string[]
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
  { key: 'architecture', label: 'Architecture' },
  { key: 'city', label: 'City' },
  { key: 'graphrag', label: 'GraphRAG' },
  { key: 'ui_map', label: 'UI Map' },
  { key: 'test_matrix', label: 'Test Matrix' },
  { key: 'user_flows', label: 'User Flows' },
  { key: 'rebuild_readiness', label: 'Rebuild Readiness' },
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
  { key: 'twin_manifest', label: 'Twin Manifest', extension: 'json' },
]
