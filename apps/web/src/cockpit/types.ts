export type CockpitLayer =
  | 'portfolio_system'
  | 'domain_container'
  | 'component_interface'
  | 'code_controlflow'

export type CockpitView = 'overview' | 'topology' | 'deep_dive' | 'c4_diff' | 'exports'

export type CockpitLoadState = 'idle' | 'loading' | 'ready' | 'empty' | 'error'

export type ExportFormat = 'lpg_jsonl' | 'cc_json' | 'cx2' | 'jgf' | 'mermaid_c4'

export interface CockpitSelection {
  collectionId: string
  scenarioId: string
  layer: CockpitLayer
  view: CockpitView
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
}

export interface GraphViewPayload {
  collection_id: string
  scenario: ViewScenario
  layer: CockpitLayer | null
  graph: TwinGraphResponse
}

export interface CityHotspot {
  node_natural_key: string
  loc: number
  symbol_count: number
  coverage: number
  complexity: number
  coupling: number
}

export interface CityPayload {
  collection_id: string
  scenario: ViewScenario
  summary: {
    metric_nodes: number
    coverage_avg: number
    complexity_avg: number
    coupling_avg: number
  }
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

export interface CockpitToast {
  id: number
  kind: 'success' | 'error' | 'info'
  message: string
}

export const DEFAULT_LAYER: CockpitLayer = 'domain_container'
export const DEFAULT_VIEW: CockpitView = 'overview'

export const COCKPIT_VIEWS: Array<{ key: CockpitView; label: string }> = [
  { key: 'overview', label: 'Overview' },
  { key: 'topology', label: 'Topology' },
  { key: 'deep_dive', label: 'Deep Dive' },
  { key: 'c4_diff', label: 'C4 Diff' },
  { key: 'exports', label: 'Exports' },
]

export const COCKPIT_LAYERS: Array<{ key: CockpitLayer; label: string }> = [
  { key: 'portfolio_system', label: 'Portfolio / System' },
  { key: 'domain_container', label: 'Domain / Container' },
  { key: 'component_interface', label: 'Component / Interface' },
  { key: 'code_controlflow', label: 'Code / Controlflow' },
]

export const EXPORT_FORMATS: Array<{ key: ExportFormat; label: string; extension: string }> = [
  { key: 'cc_json', label: 'CodeCharta (cc.json)', extension: 'cc.json' },
  { key: 'cx2', label: 'CX2', extension: 'cx2.json' },
  { key: 'jgf', label: 'JGF', extension: 'jgf.json' },
  { key: 'lpg_jsonl', label: 'LPG JSONL', extension: 'lpg.jsonl' },
  { key: 'mermaid_c4', label: 'Mermaid C4', extension: 'mmd' },
]
