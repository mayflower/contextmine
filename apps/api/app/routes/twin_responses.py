"""OpenAPI response schemas for the digital-twin REST API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TwinResponse(BaseModel):
    pass


class ViewScenario(TwinResponse):
    id: str
    collection_id: str
    name: str
    version: int
    is_as_is: bool
    base_scenario_id: str | None


class ScenarioListResponse(TwinResponse):
    scenarios: list[ViewScenario]


class TwinGraphNode(TwinResponse):
    id: str
    natural_key: str
    kind: str
    name: str
    meta: dict[str, Any]


class TwinGraphEdge(TwinResponse):
    id: str
    source_node_id: str
    target_node_id: str
    kind: str
    meta: dict[str, Any]


class TwinGraphResponse(TwinResponse):
    nodes: list[TwinGraphNode]
    edges: list[TwinGraphEdge]
    page: int
    limit: int
    total_nodes: int
    projection: Literal["architecture", "code_file", "code_symbol", "graphrag"] | None = None
    entity_level: str | None = None
    grouping_strategy: Literal["explicit", "heuristic", "mixed"] | None = None
    excluded_kinds: list[str] | None = None
    slice_strategy: str | None = None
    sorted_by: str | None = None
    candidate_nodes: int | None = None
    visible_nodes: int | None = None
    visible_edges: int | None = None
    dropped_cross_page_edges: int | None = None
    warnings: list[str] | None = None
    provenance: dict[str, Any] | None = None


class GraphNeighborhoodResponse(TwinResponse):
    scenario_id: str
    node_id: str
    hops: int
    projection: Literal["architecture", "code_file", "code_symbol", "graphrag"]
    graph: TwinGraphResponse


class GraphViewResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    layer: str | None = None
    projection: Literal["architecture", "code_file", "code_symbol", "graphrag"] | None = None
    entity_level: str | None = None
    grouping_strategy: Literal["explicit", "heuristic", "mixed"] | None = None
    excluded_kinds: list[str] | None = None
    warnings: list[str] | None = None
    provenance: dict[str, Any] | None = None
    graph: TwinGraphResponse


class UIMapSummary(TwinResponse):
    routes: int
    views: int
    components: int
    contracts: int
    trace_edges: int


class UIMapResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    projection: Literal["ui_map"]
    entity_level: Literal["ui"]
    summary: UIMapSummary
    warnings: list[str]
    graph_source: Literal["scenario", "knowledge_recovery"]
    graph: TwinGraphResponse


class TestMatrixRow(TwinResponse):
    test_case_id: str
    test_case_key: str
    test_case_name: str
    covers_symbols: list[str]
    validates_rules: list[str]
    fixtures: list[str]
    verifies_flows: list[str]
    evidence_ids: list[str]


class TestMatrixSummary(TwinResponse):
    test_cases: int
    test_suites: int
    test_fixtures: int
    matrix_rows: int


class TestMatrixResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    projection: Literal["test_matrix"]
    entity_level: Literal["test_case"]
    summary: TestMatrixSummary
    matrix: list[TestMatrixRow]
    warnings: list[str]
    graph: TwinGraphResponse


class UserFlowStep(TwinResponse):
    step_id: str
    name: str
    order: int
    endpoint_hints: list[str]
    calls_endpoints: list[str]
    evidence_ids: list[str]


class UserFlowItem(TwinResponse):
    flow_id: str
    flow_key: str
    flow_name: str
    route_path: str
    steps: list[UserFlowStep]
    verified_by_tests: list[str]
    evidence_ids: list[str]


class UserFlowsSummary(TwinResponse):
    user_flows: int
    flow_steps: int
    flow_edges: int


class UserFlowsResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    projection: Literal["user_flows"]
    entity_level: Literal["user_flow"]
    summary: UserFlowsSummary
    flows: list[UserFlowItem]
    warnings: list[str]
    graph_source: Literal["scenario", "knowledge_recovery"]
    graph: TwinGraphResponse


class RebuildReadinessResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    projection: Literal["rebuild_readiness"]
    score: float
    summary: RebuildReadinessSummary
    known_gaps: list[str]
    critical_inferred_only: list[RebuildReadinessCriticalNode]
    evidence_handles: list[EvidenceHandle]
    behavioral_layers_status: str | None
    last_behavioral_materialized_at: str | None
    deep_warnings: list[str]
    scip_status: Literal["ready", "degraded", "failed"] | None = None
    scip_projects_by_language: dict[str, int] | None = None
    scip_failed_projects: list[dict[str, str]] | None = None
    metrics_gate: MetricsGate | None = None


class RebuildReadinessSummary(TwinResponse):
    interface_test_coverage: float
    flow_evidence_density: float
    ui_to_endpoint_traceability: float
    critical_inferred_only_count: int
    total_nodes: int
    total_edges: int


class RebuildReadinessCriticalNode(TwinResponse):
    node_id: str
    kind: str
    name: str
    confidence: float
    evidence_ids: list[str]


class EvidenceHandle(TwinResponse):
    kind: str
    ref: str
    node_id: str | None = None


class MetricsGate(TwinResponse):
    status: Literal["pass", "fail"] | None = None
    requested_files: int | None = None
    mapped_files: int | None = None
    unmapped_sample: list[str] | None = None


class GraphRagStatus(TwinResponse):
    status: Literal["ready", "unavailable"]
    reason: Literal["ok", "no_knowledge_graph", "no_graphrag_semantic_graph", "degraded_no_edges"]


class GraphRagResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    projection: Literal["graphrag"]
    entity_level: Literal["knowledge_node"]
    community_mode: str | None = None
    community_id: str | None = None
    status: GraphRagStatus
    warnings: list[str] = Field(default_factory=list)
    graph: TwinGraphResponse


class KindCount(TwinResponse):
    kind: str
    count: int


class GraphRagCommunityNodePreview(TwinResponse):
    id: str
    name: str
    kind: str
    natural_key: str


class GraphRagCommunity(TwinResponse):
    id: str
    label: str
    size: int
    cohesion: float
    top_kinds: list[KindCount]
    sample_nodes: list[GraphRagCommunityNodePreview]


class GraphRagCommunitiesResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    items: list[GraphRagCommunity]
    page: int
    limit: int
    total: int


class GraphRagPathResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    status: Literal["found", "not_found", "truncated"]
    from_node_id: str
    to_node_id: str
    max_hops: int
    path: GraphRagPath


class GraphRagPath(TwinResponse):
    nodes: list[TwinGraphNode]
    edges: list[TwinGraphEdge]
    hops: int


class GraphRagProcessSummary(TwinResponse):
    id: str
    label: str
    process_type: Literal["intra_community", "cross_community"]
    step_count: int
    community_ids: list[str]
    entry_node_id: str
    terminal_node_id: str


class GraphRagProcessesResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    items: list[GraphRagProcessSummary]
    total: int


class GraphRagProcessDetailResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    process: GraphRagProcessSummary
    steps: list[GraphRagProcessStep]
    edges: list[TwinGraphEdge]


class GraphRagProcessStep(TwinResponse):
    step: int
    node_id: str
    node_name: str
    node_kind: str
    node_natural_key: str


class GraphRagEvidenceItem(TwinResponse):
    evidence_id: str
    file_path: str
    start_line: int
    end_line: int
    text: str
    text_source: Literal["snippet", "document_lines", "unavailable"]


class GraphRagEvidenceResponse(TwinResponse):
    collection_id: str
    node_id: str
    node_name: str
    node_kind: str
    items: list[GraphRagEvidenceItem]
    total: int


class SemanticMapThresholds(TwinResponse):
    mixed_cluster_max_dominant_ratio: float
    isolated_distance_multiplier: float
    semantic_duplication_min_similarity: float
    semantic_duplication_max_source_overlap: float
    misplaced_min_dominant_ratio: float


class SemanticMapStatus(TwinResponse):
    status: Literal["ready", "unavailable"]
    reason: Literal[
        "ok", "no_symbol_communities", "no_semantic_communities", "no_community_embeddings"
    ]


class SemanticMapPoint(TwinResponse):
    id: str
    label: str
    x: float
    y: float
    member_count: int
    cohesion: float
    top_kinds: list[KindCount]
    domain_counts: list[DomainCount]
    dominant_domain: str | None
    dominant_ratio: float
    summary: str | None
    anchor_node_id: str
    sample_nodes: list[GraphRagCommunityNodePreview]
    member_node_ids: list[str]


class SemanticMapSignal(TwinResponse):
    community_id: str | None = None
    left_community_id: str | None = None
    right_community_id: str | None = None
    left_label: str | None = None
    right_label: str | None = None
    label: str | None = None
    score: float
    anchor_node_id: str
    reason: str
    sample_nodes: list[SemanticMapSampleNode] | None = None


class DomainCount(TwinResponse):
    domain: str
    count: int


class SemanticMapSampleNode(TwinResponse):
    id: str
    name: str
    kind: str
    domain: str


class SemanticMapSignals(TwinResponse):
    mixed_clusters: list[SemanticMapSignal]
    isolated_points: list[SemanticMapSignal]
    semantic_duplication: list[SemanticMapSignal]
    misplaced_code: list[SemanticMapSignal]


class SemanticMapSummary(TwinResponse):
    points: int
    mixed_clusters: int
    isolated_points: int
    semantic_duplication: int
    misplaced_code: int


class SemanticMapResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    projection: Literal["semantic_map"]
    map_mode: Literal["code_structure", "semantic"]
    status: SemanticMapStatus
    thresholds: SemanticMapThresholds
    summary: SemanticMapSummary
    warnings: list[str]
    signals: SemanticMapSignals
    points: list[SemanticMapPoint]


class CityHotspot(TwinResponse):
    node_natural_key: str
    loc: float
    symbol_count: int
    coverage: float
    complexity: float
    coupling: float
    change_frequency: float
    churn: float


class MetricsStatus(TwinResponse):
    status: Literal["ready", "unavailable"]
    reason: Literal["ok", "no_real_metrics", "awaiting_ci_coverage", "coverage_ingest_failed"]
    strict_mode: bool


class CitySummary(TwinResponse):
    metric_nodes: int
    coverage_avg: float | None
    complexity_avg: float | None
    coupling_avg: float | None
    change_frequency_avg: float | None
    churn_avg: float | None


class CityResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    summary: CitySummary
    metrics_status: MetricsStatus
    hotspots: list[CityHotspot]
    cc_json: dict[str, Any]


class EvolutionResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    status: Literal["ready", "unavailable"]
    reason: str
    entity_level: str | None = None
    window_days: int
    warnings: list[str]


class InvestmentUtilizationItem(TwinResponse):
    entity_key: str
    label: str
    size: float
    investment_score: float
    utilization_score: float | None
    coverage_avg: float | None
    change_frequency_avg: float
    churn_avg: float
    quadrant: Literal[
        "strength", "overinvestment", "efficient_core", "opportunity_or_retire", "unknown"
    ]


class InvestmentUtilizationSummary(TwinResponse):
    total_entities: int
    coverage_entity_ratio: float
    utilization_available: bool
    quadrants: dict[str, int]


class InvestmentUtilizationResponse(EvolutionResponse):
    entity_level: Literal["container", "component"]
    summary: InvestmentUtilizationSummary
    items: list[InvestmentUtilizationItem]


class KnowledgeIslandEntity(TwinResponse):
    entity_key: str
    label: str
    files: int
    bus_factor: int
    dominant_owner: str | None
    dominant_share: float
    single_owner_ratio: float


class KnowledgeIslandFileRisk(TwinResponse):
    node_natural_key: str
    path: str | None
    entity_key: str
    dominant_owner: str
    dominant_share: float
    additions_total: int
    touches: int
    single_owner: bool
    churn: float
    coverage: float | None
    last_touched_at: str | None


class KnowledgeIslandsSummary(TwinResponse):
    files: int
    entities: int
    bus_factor_global: int
    single_owner_files: int
    churn_p75: float


class KnowledgeIslandsResponse(EvolutionResponse):
    entity_level: Literal["container", "component"]
    ownership_threshold: float
    summary: KnowledgeIslandsSummary
    entities: list[KnowledgeIslandEntity]
    at_risk_files: list[KnowledgeIslandFileRisk]


class TemporalCouplingNode(TwinResponse):
    id: str
    key: str
    label: str
    entity_level: Literal["file", "container", "component"]


class TemporalCouplingEdge(TwinResponse):
    id: str
    source: str
    target: str
    co_change_count: int
    source_change_count: int
    target_change_count: int
    ratio_source_to_target: float
    ratio_target_to_source: float
    jaccard: float
    cross_boundary: bool


class TemporalCouplingGraph(TwinResponse):
    nodes: list[TemporalCouplingNode]
    edges: list[TemporalCouplingEdge]


class TemporalCouplingSummary(TwinResponse):
    nodes: int
    edges: int
    cross_boundary_edges: int
    avg_jaccard: float


class TemporalCouplingResponse(EvolutionResponse):
    entity_level: Literal["file", "container", "component"]
    min_jaccard: float
    max_edges: int
    summary: TemporalCouplingSummary
    graph: TemporalCouplingGraph


Severity = Literal["critical", "high", "medium", "low"]


class FitnessRuleSummary(TwinResponse):
    rule_id: str
    finding_type: str
    count: int
    open: int
    resolved: int
    highest_severity: Severity


class FitnessViolationItem(TwinResponse):
    id: str
    rule_id: str
    finding_type: str
    severity: Severity
    confidence: str
    status: str
    subject: str | None
    message: str
    filename: str
    line_number: int
    created_at: str
    updated_at: str
    meta: dict[str, Any]


class FitnessFunctionsSummary(TwinResponse):
    rules: int
    violations: int
    open: int
    resolved: int
    highest_severity: Severity


class FitnessFunctionsResponse(EvolutionResponse):
    include_resolved: bool
    summary: FitnessFunctionsSummary
    rules: list[FitnessRuleSummary]
    violations: list[FitnessViolationItem]


class ConfidenceSourceSummary(TwinResponse):
    count: int
    avg: float | None


class ConfidenceSummary(TwinResponse):
    total: int
    avg: float | None = None
    by_source: dict[str, ConfidenceSourceSummary] = Field(default_factory=dict)


class Arc42Payload(TwinResponse):
    title: str
    generated_at: str
    sections: dict[str, str]
    markdown: str
    warnings: list[str]
    confidence_summary: ConfidenceSummary
    section_coverage: dict[str, bool]


class Arc42Artifact(TwinResponse):
    id: str
    name: str
    kind: str
    cached: bool


class Arc42ViewResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    artifact: Arc42Artifact
    section: str | None
    arc42: Arc42Payload
    facts_hash: str | None = None
    facts_count: int | None = None
    ports_adapters_count: int | None = None
    warnings: list[str]


class PortAdapterEvidenceRef(TwinResponse):
    kind: Literal["file", "node", "edge", "artifact"]
    ref: str
    start_line: int | None = None
    end_line: int | None = None


class PortAdapterItem(TwinResponse):
    fact_id: str
    direction: Literal["inbound", "outbound"]
    port_name: str
    adapter_name: str | None
    container: str | None
    component: str | None
    protocol: str | None
    source: Literal["deterministic", "hybrid", "llm"]
    confidence: float
    attributes: dict[str, Any]
    evidence: list[PortAdapterEvidenceRef]


class PortsAdaptersSummary(TwinResponse):
    total: int
    inbound: int
    outbound: int


class PortsAdaptersFilters(TwinResponse):
    direction: str | None
    container: str | None


class PortsAdaptersResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    summary: PortsAdaptersSummary
    filters: PortsAdaptersFilters
    items: list[PortAdapterItem]
    warnings: list[str]


class Arc42DriftDelta(TwinResponse):
    delta_type: Literal[
        "added", "removed", "changed_confidence", "moved_component", "new_port", "removed_adapter"
    ]
    subject: str
    detail: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    confidence: float


class Arc42DriftSummary(TwinResponse):
    total: int
    by_type: dict[str, int]
    severity: Literal["low", "medium"]


class Arc42DriftResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    baseline_scenario: ViewScenario | None
    generated_at: str
    current_hash: str
    baseline_hash: str | None
    summary: Arc42DriftSummary
    deltas: list[Arc42DriftDelta]
    warnings: list[str]


class ErmColumnItem(TwinResponse):
    id: str
    natural_key: str
    name: str
    table: str | None
    type: str | None
    nullable: bool
    primary_key: bool
    foreign_key: str | None


class ErmTableItem(TwinResponse):
    id: str
    natural_key: str
    name: str
    description: str | None
    column_count: int
    primary_keys: list[str]
    columns: list[ErmColumnItem]


class ErmForeignKeyItem(TwinResponse):
    id: str
    fk_name: str | None
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    source_column_node_id: str
    target_column_node_id: str


class ErmSummary(TwinResponse):
    tables: int
    columns: int
    foreign_keys: int
    has_mermaid: bool


class ErmMermaid(TwinResponse):
    artifact_id: str
    name: str
    content: str
    meta: dict[str, Any]


class ErmResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    summary: ErmSummary
    tables: list[ErmTableItem]
    foreign_keys: list[ErmForeignKeyItem]
    mermaid: ErmMermaid | None
    warnings: list[str]


class MermaidResponse(TwinResponse):
    collection_id: str
    scenario: ViewScenario
    mode: Literal["single", "compare"]
    c4_view: str | None = None
    c4_scope: str | None = None
    max_nodes: int | None = None
    warnings: list[str] | None = None
    as_is_warnings: list[str] | None = None
    to_be_warnings: list[str] | None = None
    content: str | None = None
    as_is: str | None = None
    to_be: str | None = None
    as_is_scenario_id: str | None = None


class RefreshResponse(TwinResponse):
    collection_id: str
    created: int
    skipped: int
    items: list[dict[str, Any]]


class ExportItem(TwinResponse):
    id: str
    name: str


class CreateExportResponse(TwinResponse):
    id: str | None = None
    name: str | None = None
    kind: str | None = None
    format: str | None = None
    exports: list[ExportItem] | None = None


class ExportArtifactResponse(TwinResponse):
    id: str
    name: str
    kind: str
    content: str
    meta: dict[str, Any]
    updated_at: Any
