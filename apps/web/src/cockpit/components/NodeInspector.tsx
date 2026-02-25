import type { CockpitLoadState, OverlayState, TwinGraphResponse } from '../types'

interface NodeInspectorProps {
  selectedNodeId: string
  graph: TwinGraphResponse
  neighborhood: TwinGraphResponse
  neighborhoodState: CockpitLoadState
  neighborhoodError: string
  overlay: OverlayState
  onClearSelection: () => void
}

function pretty(value: unknown): string {
  if (value === null || value === undefined) return 'N/A'
  if (typeof value === 'number') return Number.isFinite(value) ? value.toFixed(2) : 'N/A'
  return String(value)
}

function riskLevel(score: number, count: number): 'high' | 'medium' | 'low' {
  if (score >= 8 || count >= 10) return 'high'
  if (score >= 4 || count >= 1) return 'medium'
  return 'low'
}

export default function NodeInspector({
  selectedNodeId,
  graph,
  neighborhood,
  neighborhoodState,
  neighborhoodError,
  overlay,
  onClearSelection,
}: NodeInspectorProps) {
  const node = graph.nodes.find((entry) => entry.id === selectedNodeId) ?? null
  if (!node) {
    return (
      <aside className="cockpit2-panel cockpit2-inspector">
        <div className="cockpit2-panel-header-row">
          <h3>Node inspector</h3>
        </div>
        <p className="muted">Select a node in Topology, Deep Dive, or Semantic Map to inspect details.</p>
      </aside>
    )
  }

  const runtime = overlay.runtimeByNodeKey[node.natural_key] || overlay.runtimeByNodeKey[node.name]
  const risk = overlay.riskByNodeKey[node.natural_key] || overlay.riskByNodeKey[node.name]
  const metaRiskCount = Number((node.meta?.vuln_count as number) || 0)
  const metaRiskScore = Number((node.meta?.severity_score as number) || 0)
  const vulnCount = Number(risk?.vuln_count || metaRiskCount || 0)
  const severity = Number(risk?.severity_score || metaRiskScore || 0)
  const level = riskLevel(severity, vulnCount)

  return (
    <aside className="cockpit2-panel cockpit2-inspector">
      <div className="cockpit2-panel-header-row">
        <h3>Node inspector</h3>
        <button type="button" className="ghost" onClick={onClearSelection}>
          Clear
        </button>
      </div>

      <div className="cockpit2-inspector-summary">
        <strong>{node.name}</strong>
        <span className="muted">{node.kind}</span>
        <code>{node.natural_key}</code>
      </div>

      <div className="cockpit2-inspector-grid">
        <div>
          <span>Runtime p95</span>
          <strong>{pretty(runtime?.latency_p95)}</strong>
        </div>
        <div>
          <span>Error rate</span>
          <strong>{pretty(runtime?.error_rate)}</strong>
        </div>
        <div>
          <span>Vulnerabilities</span>
          <strong>{vulnCount}</strong>
        </div>
        <div>
          <span>Severity score</span>
          <strong>{pretty(severity)}</strong>
        </div>
      </div>

      <p className={`cockpit2-risk-badge level-${level}`}>Risk: {level}</p>

      <details open>
        <summary>Metadata</summary>
        <pre>{JSON.stringify(node.meta || {}, null, 2)}</pre>
      </details>

      <details open>
        <summary>Neighborhood (1 hop)</summary>
        {neighborhoodState === 'loading' ? (
          <p className="muted">Loading neighborhood…</p>
        ) : neighborhoodState === 'error' ? (
          <p className="muted">{neighborhoodError}</p>
        ) : (
          <p className="muted">
            Nodes: {neighborhood.nodes.length} • Edges: {neighborhood.edges.length}
          </p>
        )}
      </details>
    </aside>
  )
}
