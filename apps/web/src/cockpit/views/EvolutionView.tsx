import { useMemo } from 'react'
import ReactFlow, { Background, Controls, type Edge as RFEdge, type Node as RFNode } from 'reactflow'

import type {
  CockpitLoadState,
  FitnessFunctionsPayload,
  InvestmentUtilizationPayload,
  KnowledgeIslandsPayload,
  TemporalCouplingPayload,
} from '../types'

interface EvolutionViewProps {
  state: CockpitLoadState
  error: string
  investmentUtilization: InvestmentUtilizationPayload | null
  knowledgeIslands: KnowledgeIslandsPayload | null
  temporalCoupling: TemporalCouplingPayload | null
  fitnessFunctions: FitnessFunctionsPayload | null
  panelErrors: {
    investment: string
    knowledge: string
    coupling: string
    fitness: string
  }
  onRetry: () => void
}

function severityClass(severity: string): string {
  const normalized = severity.trim().toLowerCase()
  if (normalized === 'critical' || normalized === 'high') return 'risk-high'
  if (normalized === 'medium') return 'risk-medium'
  return 'risk-low'
}

function bubbleColor(quadrant: string): string {
  if (quadrant === 'strength') return '#166534'
  if (quadrant === 'overinvestment') return '#b91c1c'
  if (quadrant === 'efficient_core') return '#1d4ed8'
  if (quadrant === 'opportunity_or_retire') return '#92400e'
  return '#6b7280'
}

function toCouplingNodes(payload: TemporalCouplingPayload | null): RFNode[] {
  if (!payload) return []
  const nodes = payload.graph.nodes || []
  const count = nodes.length || 1
  const radius = Math.max(220, 50 + count * 8)

  return nodes.map((node, index) => {
    const angle = (2 * Math.PI * index) / count
    return {
      id: node.id,
      position: {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
      },
      data: {
        label: node.label,
      },
      style: {
        width: 180,
        borderRadius: 10,
        border: '1px solid #1f2937',
        fontSize: 11,
        background: '#ffffff',
      },
    }
  })
}

function toCouplingEdges(payload: TemporalCouplingPayload | null): RFEdge[] {
  if (!payload) return []
  return (payload.graph.edges || []).map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    type: 'smoothstep',
    label: `${edge.jaccard.toFixed(2)}`,
    style: {
      strokeWidth: Math.max(1, Math.min(7, 1 + edge.jaccard * 6)),
      stroke: edge.cross_boundary ? '#dc2626' : '#2563eb',
    },
  }))
}

export default function EvolutionView({
  state,
  error,
  investmentUtilization,
  knowledgeIslands,
  temporalCoupling,
  fitnessFunctions,
  panelErrors,
  onRetry,
}: EvolutionViewProps) {
  const couplingNodes = useMemo(() => toCouplingNodes(temporalCoupling), [temporalCoupling])
  const couplingEdges = useMemo(() => toCouplingEdges(temporalCoupling), [temporalCoupling])
  const investmentItems = investmentUtilization?.items || []
  const fitnessRules = fitnessFunctions?.rules || []
  const fitnessViolations = fitnessFunctions?.violations || []
  const maxBubbleSize = Math.max(...investmentItems.map((item) => item.size), 1)

  if (state === 'loading' && !investmentUtilization && !knowledgeIslands && !temporalCoupling && !fitnessFunctions) {
    return (
      <div className="cockpit2-skeleton-grid" id="cockpit-panel-evolution" role="tabpanel">
        <div className="cockpit2-skeleton-card" />
        <div className="cockpit2-skeleton-card" />
        <div className="cockpit2-skeleton-card tall" />
      </div>
    )
  }

  if (state === 'error' && !investmentUtilization && !knowledgeIslands && !temporalCoupling && !fitnessFunctions) {
    return (
      <section className="cockpit2-alert error" id="cockpit-panel-evolution" role="tabpanel">
        <h3>Evolution view failed</h3>
        <p>{error || 'Could not load evolution analytics payloads.'}</p>
        <button type="button" onClick={onRetry}>Retry</button>
      </section>
    )
  }

  return (
    <section className="cockpit2-panel" id="cockpit-panel-evolution" role="tabpanel">
      {error ? (
        <div className="cockpit2-alert error inline">
          <p>{error}</p>
          <button type="button" onClick={onRetry}>Retry</button>
        </div>
      ) : null}

      <div className="cockpit2-panel-header-row">
        <h3>Evolution analytics</h3>
        <p className="muted">Investment vs utilization, ownership, temporal coupling, and fitness</p>
      </div>

      <div className="cockpit2-architecture-grid">
        <article className="cockpit2-architecture-card">
          <div className="cockpit2-panel-header-row">
            <h4>Investment vs utilization</h4>
            <p className="muted">Bubble size = LOC</p>
          </div>

          {panelErrors.investment ? (
            <div className="cockpit2-alert error inline">
              <p>{panelErrors.investment}</p>
              <button type="button" className="secondary" onClick={onRetry}>Retry</button>
            </div>
          ) : null}

          {investmentItems.length === 0 ? (
            <div className="cockpit2-empty">
              <h3>No investment data available</h3>
              <p>Evolution snapshots may still be missing for this scenario.</p>
            </div>
          ) : (
            <div className="cockpit2-evolution-scatter-wrap">
              <svg viewBox="0 0 900 320" className="cockpit2-evolution-scatter" role="img" aria-label="Investment utilization chart">
                <rect x="40" y="20" width="820" height="260" fill="#ffffff" stroke="#e5e7eb" />
                <line x1="450" y1="20" x2="450" y2="280" stroke="#d1d5db" strokeDasharray="4 4" />
                <line x1="40" y1="150" x2="860" y2="150" stroke="#d1d5db" strokeDasharray="4 4" />
                <text x="45" y="14" fontSize="11" fill="#6b7280">Utilization ↑</text>
                <text x="790" y="302" fontSize="11" fill="#6b7280">Investment →</text>
                {investmentItems.slice(0, 60).map((item) => {
                  const x = 40 + item.investment_score * 820
                  const y = item.utilization_score === null ? 150 : 280 - item.utilization_score * 260
                  const radius = 6 + Math.sqrt(item.size / maxBubbleSize) * 24
                  return (
                    <g key={item.entity_key}>
                      <circle cx={x} cy={y} r={radius} fill={bubbleColor(item.quadrant)} opacity={0.75} />
                      <title>
                        {item.label} · invest={item.investment_score.toFixed(2)} · util={item.utilization_score === null ? 'n/a' : item.utilization_score.toFixed(2)}
                      </title>
                    </g>
                  )
                })}
              </svg>
              <div className="cockpit2-chip-row">
                {(investmentUtilization?.warnings || []).map((warning) => (
                  <span key={warning} className="cockpit2-warning-chip">{warning}</span>
                ))}
              </div>
            </div>
          )}
        </article>

        <article className="cockpit2-architecture-card">
          <div className="cockpit2-panel-header-row">
            <h4>Knowledge islands</h4>
            <p className="muted">Bus-factor and ownership concentration</p>
          </div>

          {panelErrors.knowledge ? (
            <div className="cockpit2-alert error inline">
              <p>{panelErrors.knowledge}</p>
              <button type="button" className="secondary" onClick={onRetry}>Retry</button>
            </div>
          ) : null}

          {knowledgeIslands?.entities?.length ? (
            <>
              <div className="cockpit2-arch-kpis">
                <div>
                  <strong>{knowledgeIslands.summary.bus_factor_global}</strong>
                  <span>Global bus factor</span>
                </div>
                <div>
                  <strong>{knowledgeIslands.summary.single_owner_files}</strong>
                  <span>Single-owner files</span>
                </div>
                <div>
                  <strong>{knowledgeIslands.summary.entities}</strong>
                  <span>Entities analyzed</span>
                </div>
              </div>
              <div className="cockpit2-table-wrap">
                <table className="cockpit2-table">
                  <thead>
                    <tr>
                      <th>Entity</th>
                      <th>Bus factor</th>
                      <th>Dominant owner</th>
                      <th>Single-owner ratio</th>
                    </tr>
                  </thead>
                  <tbody>
                    {knowledgeIslands.entities.slice(0, 15).map((item) => (
                      <tr key={item.entity_key}>
                        <td>{item.label}</td>
                        <td>{item.bus_factor}</td>
                        <td>{item.dominant_owner || 'n/a'}</td>
                        <td>{(item.single_owner_ratio * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="cockpit2-empty">
              <h3>No ownership data available</h3>
              <p>Run a Git-backed sync to compute knowledge islands.</p>
            </div>
          )}
        </article>

        <article className="cockpit2-architecture-card">
          <div className="cockpit2-panel-header-row">
            <h4>Temporal coupling</h4>
            <p className="muted">Cross-boundary edges are highlighted in red</p>
          </div>

          {panelErrors.coupling ? (
            <div className="cockpit2-alert error inline">
              <p>{panelErrors.coupling}</p>
              <button type="button" className="secondary" onClick={onRetry}>Retry</button>
            </div>
          ) : null}

          {couplingNodes.length > 0 ? (
            <div className="cockpit2-evolution-coupling-graph">
              <ReactFlow
                nodes={couplingNodes}
                edges={couplingEdges}
                fitView
                fitViewOptions={{ padding: 0.3 }}
                nodesDraggable={false}
                elementsSelectable={false}
                panOnDrag
                zoomOnScroll
              >
                <Background />
                <Controls />
              </ReactFlow>
            </div>
          ) : (
            <div className="cockpit2-empty">
              <h3>No coupling graph available</h3>
              <p>Temporal coupling edges are unavailable for this scenario.</p>
            </div>
          )}
        </article>
      </div>

      <article className="cockpit2-architecture-card">
        <div className="cockpit2-panel-header-row">
          <h4>Fitness functions</h4>
          <p className="muted">Persisted advisory findings from evolution checks</p>
        </div>

        {panelErrors.fitness ? (
          <div className="cockpit2-alert error inline">
            <p>{panelErrors.fitness}</p>
            <button type="button" className="secondary" onClick={onRetry}>Retry</button>
          </div>
        ) : null}

        {fitnessRules.length > 0 ? (
          <>
            <div className="cockpit2-chip-row">
              {fitnessRules.map((rule) => (
                <span key={rule.rule_id} className={`cockpit2-chip ${severityClass(rule.highest_severity)}`}>
                  {rule.rule_id} · open {rule.open}
                </span>
              ))}
            </div>
            <div className="cockpit2-table-wrap">
              <table className="cockpit2-table">
                <thead>
                  <tr>
                    <th>Rule</th>
                    <th>Severity</th>
                    <th>Status</th>
                    <th>Message</th>
                    <th>File</th>
                  </tr>
                </thead>
                <tbody>
                  {fitnessViolations.slice(0, 40).map((violation) => (
                    <tr key={violation.id}>
                      <td>{violation.rule_id}</td>
                      <td><span className={`cockpit2-chip ${severityClass(violation.severity)}`}>{violation.severity}</span></td>
                      <td>{violation.status}</td>
                      <td>{violation.message}</td>
                      <td>{violation.filename}:{violation.line_number}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div className="cockpit2-empty">
            <h3>No fitness findings available</h3>
            <p>Fitness rules have not produced findings for this scenario yet.</p>
          </div>
        )}
      </article>
    </section>
  )
}
