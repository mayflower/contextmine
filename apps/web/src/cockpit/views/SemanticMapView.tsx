import { useMemo } from 'react'

import type {
  CockpitLoadState,
  SemanticMapMode,
  SemanticMapPayload,
  SemanticMapPoint,
  SemanticMapSignal,
} from '../types'

interface SemanticMapDiffItem {
  point_id: string
  point_label: string
  matched_point_id: string
  matched_point_label: string
  drift: number
  overlap_ratio: number
  overlap_count: number
  anchor_node_id: string
}

interface SemanticMapViewProps {
  state: CockpitLoadState
  error: string
  payload: SemanticMapPayload | null
  comparisonPayload: SemanticMapPayload | null
  mode: SemanticMapMode
  selectedNodeId: string
  showDiffOverlay: boolean
  diffMinDrift: number
  onModeChange: (mode: SemanticMapMode) => void
  onSelectNodeId: (nodeId: string) => void
  onRetry: () => void
}

const DOMAIN_COLORS = [
  '#0ea5e9',
  '#22c55e',
  '#f59e0b',
  '#8b5cf6',
  '#ef4444',
  '#14b8a6',
  '#f97316',
  '#3b82f6',
]

function colorForDomain(domain: string | null): string {
  if (!domain) return '#64748b'
  const normalized = domain.toLowerCase()
  let hash = 0
  for (let i = 0; i < normalized.length; i += 1) {
    hash = (hash * 31 + normalized.charCodeAt(i)) >>> 0
  }
  return DOMAIN_COLORS[hash % DOMAIN_COLORS.length]
}

function isPointSelected(point: SemanticMapPoint, selectedNodeId: string): boolean {
  if (!selectedNodeId) return false
  if (point.anchor_node_id === selectedNodeId) return true
  return point.sample_nodes.some((node) => node.id === selectedNodeId)
}

function renderSignals(
  title: string,
  signals: SemanticMapSignal[],
  onSelectNodeId: (nodeId: string) => void,
) {
  return (
    <section className="cockpit2-semantic-signal-block">
      <header>
        <h4>{title}</h4>
        <span className="muted">{signals.length}</span>
      </header>
      {signals.length === 0 ? (
        <p className="muted">No findings in this category.</p>
      ) : (
        <div className="cockpit2-semantic-signal-list">
          {signals.slice(0, 8).map((signal, index) => (
            <article key={`${title}-${index}`} className="cockpit2-semantic-signal-item">
              <div>
                <strong>
                  {signal.label || signal.left_label || signal.community_id || 'Signal'}
                  {signal.right_label ? ` ↔ ${signal.right_label}` : ''}
                </strong>
                <p className="muted">{signal.reason}</p>
              </div>
              <div className="actions">
                <span className="cockpit2-warning-chip">Score {signal.score.toFixed(2)}</span>
                {signal.anchor_node_id ? (
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => onSelectNodeId(signal.anchor_node_id)}
                  >
                    Focus node
                  </button>
                ) : null}
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}

function renderDiffSignals(
  items: SemanticMapDiffItem[],
  options: {
    threshold: number
    onSelectNodeId: (nodeId: string) => void
  },
) {
  const { threshold, onSelectNodeId } = options
  const flagged = items.filter((item) => item.drift >= threshold)
  return (
    <section className="cockpit2-semantic-signal-block">
      <header>
        <h4>Code vs semantic drift</h4>
        <span className="muted">{flagged.length}</span>
      </header>
      {flagged.length === 0 ? (
        <p className="muted">No drift findings above current threshold.</p>
      ) : (
        <div className="cockpit2-semantic-signal-list">
          {flagged.slice(0, 10).map((item) => (
            <article key={`${item.point_id}:${item.matched_point_id}`} className="cockpit2-semantic-signal-item">
              <div>
                <strong>{item.point_label}</strong>
                <p className="muted">
                  Best match: {item.matched_point_label} • overlap {item.overlap_count} nodes
                </p>
              </div>
              <div className="actions">
                <span className="cockpit2-warning-chip">Drift {item.drift.toFixed(2)}</span>
                {item.anchor_node_id ? (
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => onSelectNodeId(item.anchor_node_id)}
                  >
                    Focus node
                  </button>
                ) : null}
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}

export default function SemanticMapView({
  state,
  error,
  payload,
  comparisonPayload,
  mode,
  selectedNodeId,
  showDiffOverlay,
  diffMinDrift,
  onModeChange,
  onSelectNodeId,
  onRetry,
}: SemanticMapViewProps) {
  const points = payload?.points || []

  const diffItems = useMemo(() => {
    if (!payload || !comparisonPayload) {
      return [] as SemanticMapDiffItem[]
    }

    const otherPointsById = new Map(comparisonPayload.points.map((point) => [point.id, point]))
    const nodeToOtherPointIds = new Map<string, string[]>()
    comparisonPayload.points.forEach((point) => {
      point.member_node_ids.forEach((nodeId) => {
        const existing = nodeToOtherPointIds.get(nodeId)
        if (existing) {
          existing.push(point.id)
        } else {
          nodeToOtherPointIds.set(nodeId, [point.id])
        }
      })
    })

    const items: SemanticMapDiffItem[] = []

    points.forEach((point) => {
      if (!point.member_node_ids.length) {
        return
      }
      const overlaps = new Map<string, number>()
      point.member_node_ids.forEach((memberId) => {
        const otherPointIds = nodeToOtherPointIds.get(memberId) || []
        otherPointIds.forEach((otherPointId) => {
          overlaps.set(otherPointId, (overlaps.get(otherPointId) || 0) + 1)
        })
      })
      if (overlaps.size === 0) {
        return
      }

      let bestPointId = ''
      let bestOverlap = 0
      overlaps.forEach((count, otherPointId) => {
        if (count > bestOverlap) {
          bestOverlap = count
          bestPointId = otherPointId
        }
      })

      if (!bestPointId || bestOverlap === 0) {
        return
      }

      const matchedPoint = otherPointsById.get(bestPointId)
      if (!matchedPoint) {
        return
      }

      const memberSet = new Set(point.member_node_ids)
      const matchedSet = new Set(matchedPoint.member_node_ids)
      const unionSize = new Set([...memberSet, ...matchedSet]).size
      const overlapRatio = unionSize > 0 ? bestOverlap / unionSize : 0

      items.push({
        point_id: point.id,
        point_label: point.label,
        matched_point_id: matchedPoint.id,
        matched_point_label: matchedPoint.label,
        drift: 1 - overlapRatio,
        overlap_ratio: overlapRatio,
        overlap_count: bestOverlap,
        anchor_node_id: point.anchor_node_id,
      })
    })

    return items.sort((left, right) => right.drift - left.drift)
  }, [comparisonPayload, payload, points])

  const driftByPointId = useMemo(() => {
    const map = new Map<string, number>()
    diffItems.forEach((item) => {
      map.set(item.point_id, item.drift)
    })
    return map
  }, [diffItems])

  const plotPoints = useMemo(() => {
    const width = 980
    const height = 560
    const padding = 34
    const maxMembers = Math.max(...points.map((point) => point.member_count), 1)

    return points.map((point) => {
      const nx = Number.isFinite(point.x) ? point.x : 0
      const ny = Number.isFinite(point.y) ? point.y : 0
      const cx = ((nx + 1) / 2) * (width - padding * 2) + padding
      const cy = ((1 - (ny + 1) / 2) * (height - padding * 2)) + padding
      const radius = 6 + Math.sqrt(point.member_count / maxMembers) * 20
      return {
        ...point,
        cx,
        cy,
        radius,
        selected: isPointSelected(point, selectedNodeId),
        color: colorForDomain(point.dominant_domain),
        driftScore: driftByPointId.get(point.id) || 0,
      }
    })
  }, [driftByPointId, points, selectedNodeId])

  const hasSelection = Boolean(selectedNodeId)
  const flaggedDriftCount = diffItems.filter((item) => item.drift >= diffMinDrift).length

  if (state === 'loading' && !payload) {
    return (
      <div className="cockpit2-skeleton-grid" id="cockpit-panel-semantic_map" role="tabpanel">
        <div className="cockpit2-skeleton-card tall" />
      </div>
    )
  }

  if (state === 'error' && !payload) {
    return (
      <section className="cockpit2-alert error" id="cockpit-panel-semantic_map" role="tabpanel">
        <h3>Semantic map request failed</h3>
        <p>{error}</p>
        <button type="button" onClick={onRetry}>Retry</button>
      </section>
    )
  }

  const unavailable = payload?.status.status === 'unavailable'

  return (
    <section className="cockpit2-panel cockpit2-semantic-panel" id="cockpit-panel-semantic_map" role="tabpanel">
      {error ? (
        <div className="cockpit2-alert error inline">
          <p>{error}</p>
          <button type="button" onClick={onRetry}>Retry</button>
        </div>
      ) : null}

      <div className="cockpit2-panel-header-row">
        <h3>Semantic Map</h3>
        <p className="muted">
          Mode: {mode === 'semantic' ? 'Semantic communities' : 'Code-structure communities'} • Points: {payload?.summary.points || 0}
        </p>
      </div>

      <div className="cockpit2-graph-toolbar">
        <label>
          Mode
          <select value={mode} onChange={(event) => onModeChange(event.target.value as SemanticMapMode)}>
            <option value="code_structure">Code structure</option>
            <option value="semantic">Semantic</option>
          </select>
        </label>
      </div>

      {showDiffOverlay && !comparisonPayload ? (
        <div className="cockpit2-alert inline">
          <p>Diff overlay unavailable because comparison mode data could not be loaded.</p>
        </div>
      ) : null}

      {payload?.warnings?.length ? (
        <div className="cockpit2-alert inline">
          {payload.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}

      {unavailable ? (
        <section className="cockpit2-empty">
          <h3>No semantic map available</h3>
          <p>
            {payload?.status.reason === 'no_semantic_communities'
              ? 'No semantic communities were found. Run GraphRAG semantic extraction and community detection first.'
              : 'No code-structure communities were found in the current selection.'}
          </p>
        </section>
      ) : (
        <>
          <div className="cockpit2-semantic-canvas-wrap">
            <svg
              className="cockpit2-semantic-canvas"
              viewBox="0 0 980 560"
              role="img"
              aria-label="Semantic map scatter plot"
            >
              {plotPoints.map((point) => {
                const dimmed = hasSelection && !point.selected
                const showDriftHalo = showDiffOverlay && point.driftScore >= diffMinDrift
                return (
                  <g key={point.id}>
                    {showDriftHalo ? (
                      <circle
                        cx={point.cx}
                        cy={point.cy}
                        r={point.radius + 6 + point.driftScore * 6}
                        className={[dimmed ? 'dimmed' : '', 'cockpit2-semantic-diff-halo'].join(' ').trim()}
                      />
                    ) : null}
                    <circle
                      cx={point.cx}
                      cy={point.cy}
                      r={point.radius}
                      className={[
                        'cockpit2-semantic-point',
                        point.selected ? 'selected' : '',
                        dimmed ? 'dimmed' : '',
                      ].join(' ').trim()}
                      fill={point.color}
                      onClick={() => onSelectNodeId(point.anchor_node_id)}
                    >
                      <title>
                        {point.label}\nMembers: {point.member_count}\nDominant domain: {point.dominant_domain || 'n/a'}
                        {showDriftHalo ? `\nDrift: ${point.driftScore.toFixed(2)}` : ''}
                      </title>
                    </circle>
                    {plotPoints.length <= 80 ? (
                      <text
                        x={point.cx + point.radius + 3}
                        y={point.cy + 3}
                        className={dimmed ? 'dimmed' : ''}
                      >
                        {point.label}
                      </text>
                    ) : null}
                  </g>
                )
              })}
            </svg>
          </div>

          <div className="cockpit2-arch-kpis">
            <div>
              <strong>{payload?.summary.mixed_clusters || 0}</strong>
              <span>Mixed clusters</span>
            </div>
            <div>
              <strong>{payload?.summary.isolated_points || 0}</strong>
              <span>Isolated points</span>
            </div>
            <div>
              <strong>{payload?.summary.semantic_duplication || 0}</strong>
              <span>Duplication hints</span>
            </div>
            <div>
              <strong>{payload?.summary.misplaced_code || 0}</strong>
              <span>Misplaced code</span>
            </div>
            {showDiffOverlay ? (
              <div>
                <strong>{flaggedDriftCount}</strong>
                <span>Drift overlay hits</span>
              </div>
            ) : null}
          </div>

          <div className="cockpit2-semantic-signal-grid">
            {renderSignals('Mixed clusters', payload?.signals.mixed_clusters || [], onSelectNodeId)}
            {renderSignals('Isolated points', payload?.signals.isolated_points || [], onSelectNodeId)}
            {renderSignals('Semantic duplication', payload?.signals.semantic_duplication || [], onSelectNodeId)}
            {renderSignals('Misplaced code', payload?.signals.misplaced_code || [], onSelectNodeId)}
            {showDiffOverlay
              ? renderDiffSignals(diffItems, { threshold: diffMinDrift, onSelectNodeId })
              : null}
          </div>
        </>
      )}
    </section>
  )
}
