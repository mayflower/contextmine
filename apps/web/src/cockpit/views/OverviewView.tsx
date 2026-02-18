import { useMemo, useState } from 'react'

import type { CityPayload, CockpitLoadState } from '../types'

type SortKey = 'node' | 'complexity' | 'coupling' | 'coverage' | 'loc' | 'change_frequency' | 'churn'
type SortDirection = 'asc' | 'desc'

interface OverviewViewProps {
  city: CityPayload | null
  state: CockpitLoadState
  error: string
  filter: string
  onRetry: () => void
  onOpenTopology: () => void
  onSelectHotspot: (nodeNaturalKey: string) => void
}

function levelFromComplexity(complexity: number): 'high' | 'medium' | 'low' {
  if (complexity >= 20) return 'high'
  if (complexity >= 8) return 'medium'
  return 'low'
}

function formatMetricValue(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return 'N/A'
  }
  return value.toFixed(2)
}

function metricsUnavailableMessage(reason: string | undefined): string {
  if (reason === 'awaiting_ci_coverage') {
    return 'Structural metrics are ready. Waiting for CI-pushed coverage reports for this commit.'
  }
  if (reason === 'coverage_ingest_failed') {
    return 'Coverage ingest failed for the latest CI upload. Check ingest job diagnostics and retry.'
  }
  return 'Real metrics are currently unavailable for this scenario.'
}

function sortHotspots(
  city: CityPayload | null,
  sortKey: SortKey,
  sortDirection: SortDirection,
  filter: string,
) {
  const hotspots = city?.hotspots || []
  const normalizedFilter = filter.trim().toLowerCase()

  const filtered = normalizedFilter
    ? hotspots.filter((item) => item.node_natural_key.toLowerCase().includes(normalizedFilter))
    : hotspots

  const sorted = [...filtered].sort((a, b) => {
    if (sortKey === 'node') {
      return a.node_natural_key.localeCompare(b.node_natural_key)
    }

    if (sortKey === 'loc') {
      return a.loc - b.loc
    }

    return (a[sortKey] || 0) - (b[sortKey] || 0)
  })

  return sortDirection === 'asc' ? sorted : sorted.reverse()
}

export default function OverviewView({
  city,
  state,
  error,
  filter,
  onRetry,
  onOpenTopology,
  onSelectHotspot,
}: OverviewViewProps) {
  const [sortKey, setSortKey] = useState<SortKey>('complexity')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')

  const hotspots = useMemo(
    () => sortHotspots(city, sortKey, sortDirection, filter),
    [city, sortKey, sortDirection, filter],
  )

  const handleSort = (nextKey: SortKey) => {
    if (nextKey === sortKey) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'))
      return
    }
    setSortKey(nextKey)
    setSortDirection(nextKey === 'node' ? 'asc' : 'desc')
  }

  if (state === 'loading' && !city) {
    return (
      <div className="cockpit2-skeleton-grid" id="cockpit-panel-overview" role="tabpanel">
        <div className="cockpit2-skeleton-card" />
        <div className="cockpit2-skeleton-card" />
        <div className="cockpit2-skeleton-card" />
        <div className="cockpit2-skeleton-card" />
      </div>
    )
  }

  if ((state === 'empty' || !city) && !error) {
    return (
      <section className="cockpit2-empty" id="cockpit-panel-overview" role="tabpanel">
        <h3>No city data available yet</h3>
        <p>Run a source sync first, then reload this view to generate metric snapshots and hotspots.</p>
        <button type="button" onClick={onRetry}>Retry</button>
      </section>
    )
  }

  if (state === 'error' && !city) {
    return (
      <section className="cockpit2-alert error" id="cockpit-panel-overview" role="tabpanel">
        <h3>Overview request failed</h3>
        <p>{error}</p>
        <button type="button" onClick={onRetry}>Retry</button>
      </section>
    )
  }

  const metricsUnavailable = city?.metrics_status?.status === 'unavailable'
  const unavailableReason = city?.metrics_status?.reason

  return (
    <section className="cockpit2-workspace" id="cockpit-panel-overview" role="tabpanel">
      {error ? (
        <div className="cockpit2-alert error inline">
          <p>{error}</p>
          <button type="button" onClick={onRetry}>Retry</button>
        </div>
      ) : null}

      <div className="cockpit2-main">
        <article className="cockpit2-panel">
          <h3>System health summary</h3>
          {metricsUnavailable ? (
            <>
              <p className="muted">
                {metricsUnavailableMessage(unavailableReason)}
              </p>
              <div className="actions">
                <button type="button" className="secondary" onClick={onOpenTopology}>
                  Open Topology (Code / Controlflow)
                </button>
              </div>
            </>
          ) : null}
          <div className="cockpit2-kpis">
            <div>
              <strong>{city?.summary.metric_nodes ?? 0}</strong>
              <span>Metric nodes</span>
            </div>
            <div>
              <strong>{formatMetricValue(city?.summary.coverage_avg)}</strong>
              <span>Average coverage</span>
            </div>
            <div>
              <strong>{formatMetricValue(city?.summary.complexity_avg)}</strong>
              <span>Average complexity</span>
            </div>
            <div>
              <strong>{formatMetricValue(city?.summary.coupling_avg)}</strong>
              <span>Average coupling</span>
            </div>
            <div>
              <strong>{formatMetricValue(city?.summary.change_frequency_avg)}</strong>
              <span>Average change frequency</span>
            </div>
            <div>
              <strong>{formatMetricValue(city?.summary.churn_avg)}</strong>
              <span>Average churn</span>
            </div>
          </div>
        </article>

        <article className="cockpit2-panel">
          <h3>Top hotspots</h3>
          <p className="muted">Sorted and color-coded by complexity risk.</p>
          <div className="cockpit2-table-wrap">
            <table className="cockpit2-table">
              <thead>
                <tr>
                  <th><button type="button" onClick={() => handleSort('node')}>Node</button></th>
                  <th><button type="button" onClick={() => handleSort('complexity')}>Complexity</button></th>
                  <th><button type="button" onClick={() => handleSort('coupling')}>Coupling</button></th>
                  <th><button type="button" onClick={() => handleSort('coverage')}>Coverage</button></th>
                  <th><button type="button" onClick={() => handleSort('loc')}>LOC</button></th>
                  <th><button type="button" onClick={() => handleSort('change_frequency')}>Change frequency</button></th>
                  <th><button type="button" onClick={() => handleSort('churn')}>Churn</button></th>
                </tr>
              </thead>
              <tbody>
                {hotspots.slice(0, 20).map((spot) => (
                  <tr key={spot.node_natural_key} className={`risk-${levelFromComplexity(spot.complexity)}`}>
                    <td>
                      <button
                        type="button"
                        className="cockpit2-linkish"
                        onClick={() => onSelectHotspot(spot.node_natural_key)}
                      >
                        {spot.node_natural_key}
                      </button>
                    </td>
                    <td>{(spot.complexity || 0).toFixed(2)}</td>
                    <td>{(spot.coupling || 0).toFixed(2)}</td>
                    <td>{(spot.coverage || 0).toFixed(2)}</td>
                    <td>{spot.loc}</td>
                    <td>{(spot.change_frequency || 0).toFixed(2)}</td>
                    <td>{(spot.churn || 0).toFixed(2)}</td>
                  </tr>
                ))}
                {hotspots.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="empty-row">
                      {metricsUnavailable
                        ? metricsUnavailableMessage(unavailableReason)
                        : 'No hotspots match the current filter.'}
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </article>
      </div>
    </section>
  )
}
