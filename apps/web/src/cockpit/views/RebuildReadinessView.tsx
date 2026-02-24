import type { CockpitLoadState, RebuildReadinessPayload } from '../types'

interface RebuildReadinessViewProps {
  state: CockpitLoadState
  error: string
  payload: RebuildReadinessPayload | null
  onRetry: () => void
}

function percentage(value: number): string {
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`
}

export default function RebuildReadinessView({
  state,
  error,
  payload,
  onRetry,
}: RebuildReadinessViewProps) {
  if (state === 'loading' && !payload) {
    return (
      <div className="cockpit2-skeleton-grid" id="cockpit-panel-rebuild_readiness" role="tabpanel">
        <div className="cockpit2-skeleton-card" />
        <div className="cockpit2-skeleton-card" />
      </div>
    )
  }

  if (state === 'error' && !payload) {
    return (
      <section className="cockpit2-alert error" id="cockpit-panel-rebuild_readiness" role="tabpanel">
        <h3>Readiness request failed</h3>
        <p>{error}</p>
        <button type="button" onClick={onRetry}>Retry</button>
      </section>
    )
  }

  return (
    <section className="cockpit2-panel" id="cockpit-panel-rebuild_readiness" role="tabpanel">
      {error ? (
        <div className="cockpit2-alert error inline">
          <p>{error}</p>
          <button type="button" onClick={onRetry}>Retry</button>
        </div>
      ) : null}

      <div className="cockpit2-panel-header-row">
        <h3>Rebuild readiness</h3>
        <p className="muted">
          Deep stage: {payload?.behavioral_layers_status || 'pending'}
          {payload?.last_behavioral_materialized_at
            ? ` • ${new Date(payload.last_behavioral_materialized_at).toLocaleString()}`
            : ''}
        </p>
      </div>

      <p className="muted">
        SCIP: {payload?.scip_status || 'failed'} • Metrics gate: {payload?.metrics_gate?.status || 'pass'}
        {typeof payload?.metrics_gate?.mapped_files === 'number' &&
        typeof payload?.metrics_gate?.requested_files === 'number'
          ? ` (${payload.metrics_gate.mapped_files}/${payload.metrics_gate.requested_files})`
          : ''}
      </p>

      <div className="cockpit2-arch-kpis">
        <div>
          <strong>{payload?.score ?? 0}</strong>
          <span>Overall score</span>
        </div>
        <div>
          <strong>{percentage(payload?.summary.interface_test_coverage ?? 0)}</strong>
          <span>Interface test coverage</span>
        </div>
        <div>
          <strong>{percentage(payload?.summary.ui_to_endpoint_traceability ?? 0)}</strong>
          <span>UI traceability</span>
        </div>
        <div>
          <strong>{payload?.summary.critical_inferred_only_count ?? 0}</strong>
          <span>Critical inferred-only nodes</span>
        </div>
      </div>

      {(payload?.known_gaps || []).length > 0 ? (
        <article className="cockpit2-architecture-card">
          <h4>Known gaps</h4>
          <ul>
            {(payload?.known_gaps || []).map((gap) => (
              <li key={gap}>{gap}</li>
            ))}
          </ul>
        </article>
      ) : null}

      {(payload?.deep_warnings || []).length > 0 ? (
        <article className="cockpit2-architecture-card">
          <h4>Deep warnings</h4>
          <ul>
            {(payload?.deep_warnings || []).map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </article>
      ) : null}

      {(payload?.scip_failed_projects || []).length > 0 ? (
        <article className="cockpit2-architecture-card">
          <h4>SCIP failed projects</h4>
          <ul>
            {(payload?.scip_failed_projects || []).slice(0, 10).map((project, index) => (
              <li key={`${project.project_root || 'project'}-${project.language || 'lang'}-${index}`}>
                [{project.language || 'unknown'}] {project.project_root || 'unknown root'}: {project.error || 'unknown error'}
              </li>
            ))}
          </ul>
        </article>
      ) : null}
    </section>
  )
}
