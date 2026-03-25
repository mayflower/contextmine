import ViewShell from '../components/ViewShell'
import type { CockpitLoadState, TestMatrixPayload } from '../types'

interface TestMatrixViewProps {
  state: CockpitLoadState
  error: string
  payload: TestMatrixPayload | null
  onRetry: () => void
}

export default function TestMatrixView({ state, error, payload, onRetry }: Readonly<TestMatrixViewProps>) {
  const rows = payload?.matrix || []

  return (
    <ViewShell
      state={state}
      error={error || null}
      panelId="cockpit-panel-test_matrix"
      title="Test matrix"
      hasData={Boolean(payload)}
      onRetry={onRetry}
      skeletonCount={1}
      skeletonTall
    >
    <section className="cockpit2-panel" id="cockpit-panel-test_matrix" role="tabpanel">
      <div className="cockpit2-panel-header-row">
        <h3>Test matrix</h3>
        <p className="muted">
          Cases: {payload?.summary.test_cases ?? 0} • Suites: {payload?.summary.test_suites ?? 0}
        </p>
      </div>

      {rows.length === 0 ? (
        <section className="cockpit2-empty">
          <h3>No test matrix rows</h3>
          <p>Behavioral extraction may still be pending for tests.</p>
        </section>
      ) : (
        <div className="cockpit2-table-wrap">
          <table className="cockpit2-table">
            <thead>
              <tr>
                <th>Test case</th>
                <th>Symbols</th>
                <th>Rules</th>
                <th>Flows</th>
                <th>Fixtures</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.test_case_id}>
                  <td>{row.test_case_name}</td>
                  <td>{row.covers_symbols.slice(0, 4).join(', ') || '—'}</td>
                  <td>{row.validates_rules.slice(0, 4).join(', ') || '—'}</td>
                  <td>{row.verifies_flows.slice(0, 4).join(', ') || '—'}</td>
                  <td>{row.fixtures.slice(0, 4).join(', ') || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
    </ViewShell>
  )
}
