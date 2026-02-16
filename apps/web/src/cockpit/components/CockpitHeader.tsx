import type { CockpitLoadState, CockpitSelection, ScenarioLite } from '../types'

interface CockpitHeaderProps {
  selection: CockpitSelection
  scenario: ScenarioLite | null
  activeState: CockpitLoadState
  activeUpdatedAt: string | null
}

function formatFreshness(timestamp: string | null): string {
  if (!timestamp) {
    return 'Not loaded yet'
  }

  const date = new Date(timestamp)
  return `${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
}

function formatStatus(state: CockpitLoadState): string {
  if (state === 'loading') return 'Loading'
  if (state === 'ready') return 'Ready'
  if (state === 'error') return 'Error'
  if (state === 'empty') return 'Empty'
  return 'Idle'
}

export default function CockpitHeader({ selection, scenario, activeState, activeUpdatedAt }: CockpitHeaderProps) {
  const scenarioKind = scenario?.is_as_is ? 'AS-IS' : 'TO-BE'

  return (
    <header className="cockpit2-header">
      <div>
        <h2>Architecture Cockpit</h2>
        <p>Explore extracted architecture views with project and scenario context.</p>
      </div>

      <div className="cockpit2-chips" aria-label="Cockpit metadata">
        <span className="cockpit2-chip">View: {selection.view.replace('_', ' ')}</span>
        <span className="cockpit2-chip">Scenario: {scenario ? scenario.name : 'Not selected'}</span>
        {scenario && <span className="cockpit2-chip">Version: v{scenario.version}</span>}
        {scenario && <span className="cockpit2-chip">Mode: {scenarioKind}</span>}
        <span className={`cockpit2-chip state-${activeState}`}>Status: {formatStatus(activeState)}</span>
        <span className="cockpit2-chip">Updated: {formatFreshness(activeUpdatedAt)}</span>
      </div>
    </header>
  )
}
