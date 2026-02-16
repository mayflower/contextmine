import { COCKPIT_VIEWS, type CockpitView } from '../types'

interface CockpitTabsProps {
  activeView: CockpitView
  onViewChange: (view: CockpitView) => void
}

export default function CockpitTabs({ activeView, onViewChange }: CockpitTabsProps) {
  return (
    <div className="cockpit2-tabs" role="tablist" aria-label="Architecture Cockpit views">
      {COCKPIT_VIEWS.map((tab) => {
        const isActive = tab.key === activeView
        const tabId = `cockpit-tab-${tab.key}`
        const panelId = `cockpit-panel-${tab.key}`

        return (
          <button
            key={tab.key}
            id={tabId}
            role="tab"
            type="button"
            aria-controls={panelId}
            aria-selected={isActive}
            className={`cockpit2-tab ${isActive ? 'active' : ''}`}
            onClick={() => onViewChange(tab.key)}
          >
            {tab.label}
          </button>
        )
      })}
    </div>
  )
}
