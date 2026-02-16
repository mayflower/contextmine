import { COCKPIT_LAYERS, type CockpitLayer, type CockpitView, type CollectionLite, type ScenarioLite } from '../types'

interface CockpitCommandBarProps {
  collections: CollectionLite[]
  scenarios: ScenarioLite[]
  collectionId: string
  scenarioId: string
  layer: CockpitLayer
  activeView: CockpitView
  hotspotFilter: string
  onCollectionChange: (collectionId: string) => void
  onScenarioChange: (scenarioId: string) => void
  onLayerChange: (layer: CockpitLayer) => void
  onFilterChange: (value: string) => void
  onRefresh: () => void
  onOpenCollections: () => void
  onOpenRuns: () => void
}

export default function CockpitCommandBar({
  collections,
  scenarios,
  collectionId,
  scenarioId,
  layer,
  activeView,
  hotspotFilter,
  onCollectionChange,
  onScenarioChange,
  onLayerChange,
  onFilterChange,
  onRefresh,
  onOpenCollections,
  onOpenRuns,
}: CockpitCommandBarProps) {
  const showLayer = activeView === 'topology' || activeView === 'deep_dive'
  const showFilter = activeView === 'overview'

  return (
    <section className="cockpit2-commandbar" aria-label="Cockpit controls">
      <div className="cockpit2-command-grid">
        <label>
          <span>Project</span>
          <select value={collectionId} onChange={(event) => onCollectionChange(event.target.value)}>
            <option value="">Select project...</option>
            {collections.map((collection) => (
              <option key={collection.id} value={collection.id}>
                {collection.name}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Scenario</span>
          <select value={scenarioId} onChange={(event) => onScenarioChange(event.target.value)}>
            <option value="">Select scenario...</option>
            {scenarios.map((scenario) => (
              <option key={scenario.id} value={scenario.id}>
                {scenario.name} (v{scenario.version}) {scenario.is_as_is ? '• AS-IS' : '• TO-BE'}
              </option>
            ))}
          </select>
        </label>

        {showLayer ? (
          <label>
            <span>Layer</span>
            <select value={layer} onChange={(event) => onLayerChange(event.target.value as CockpitLayer)}>
              {COCKPIT_LAYERS.map((entry) => (
                <option key={entry.key} value={entry.key}>
                  {entry.label}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <div className="cockpit2-command-placeholder" />
        )}

        {showFilter ? (
          <label>
            <span>Hotspot filter</span>
            <input
              type="search"
              placeholder="Filter nodes..."
              value={hotspotFilter}
              onChange={(event) => onFilterChange(event.target.value)}
            />
          </label>
        ) : (
          <div className="cockpit2-command-placeholder" />
        )}
      </div>

      <div className="cockpit2-command-actions">
        <button type="button" onClick={onRefresh} className="secondary">
          Refresh view
        </button>
        <button type="button" onClick={onOpenCollections} className="ghost">
          Open Collections
        </button>
        <button type="button" onClick={onOpenRuns} className="ghost">
          Open Runs
        </button>
      </div>
    </section>
  )
}
