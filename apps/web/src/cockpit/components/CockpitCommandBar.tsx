import { useMemo, useState } from 'react'

import {
  COCKPIT_LAYERS,
  type C4ViewMode,
  type CockpitLayer,
  type CockpitView,
  type CollectionLite,
  type ScenarioLite,
} from '../types'

interface CockpitCommandBarProps {
  collections: CollectionLite[]
  scenarios: ScenarioLite[]
  collectionId: string
  scenarioId: string
  layer: CockpitLayer
  activeView: CockpitView
  hotspotFilter: string
  graphQuery: string
  hideIsolated: boolean
  graphPage: number
  graphLimit: number
  includeKinds: string[]
  excludeKinds: string[]
  edgeKinds: string[]
  overlayMode: 'none' | 'runtime' | 'risk'
  c4View: C4ViewMode
  c4Scope: string
  c4MaxNodes: number
  availableNodeKinds: string[]
  availableEdgeKinds: string[]
  onCollectionChange: (collectionId: string) => void
  onScenarioChange: (scenarioId: string) => void
  onLayerChange: (layer: CockpitLayer) => void
  onFilterChange: (value: string) => void
  onGraphQueryChange: (value: string) => void
  onHideIsolatedChange: (next: boolean) => void
  onGraphPageChange: (value: number) => void
  onGraphLimitChange: (value: number) => void
  onIncludeKindsChange: (value: string[]) => void
  onExcludeKindsChange: (value: string[]) => void
  onEdgeKindsChange: (value: string[]) => void
  onOverlayModeChange: (mode: 'none' | 'runtime' | 'risk') => void
  onC4ViewChange: (value: C4ViewMode) => void
  onC4ScopeChange: (value: string) => void
  onC4MaxNodesChange: (value: number) => void
  onRefresh: () => void
  onLoadOverlayFile: (file: File) => void
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
  graphQuery,
  hideIsolated,
  graphPage,
  graphLimit,
  includeKinds,
  excludeKinds,
  edgeKinds,
  overlayMode,
  c4View,
  c4Scope,
  c4MaxNodes,
  availableNodeKinds,
  availableEdgeKinds,
  onCollectionChange,
  onScenarioChange,
  onLayerChange,
  onFilterChange,
  onGraphQueryChange,
  onHideIsolatedChange,
  onGraphPageChange,
  onGraphLimitChange,
  onIncludeKindsChange,
  onExcludeKindsChange,
  onEdgeKindsChange,
  onOverlayModeChange,
  onC4ViewChange,
  onC4ScopeChange,
  onC4MaxNodesChange,
  onRefresh,
  onLoadOverlayFile,
  onOpenCollections,
  onOpenRuns,
}: CockpitCommandBarProps) {
  const [showAdvancedGraphControls, setShowAdvancedGraphControls] = useState(false)
  const showLayer = activeView === 'topology' || activeView === 'deep_dive'
  const showFilter = activeView === 'overview'
  const showC4Controls = activeView === 'c4_diff'
  const showGraphControls =
    activeView === 'topology' || activeView === 'deep_dive' || activeView === 'graphrag'
  const advancedFilterCount = useMemo(() => {
    let count = 0
    if (includeKinds.length > 0) count += 1
    if (excludeKinds.length > 0) count += 1
    if (edgeKinds.length > 0) count += 1
    if (hideIsolated) count += 1
    if (overlayMode !== 'none') count += 1
    return count
  }, [edgeKinds.length, excludeKinds.length, hideIsolated, includeKinds.length, overlayMode])

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
        ) : showC4Controls ? (
          <label>
            <span>C4 view</span>
            <select value={c4View} onChange={(event) => onC4ViewChange(event.target.value as C4ViewMode)}>
              <option value="container">Container</option>
              <option value="component">Component</option>
              <option value="code">Code</option>
              <option value="context">Context</option>
              <option value="deployment">Deployment</option>
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
        ) : showGraphControls ? (
          <label>
            <span>Graph search</span>
            <input
              type="search"
              placeholder="name, kind, path..."
              value={graphQuery}
              onChange={(event) => onGraphQueryChange(event.target.value)}
            />
          </label>
        ) : showC4Controls ? (
          <label>
            <span>C4 scope</span>
            <input
              type="search"
              placeholder="container/component/file..."
              value={c4Scope}
              onChange={(event) => onC4ScopeChange(event.target.value)}
            />
          </label>
        ) : (
          <div className="cockpit2-command-placeholder" />
        )}
      </div>

      {showC4Controls ? (
        <div className="cockpit2-command-grid">
          <label>
            <span>Max nodes</span>
            <input
              type="number"
              min={10}
              max={5000}
              step={10}
              value={c4MaxNodes}
              onChange={(event) => onC4MaxNodesChange(Number(event.target.value))}
            />
          </label>
        </div>
      ) : null}

      {showGraphControls ? (
        <details
          className="cockpit2-advanced"
          open={showAdvancedGraphControls}
          onToggle={(event) => setShowAdvancedGraphControls(event.currentTarget.open)}
        >
          <summary>
            Advanced graph controls
            {advancedFilterCount > 0 ? ` (${advancedFilterCount} active)` : ''}
          </summary>
          <div className="cockpit2-command-grid">
            <label>
              <span>Page</span>
              <input
                type="number"
                min={0}
                value={graphPage}
                onChange={(event) => onGraphPageChange(Number(event.target.value))}
              />
            </label>
            <label>
              <span>Limit</span>
              <input
                type="number"
                min={1}
                max={10000}
                step={1}
                value={graphLimit}
                onChange={(event) => onGraphLimitChange(Number(event.target.value))}
              />
            </label>
            <label>
              <span>Overlay</span>
              <select
                value={overlayMode}
                onChange={(event) => onOverlayModeChange(event.target.value as 'none' | 'runtime' | 'risk')}
              >
                <option value="none">None</option>
                <option value="runtime">Runtime</option>
                <option value="risk">Dependency risk</option>
              </select>
            </label>
            <label className="cockpit2-upload-label">
              <span>Overlay file (CSV/JSON)</span>
              <input
                type="file"
                accept=".csv,.json,.txt"
                onChange={(event) => {
                  const file = event.target.files?.[0]
                  if (!file) return
                  onLoadOverlayFile(file)
                  event.currentTarget.value = ''
                }}
              />
            </label>
          </div>
          <div className="cockpit2-command-grid">
            <label>
              <span>Include kinds</span>
              <select
                multiple
                value={includeKinds}
                onChange={(event) => onIncludeKindsChange(Array.from(event.target.selectedOptions).map((entry) => entry.value))}
              >
                {availableNodeKinds.map((kind) => (
                  <option key={kind} value={kind}>{kind}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Exclude kinds</span>
              <select
                multiple
                value={excludeKinds}
                onChange={(event) => onExcludeKindsChange(Array.from(event.target.selectedOptions).map((entry) => entry.value))}
              >
                {availableNodeKinds.map((kind) => (
                  <option key={kind} value={kind}>{kind}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Edge kind chips</span>
              <select
                multiple
                value={edgeKinds}
                onChange={(event) => onEdgeKindsChange(Array.from(event.target.selectedOptions).map((entry) => entry.value))}
              >
                {availableEdgeKinds.map((kind) => (
                  <option key={kind} value={kind}>{kind}</option>
                ))}
              </select>
            </label>
            <label className="cockpit2-checkbox">
              <input
                type="checkbox"
                checked={hideIsolated}
                onChange={(event) => onHideIsolatedChange(event.target.checked)}
              />
              <span>Hide isolated nodes</span>
            </label>
          </div>
        </details>
      ) : null}

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
