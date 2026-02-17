import type {
  CityEntityLevel,
  CityProjection,
  CockpitLoadState,
} from '../types'

interface CityViewProps {
  state: CockpitLoadState
  error: string
  embedUrl: string
  projection: CityProjection
  entityLevel: CityEntityLevel
  onProjectionChange: (projection: CityProjection) => void
  onEntityLevelChange: (entityLevel: CityEntityLevel) => void
  onReload: () => void
}

export default function CityView({
  state,
  error,
  embedUrl,
  projection,
  entityLevel,
  onProjectionChange,
  onEntityLevelChange,
  onReload,
}: CityViewProps) {
  return (
    <section className="cockpit2-panel cockpit2-city-panel" id="cockpit-panel-city" role="tabpanel">
      <div className="cockpit2-panel-header-row">
        <h3>Inline Code City</h3>
        <p className="muted">Explore architecture and file-level maps with CodeCharta.</p>
      </div>

      <div className="cockpit2-export-toolbar">
        <label>
          <span>Projection</span>
          <select
            value={projection}
            onChange={(event) => onProjectionChange(event.target.value as CityProjection)}
          >
            <option value="architecture">Architecture</option>
            <option value="code_file">Code file</option>
          </select>
        </label>
        <label>
          <span>Entity level</span>
          <select
            value={entityLevel}
            disabled={projection !== 'architecture'}
            onChange={(event) => onEntityLevelChange(event.target.value as CityEntityLevel)}
          >
            <option value="domain">Domain</option>
            <option value="container">Container</option>
            <option value="component">Component</option>
          </select>
        </label>
        <button type="button" onClick={onReload} disabled={state === 'loading'}>
          {state === 'loading' ? 'Loading…' : 'Reload map'}
        </button>
      </div>

      {error ? (
        <div className="cockpit2-alert error inline">
          <p>{error}</p>
        </div>
      ) : null}

      {state === 'loading' && !embedUrl ? (
        <div className="cockpit2-skeleton-grid">
          <div className="cockpit2-skeleton-card" />
          <div className="cockpit2-skeleton-card" />
        </div>
      ) : null}

      {embedUrl ? (
        <div className="cockpit2-city-frame-wrap">
          <iframe
            title="CodeCharta city view"
            src={embedUrl}
            className="cockpit2-city-frame"
          />
        </div>
      ) : state !== 'loading' ? (
        <div className="cockpit2-empty">
          <h3>No city view available yet</h3>
          <p>Generate the map from the selected scenario.</p>
        </div>
      ) : null}
    </section>
  )
}
