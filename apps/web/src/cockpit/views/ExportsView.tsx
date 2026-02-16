import { EXPORT_FORMATS, type CockpitLoadState, type ExportFormat } from '../types'

interface ExportsViewProps {
  exportFormat: ExportFormat
  exportState: CockpitLoadState
  exportError: string
  exportContent: string
  onFormatChange: (format: ExportFormat) => void
  onGenerate: () => void
  onCopy: () => void
  onDownload: () => void
}

export default function ExportsView({
  exportFormat,
  exportState,
  exportError,
  exportContent,
  onFormatChange,
  onGenerate,
  onCopy,
  onDownload,
}: ExportsViewProps) {
  return (
    <section className="cockpit2-panel" id="cockpit-panel-exports" role="tabpanel">
      <div className="cockpit2-panel-header-row">
        <h3>Visualization exports</h3>
        <p className="muted">Generate, inspect, and download extracted artifacts.</p>
      </div>

      <div className="cockpit2-export-toolbar">
        <label>
          <span>Format</span>
          <select
            value={exportFormat}
            onChange={(event) => onFormatChange(event.target.value as ExportFormat)}
          >
            {EXPORT_FORMATS.map((format) => (
              <option key={format.key} value={format.key}>
                {format.label}
              </option>
            ))}
          </select>
        </label>

        <button type="button" onClick={onGenerate} disabled={exportState === 'loading'}>
          {exportState === 'loading' ? 'Generating…' : 'Generate export'}
        </button>
        <button type="button" className="secondary" onClick={onCopy} disabled={!exportContent}>
          Copy
        </button>
        <button type="button" className="secondary" onClick={onDownload} disabled={!exportContent}>
          Download
        </button>
      </div>

      {exportError ? (
        <div className="cockpit2-alert error inline">
          <p>{exportError}</p>
        </div>
      ) : null}

      <div className="cockpit2-export-output">
        <pre>{exportContent || '// Generate an export to preview its content.'}</pre>
      </div>
    </section>
  )
}
