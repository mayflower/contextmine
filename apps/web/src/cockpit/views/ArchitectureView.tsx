import mermaidLib from 'mermaid'
import { useEffect, useMemo, useRef, useState } from 'react'

import type {
  Arc42DriftDelta,
  Arc42DriftPayload,
  Arc42ViewPayload,
  CockpitLoadState,
  ErmViewPayload,
  PortAdapterItem,
  PortsAdaptersPayload,
} from '../types'

interface ArchitectureViewProps {
  state: CockpitLoadState
  error: string
  arc42: Arc42ViewPayload | null
  portsAdapters: PortsAdaptersPayload | null
  drift: Arc42DriftPayload | null
  erm: ErmViewPayload | null
  panelErrors: {
    arc42: string
    ports: string
    drift: string
    erm: string
  }
  onRetry: () => void
}

function sectionLabel(sectionKey: string): string {
  const normalized = sectionKey.replace(/^\d+_/, '').replace(/_/g, ' ').trim()
  return normalized
    .split(' ')
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(' ')
}

function shortHash(hash: string | undefined): string {
  if (!hash) return 'n/a'
  if (hash.length <= 14) return hash
  return `${hash.slice(0, 8)}...${hash.slice(-6)}`
}

function toPercentage(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'n/a'
  }
  return `${Math.round(value * 100)}%`
}

function driftTone(delta: Arc42DriftDelta): 'positive' | 'negative' | 'warning' | 'neutral' {
  if (delta.delta_type === 'added' || delta.delta_type === 'new_port') {
    return 'positive'
  }
  if (delta.delta_type === 'removed' || delta.delta_type === 'removed_adapter') {
    return 'negative'
  }
  if (delta.delta_type === 'changed_confidence' || delta.delta_type === 'moved_component') {
    return 'warning'
  }
  return 'neutral'
}

function groupPortsByContainer(items: PortAdapterItem[], direction: 'inbound' | 'outbound') {
  const map = new Map<string, PortAdapterItem[]>()
  for (const item of items) {
    if (item.direction !== direction) continue
    const key = (item.container || 'unassigned').trim() || 'unassigned'
    const existing = map.get(key)
    if (existing) {
      existing.push(item)
    } else {
      map.set(key, [item])
    }
  }

  return Array.from(map.entries())
    .sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]))
    .map(([container, rows]) => ({ container, rows }))
}

async function renderMermaid(container: HTMLElement, id: string, content: string) {
  if (!content.trim()) {
    const pre = document.createElement('pre')
    container.replaceChildren(pre)
    return
  }
  const rendered = await mermaidLib.render(id, content)
  const parsed = new DOMParser().parseFromString(rendered.svg, 'image/svg+xml')
  if (parsed.querySelector('parsererror')) {
    const pre = document.createElement('pre')
    pre.textContent = content
    container.replaceChildren(pre)
    return
  }
  container.replaceChildren(document.importNode(parsed.documentElement, true))
}

export default function ArchitectureView({
  state,
  error,
  arc42,
  portsAdapters,
  drift,
  erm,
  panelErrors,
  onRetry,
}: ArchitectureViewProps) {
  const sectionEntries = useMemo(
    () => Object.entries(arc42?.arc42.sections || {}),
    [arc42],
  )
  const [selectedSection, setSelectedSection] = useState('')
  const [erdRenderError, setErdRenderError] = useState('')
  const erdContainerRef = useRef<HTMLDivElement | null>(null)
  const activeSection = useMemo(() => {
    if (sectionEntries.length === 0) return ''
    const exists = sectionEntries.some(([key]) => key === selectedSection)
    return exists ? selectedSection : sectionEntries[0][0]
  }, [sectionEntries, selectedSection])

  const activeSectionContent = useMemo(() => {
    if (!activeSection) return ''
    const match = sectionEntries.find(([key]) => key === activeSection)
    return match?.[1] || ''
  }, [activeSection, sectionEntries])

  const coveredSections = useMemo(() => {
    const coverage = arc42?.arc42.section_coverage || {}
    const total = Object.keys(coverage).length
    const covered = Object.values(coverage).filter(Boolean).length
    return { covered, total }
  }, [arc42])

  const inboundGroups = useMemo(
    () => groupPortsByContainer(portsAdapters?.items || [], 'inbound'),
    [portsAdapters],
  )
  const outboundGroups = useMemo(
    () => groupPortsByContainer(portsAdapters?.items || [], 'outbound'),
    [portsAdapters],
  )

  const warnings = useMemo(() => {
    const items = [
      ...(arc42?.warnings || []),
      ...(arc42?.arc42?.warnings || []),
      ...(portsAdapters?.warnings || []),
      ...(drift?.warnings || []),
      ...(erm?.warnings || []),
    ]
    return Array.from(new Set(items.filter(Boolean)))
  }, [arc42, portsAdapters, drift, erm])

  useEffect(() => {
    mermaidLib.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'loose' })
  }, [])

  useEffect(() => {
    const content = erm?.mermaid?.content || ''
    if (!content || !erdContainerRef.current) {
      return
    }
    const run = async () => {
      setErdRenderError('')
      try {
        await renderMermaid(erdContainerRef.current!, 'cockpit-erd-diagram', content)
      } catch (err) {
        setErdRenderError(err instanceof Error ? err.message : 'ERD render failed')
      }
    }
    run()
  }, [erm?.mermaid?.content])

  if (state === 'loading' && !arc42 && !portsAdapters && !drift && !erm) {
    return (
      <div className="cockpit2-skeleton-grid" id="cockpit-panel-architecture" role="tabpanel">
        <div className="cockpit2-skeleton-card" />
        <div className="cockpit2-skeleton-card" />
        <div className="cockpit2-skeleton-card tall" />
      </div>
    )
  }

  if (state === 'error' && !arc42 && !portsAdapters && !drift && !erm) {
    return (
      <section className="cockpit2-alert error" id="cockpit-panel-architecture" role="tabpanel">
        <h3>Architecture view failed</h3>
        <p>{error || 'Could not load architecture intelligence payloads.'}</p>
        <button type="button" onClick={onRetry}>Retry</button>
      </section>
    )
  }

  return (
    <section className="cockpit2-panel cockpit2-architecture-panel" id="cockpit-panel-architecture" role="tabpanel">
      {error ? (
        <div className="cockpit2-alert error inline">
          <p>{error}</p>
          <button type="button" onClick={onRetry}>Retry</button>
        </div>
      ) : null}

      <div className="cockpit2-panel-header-row">
        <h3>Architecture intelligence</h3>
        <p className="muted">
          arc42 + Ports/Adapters + ERM/ERD + Drift report ({drift?.summary.total ?? 0} deltas)
        </p>
      </div>

      <div className="cockpit2-arch-kpis">
        <div>
          <strong>{coveredSections.covered}/{coveredSections.total || 0}</strong>
          <span>arc42 section coverage</span>
        </div>
        <div>
          <strong>{shortHash(arc42?.facts_hash)}</strong>
          <span>Facts hash</span>
        </div>
        <div>
          <strong>{portsAdapters?.summary.total ?? 0}</strong>
          <span>Ports (in: {portsAdapters?.summary.inbound ?? 0}, out: {portsAdapters?.summary.outbound ?? 0})</span>
        </div>
        <div>
          <strong>{erm?.summary.tables ?? 0}</strong>
          <span>ERM tables ({erm?.summary.foreign_keys ?? 0} FK)</span>
        </div>
      </div>

      <div className="cockpit2-architecture-grid">
        <article className="cockpit2-architecture-card">
          <div className="cockpit2-panel-header-row">
            <h4>arc42 sections</h4>
            <p className="muted">Generated: {arc42?.arc42.generated_at ? new Date(arc42.arc42.generated_at).toLocaleString() : 'n/a'}</p>
          </div>

          {panelErrors.arc42 ? (
            <div className="cockpit2-alert error inline">
              <p>{panelErrors.arc42}</p>
              <button type="button" className="secondary" onClick={onRetry}>Retry</button>
            </div>
          ) : null}

          {sectionEntries.length === 0 ? (
            <div className="cockpit2-empty">
              <h3>No arc42 sections available</h3>
              <p>Regenerate the architecture view after twin and knowledge graph extraction.</p>
            </div>
          ) : (
            <div className="cockpit2-arc42-layout">
              <nav className="cockpit2-arc42-nav" aria-label="arc42 sections">
                {sectionEntries.map(([key, value]) => (
                  <button
                    key={key}
                    type="button"
                    className={`cockpit2-arc42-nav-item ${key === activeSection ? 'active' : ''}`}
                    onClick={() => setSelectedSection(key)}
                  >
                    <span>{sectionLabel(key)}</span>
                    <small>{value.trim().length > 0 ? 'covered' : 'empty'}</small>
                  </button>
                ))}
              </nav>

              <div className="cockpit2-arc42-content-wrap">
                <h4>{sectionLabel(activeSection || sectionEntries[0][0])}</h4>
                <div className="cockpit2-arc42-content">{activeSectionContent || 'No content available.'}</div>
              </div>
            </div>
          )}
        </article>

        <article className="cockpit2-architecture-card">
          <div className="cockpit2-panel-header-row">
            <h4>Ports & adapters map</h4>
            <p className="muted">Confidence-backed ownership map</p>
          </div>

          {panelErrors.ports ? (
            <div className="cockpit2-alert error inline">
              <p>{panelErrors.ports}</p>
              <button type="button" className="secondary" onClick={onRetry}>Retry</button>
            </div>
          ) : null}

          {portsAdapters && portsAdapters.items.length > 0 ? (
            <div className="cockpit2-ports-lanes">
              <section className="cockpit2-ports-lane inbound">
                <header>
                  <h5>Inbound ports</h5>
                  <span>{portsAdapters.summary.inbound}</span>
                </header>
                {inboundGroups.map((group) => (
                  <div key={`in-${group.container}`} className="cockpit2-port-cluster">
                    <h6>{group.container}</h6>
                    <div className="cockpit2-port-list">
                      {group.rows.map((item) => (
                        <article key={item.fact_id} className="cockpit2-port-card inbound">
                          <div className="cockpit2-port-row">
                            <strong>{item.port_name}</strong>
                            <span>{item.protocol || 'n/a'}</span>
                          </div>
                          <p>
                            {item.adapter_name || 'unmapped adapter'}
                            {' · '}
                            {item.component || 'unmapped component'}
                          </p>
                          <div className="cockpit2-confidence-bar" aria-label="confidence">
                            <span style={{ width: toPercentage(item.confidence) }} />
                            <small>{toPercentage(item.confidence)}</small>
                          </div>
                        </article>
                      ))}
                    </div>
                  </div>
                ))}
              </section>

              <section className="cockpit2-ports-lane outbound">
                <header>
                  <h5>Outbound ports</h5>
                  <span>{portsAdapters.summary.outbound}</span>
                </header>
                {outboundGroups.map((group) => (
                  <div key={`out-${group.container}`} className="cockpit2-port-cluster">
                    <h6>{group.container}</h6>
                    <div className="cockpit2-port-list">
                      {group.rows.map((item) => (
                        <article key={item.fact_id} className="cockpit2-port-card outbound">
                          <div className="cockpit2-port-row">
                            <strong>{item.port_name}</strong>
                            <span>{item.protocol || 'n/a'}</span>
                          </div>
                          <p>
                            {item.adapter_name || 'unmapped adapter'}
                            {' · '}
                            {item.component || 'unmapped component'}
                          </p>
                          <div className="cockpit2-confidence-bar" aria-label="confidence">
                            <span style={{ width: toPercentage(item.confidence) }} />
                            <small>{toPercentage(item.confidence)}</small>
                          </div>
                        </article>
                      ))}
                    </div>
                  </div>
                ))}
              </section>
            </div>
          ) : (
            <div className="cockpit2-empty">
              <h3>No ports/adapters found</h3>
              <p>Surface extraction may still be incomplete for this scenario.</p>
            </div>
          )}
        </article>

        <article className="cockpit2-architecture-card">
          <div className="cockpit2-panel-header-row">
            <h4>ERM data model</h4>
            <p className="muted">
              {erm?.summary.tables ?? 0} tables, {erm?.summary.foreign_keys ?? 0} foreign keys
            </p>
          </div>

          {panelErrors.erm ? (
            <div className="cockpit2-alert error inline">
              <p>{panelErrors.erm}</p>
              <button type="button" className="secondary" onClick={onRetry}>Retry</button>
            </div>
          ) : null}

          {erdRenderError ? (
            <div className="cockpit2-alert error inline">
              <p>{erdRenderError}</p>
            </div>
          ) : null}

          {erm?.mermaid?.content ? (
            <div className="cockpit2-mermaid-pane cockpit2-erd-pane" ref={erdContainerRef} />
          ) : (
            <div className="cockpit2-empty">
              <h3>No ERD diagram available</h3>
              <p>No `MERMAID_ERD` artifact found. Showing table/fk structure below.</p>
            </div>
          )}

          {erm && erm.tables.length > 0 ? (
            <div className="cockpit2-erm-grid">
              {erm.tables.slice(0, 12).map((table) => (
                <article key={table.id} className="cockpit2-erm-table-card">
                  <header>
                    <h5>{table.name}</h5>
                    <span>{table.columns.length} cols</span>
                  </header>
                  <ul>
                    {table.columns.slice(0, 8).map((column) => (
                      <li key={column.id}>
                        <code>{column.name}</code>
                        <small>{column.type || 'unknown'}</small>
                      </li>
                    ))}
                  </ul>
                </article>
              ))}
            </div>
          ) : (
            <div className="cockpit2-empty">
              <h3>No ERM tables found</h3>
              <p>Schema extraction may still be running for this collection.</p>
            </div>
          )}
        </article>
      </div>

      <article className="cockpit2-architecture-card">
        <div className="cockpit2-panel-header-row">
          <h4>Advisory drift report</h4>
          <p className="muted">
            Severity: <strong>{drift?.summary.severity || 'n/a'}</strong>
          </p>
        </div>

        {panelErrors.drift ? (
          <div className="cockpit2-alert error inline">
            <p>{panelErrors.drift}</p>
            <button type="button" className="secondary" onClick={onRetry}>Retry</button>
          </div>
        ) : null}

        {drift && drift.deltas.length > 0 ? (
          <div className="cockpit2-drift-list">
            {drift.deltas.slice(0, 48).map((delta) => (
              <article key={`${delta.delta_type}-${delta.subject}-${delta.detail}`} className={`cockpit2-drift-item tone-${driftTone(delta)}`}>
                <header>
                  <span className="cockpit2-drift-type">{delta.delta_type}</span>
                  <span className="cockpit2-drift-confidence">{toPercentage(delta.confidence)}</span>
                </header>
                <p>{delta.detail}</p>
                <code>{delta.subject}</code>
              </article>
            ))}
          </div>
        ) : (
          <div className="cockpit2-empty">
            <h3>No architecture drift detected</h3>
            <p>Compared scenario snapshots are stable for the extracted architecture facts.</p>
          </div>
        )}
      </article>

      {warnings.length > 0 ? (
        <article className="cockpit2-architecture-card">
          <div className="cockpit2-panel-header-row">
            <h4>Extraction warnings</h4>
            <p className="muted">Review weak signals and rerun extraction when needed.</p>
          </div>
          <div className="cockpit2-chip-row">
            {warnings.map((warning) => (
              <span key={warning} className="cockpit2-warning-chip">{warning}</span>
            ))}
          </div>
        </article>
      ) : null}
    </section>
  )
}
