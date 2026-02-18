import mermaidLib from 'mermaid'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'

import { cockpitFlags } from '../flags'
import type { CockpitLoadState, MermaidPayload } from '../types'

interface C4DiffViewProps {
  mermaid: MermaidPayload | null
  state: CockpitLoadState
  error: string
  onRetry: () => void
}

function extractElementIds(source: string): Set<string> {
  const ids = new Set<string>()
  const regex = /\b(?:Person|System|System_Ext|SystemDb|Container|ContainerDb|Container_Instance|Component|Boundary|System_Boundary|Container_Boundary|Component_Boundary|Deployment_Node|Node|SystemQueue|SystemQueue_Ext)\s*\(\s*([a-zA-Z0-9_:-]+)/g
  let match: RegExpExecArray | null = regex.exec(source)
  while (match) {
    ids.add(match[1])
    match = regex.exec(source)
  }
  return ids
}

function withSemanticClasses(source: string, ids: Set<string>, className: 'added' | 'removed'): string {
  if (!source.trim() || ids.size === 0) return source
  const classDef = className === 'added'
    ? '\nclassDef added fill:#dcfce7,stroke:#166534,stroke-width:2px;\n'
    : '\nclassDef removed fill:#fee2e2,stroke:#991b1b,stroke-width:2px;\n'
  const classLines = [...ids].map((id) => `class ${id} ${className};`).join('\n')
  return `${source}\n${classDef}${classLines}\n`
}

async function renderMermaid(container: HTMLElement, id: string, content: string) {
  if (!content.trim()) {
    container.innerHTML = '<pre></pre>'
    return
  }
  const rendered = await mermaidLib.render(id, content)
  container.innerHTML = rendered.svg
}

export default function C4DiffView({ mermaid, state, error, onRetry }: C4DiffViewProps) {
  const leftRef = useRef<HTMLDivElement | null>(null)
  const rightRef = useRef<HTMLDivElement | null>(null)
  const syncLock = useRef(false)
  const [showSource, setShowSource] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [renderError, setRenderError] = useState('')

  const transformed = useMemo(() => {
    if (mermaid?.mode !== 'compare') {
      return {
        asIs: mermaid?.content || '',
        toBe: '',
      }
    }
    const asIs = mermaid.as_is || ''
    const toBe = mermaid.to_be || ''
    const asIsIds = extractElementIds(asIs)
    const toBeIds = extractElementIds(toBe)
    const removed = new Set([...asIsIds].filter((id) => !toBeIds.has(id)))
    const added = new Set([...toBeIds].filter((id) => !asIsIds.has(id)))
    return {
      asIs: withSemanticClasses(asIs, removed, 'removed'),
      toBe: withSemanticClasses(toBe, added, 'added'),
    }
  }, [mermaid])
  const warnings = mermaid?.warnings || []
  const asIsWarnings = mermaid?.as_is_warnings || []
  const toBeWarnings = mermaid?.to_be_warnings || []

  const syncScroll = (source: HTMLDivElement | null, target: HTMLDivElement | null) => {
    if (!source || !target || syncLock.current) {
      return
    }

    syncLock.current = true
    target.scrollTop = source.scrollTop
    target.scrollLeft = source.scrollLeft
    window.requestAnimationFrame(() => {
      syncLock.current = false
    })
  }

  useEffect(() => {
    if (!cockpitFlags.c4RenderedDiff) return
    mermaidLib.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'loose' })
  }, [])

  useEffect(() => {
    if (!cockpitFlags.c4RenderedDiff || showSource || !mermaid || !leftRef.current) {
      return
    }

    const run = async () => {
      setRenderError('')
      try {
        if (mermaid.mode === 'compare') {
          if (!rightRef.current) return
          await renderMermaid(leftRef.current!, 'cockpit-c4-asis', transformed.asIs)
          await renderMermaid(rightRef.current, 'cockpit-c4-tobe', transformed.toBe)
        } else {
          await renderMermaid(leftRef.current!, 'cockpit-c4-single', transformed.asIs)
        }
      } catch (err) {
        setRenderError(err instanceof Error ? err.message : 'Mermaid render failed')
      }
    }
    run()
  }, [mermaid, showSource, transformed])

  if (state === 'loading' && !mermaid) {
    return (
      <div className="cockpit2-skeleton-grid" id="cockpit-panel-c4_diff" role="tabpanel">
        <div className="cockpit2-skeleton-card tall" />
      </div>
    )
  }

  if (state === 'error' && !mermaid) {
    return (
      <section className="cockpit2-alert error" id="cockpit-panel-c4_diff" role="tabpanel">
        <h3>C4 diagram request failed</h3>
        <p>{error}</p>
        <button type="button" onClick={onRetry}>Retry</button>
      </section>
    )
  }

  return (
    <section className="cockpit2-panel" id="cockpit-panel-c4_diff" role="tabpanel">
      {error ? (
        <div className="cockpit2-alert error inline">
          <p>{error}</p>
          <button type="button" onClick={onRetry}>Retry</button>
        </div>
      ) : null}

      {renderError ? (
        <div className="cockpit2-alert error inline">
          <p>{renderError}</p>
        </div>
      ) : null}

      {warnings.length > 0 ? (
        <div className="cockpit2-alert inline">
          <p>{warnings.join(' ')}</p>
        </div>
      ) : null}

      <div className="cockpit2-panel-header-row">
        <h3>Mermaid C4 diff</h3>
        <p className="muted">
          {cockpitFlags.c4RenderedDiff
            ? 'Rendered AS-IS and TO-BE with semantic add/remove highlighting.'
            : 'AS-IS and TO-BE source compare.'}
        </p>
      </div>

      <div className="cockpit2-graph-toolbar">
        <button
          type="button"
          className="secondary"
          onClick={() => setShowSource((prev) => !prev)}
          disabled={!cockpitFlags.c4RenderedDiff}
        >
          {showSource ? 'Show rendered' : 'Show source'}
        </button>
        <label>
          Zoom
          <input
            type="range"
            min={0.5}
            max={2}
            step={0.1}
            value={zoom}
            onChange={(event) => setZoom(Number(event.target.value))}
          />
        </label>
      </div>

      {mermaid?.mode === 'compare' ? (
        <div className="cockpit2-compare-grid">
          <article>
            <header>
              <h4>AS-IS</h4>
              <span className="badge asis">Baseline</span>
            </header>
            {asIsWarnings.length > 0 ? (
              <p className="muted">{asIsWarnings.join(' ')}</p>
            ) : null}
            {showSource || !cockpitFlags.c4RenderedDiff ? (
              <pre>{transformed.asIs}</pre>
            ) : (
              <div
                ref={leftRef}
                className="cockpit2-mermaid-pane"
                style={{ '--cockpit-diagram-zoom': String(zoom) } as CSSProperties}
                onScroll={() => syncScroll(leftRef.current, rightRef.current)}
              />
            )}
          </article>

          <article>
            <header>
              <h4>TO-BE</h4>
              <span className="badge tobe">Target</span>
            </header>
            {toBeWarnings.length > 0 ? (
              <p className="muted">{toBeWarnings.join(' ')}</p>
            ) : null}
            {showSource || !cockpitFlags.c4RenderedDiff ? (
              <pre>{transformed.toBe}</pre>
            ) : (
              <div
                ref={rightRef}
                className="cockpit2-mermaid-pane"
                style={{ '--cockpit-diagram-zoom': String(zoom) } as CSSProperties}
                onScroll={() => syncScroll(rightRef.current, leftRef.current)}
              />
            )}
          </article>
        </div>
      ) : showSource || !cockpitFlags.c4RenderedDiff ? (
        <article className="cockpit2-panel">
          <pre>{transformed.asIs}</pre>
        </article>
      ) : (
        <article className="cockpit2-panel">
          <div
            ref={leftRef}
            className="cockpit2-mermaid-pane"
            style={{ '--cockpit-diagram-zoom': String(zoom) } as CSSProperties}
          />
        </article>
      )}
    </section>
  )
}
