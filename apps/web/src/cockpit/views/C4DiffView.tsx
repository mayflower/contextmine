import { useRef } from 'react'

import type { CockpitLoadState, MermaidPayload } from '../types'

interface C4DiffViewProps {
  mermaid: MermaidPayload | null
  state: CockpitLoadState
  error: string
  onRetry: () => void
}

export default function C4DiffView({ mermaid, state, error, onRetry }: C4DiffViewProps) {
  const leftRef = useRef<HTMLPreElement | null>(null)
  const rightRef = useRef<HTMLPreElement | null>(null)
  const syncLock = useRef(false)

  const syncScroll = (source: HTMLPreElement | null, target: HTMLPreElement | null) => {
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

      <div className="cockpit2-panel-header-row">
        <h3>Mermaid C4 diff</h3>
        <p className="muted">AS-IS and TO-BE are synchronized for easier comparison.</p>
      </div>

      {mermaid?.mode === 'compare' ? (
        <div className="cockpit2-compare-grid">
          <article>
            <header>
              <h4>AS-IS</h4>
              <span className="badge asis">Baseline</span>
            </header>
            <pre
              ref={leftRef}
              onScroll={() => syncScroll(leftRef.current, rightRef.current)}
            >
              {mermaid.as_is || ''}
            </pre>
          </article>

          <article>
            <header>
              <h4>TO-BE</h4>
              <span className="badge tobe">Target</span>
            </header>
            <pre
              ref={rightRef}
              onScroll={() => syncScroll(rightRef.current, leftRef.current)}
            >
              {mermaid.to_be || ''}
            </pre>
          </article>
        </div>
      ) : (
        <article className="cockpit2-panel">
          <pre>{mermaid?.content || ''}</pre>
        </article>
      )}
    </section>
  )
}
