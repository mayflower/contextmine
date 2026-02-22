import { useEffect, useMemo, useRef } from 'react'
import mermaid from 'mermaid'

import type { GraphRagProcessDetailPayload } from '../types'

interface GraphRagProcessModalProps {
  detail: GraphRagProcessDetailPayload
  focused: boolean
  onClose: () => void
  onSelectNodeId: (nodeId: string) => void
  onToggleFocus: () => void
}

function toMermaid(detail: GraphRagProcessDetailPayload): string {
  const lines: string[] = ['flowchart TD']
  const nodeLabelById = new Map<string, string>()

  for (const step of detail.steps) {
    const safeId = step.node_id.replace(/[^a-zA-Z0-9_]/g, '_')
    const label = `${step.step}. ${step.node_name}`
      .replace(/"/g, "'")
      .replace(/\n/g, ' ')
      .trim()
    nodeLabelById.set(step.node_id, safeId)
    lines.push(`  ${safeId}["${label}"]`)
  }

  for (const edge of detail.edges) {
    const src = nodeLabelById.get(edge.source_node_id)
    const dst = nodeLabelById.get(edge.target_node_id)
    if (!src || !dst || src === dst) continue
    lines.push(`  ${src} --> ${dst}`)
  }

  return lines.join('\n')
}

export default function GraphRagProcessModal({
  detail,
  focused,
  onClose,
  onSelectNodeId,
  onToggleFocus,
}: GraphRagProcessModalProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mermaidCode = useMemo(() => toMermaid(detail), [detail])

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
    })
  }, [])

  useEffect(() => {
    const render = async () => {
      if (!containerRef.current) return
      try {
        const id = `graphrag-process-${detail.process.id}-${Date.now()}`
        const { svg } = await mermaid.render(id, mermaidCode)
        containerRef.current.innerHTML = svg
      } catch {
        containerRef.current.innerHTML = `<pre>${mermaidCode}</pre>`
      }
    }
    render()
  }, [detail.process.id, mermaidCode])

  return (
    <div className="cockpit2-modal-backdrop" role="dialog" aria-modal="true" aria-label="Process flow">
      <section className="cockpit2-modal">
        <header className="cockpit2-panel-header-row">
          <h3>{detail.process.label}</h3>
          <button type="button" className="ghost" onClick={onClose}>
            Close
          </button>
        </header>

        <p className="muted">
          {detail.process.process_type === 'cross_community' ? 'Cross-community process' : 'Intra-community process'} •{' '}
          Steps: {detail.process.step_count}
        </p>

        <div className="cockpit2-modal-actions">
          <button type="button" className="secondary" onClick={onToggleFocus}>
            {focused ? 'Clear focus' : 'Focus in graph'}
          </button>
          <button
            type="button"
            className="secondary"
            onClick={async () => {
              await navigator.clipboard.writeText(mermaidCode)
            }}
          >
            Copy Mermaid
          </button>
        </div>

        <div className="cockpit2-mermaid-pane">
          <div ref={containerRef} />
        </div>

        <div className="cockpit2-process-step-list">
          {detail.steps.map((step) => (
            <button
              key={`${detail.process.id}-${step.step}-${step.node_id}`}
              type="button"
              className="secondary"
              onClick={() => onSelectNodeId(step.node_id)}
            >
              {step.step}. {step.node_name}
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}
