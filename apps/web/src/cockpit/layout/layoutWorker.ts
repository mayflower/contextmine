/// <reference lib="webworker" />

import { runElkLayout, runGridLayout, type LayoutEdgeInput, type LayoutEngine, type LayoutNodeInput } from './layoutCore'

interface LayoutWorkerRequest {
  nodes: LayoutNodeInput[]
  edges: LayoutEdgeInput[]
  engine: LayoutEngine
  columns: number
}

self.onmessage = async (event: MessageEvent<LayoutWorkerRequest>) => {
  const { nodes, edges, engine, columns } = event.data
  const startedAt = performance.now()
  try {
    const positions =
      engine === 'grid' ? runGridLayout(nodes, columns) : await runElkLayout(nodes, edges, engine)
    const durationMs = performance.now() - startedAt
    postMessage({ ok: true, positions, durationMs })
  } catch (error) {
    const durationMs = performance.now() - startedAt
    postMessage({
      ok: false,
      durationMs,
      error: error instanceof Error ? error.message : 'Unknown worker layout error',
    })
  }
}
