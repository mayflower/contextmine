import ELK from 'elkjs/lib/elk.bundled.js'

export type LayoutEngine = 'grid' | 'elk_layered' | 'elk_force_like'

export interface LayoutNodeInput {
  id: string
}

export interface LayoutEdgeInput {
  source: string
  target: string
}

export type LayoutPositionMap = Record<string, { x: number; y: number }>

export function runGridLayout(nodes: LayoutNodeInput[], columns: number): LayoutPositionMap {
  const safeColumns = Math.max(columns, 1)
  const positions: LayoutPositionMap = {}
  nodes.forEach((node, index) => {
    positions[node.id] = {
      x: (index % safeColumns) * 260,
      y: Math.floor(index / safeColumns) * 130,
    }
  })
  return positions
}

function algorithmFor(engine: LayoutEngine): string {
  if (engine === 'elk_force_like') return 'stress'
  return 'layered'
}

export async function runElkLayout(
  nodes: LayoutNodeInput[],
  edges: LayoutEdgeInput[],
  engine: LayoutEngine,
): Promise<LayoutPositionMap> {
  const elk = new ELK()
  const graph = {
    id: 'root',
    layoutOptions: {
      'elk.algorithm': algorithmFor(engine),
      'elk.direction': 'RIGHT',
      'elk.spacing.nodeNode': '70',
      'elk.layered.spacing.nodeNodeBetweenLayers': '90',
      'elk.edgeRouting': 'ORTHOGONAL',
    },
    children: nodes.map((node) => ({
      id: node.id,
      width: 220,
      height: 64,
    })),
    edges: edges.map((edge, index) => ({
      id: `edge-${index}`,
      sources: [edge.source],
      targets: [edge.target],
    })),
  }

  const laidOut = await elk.layout(graph as never)
  const positions: LayoutPositionMap = {}
  const children = (laidOut as { children?: Array<{ id: string; x?: number; y?: number }> }).children || []
  for (const child of children) {
    positions[String(child.id)] = {
      x: Number(child.x || 0),
      y: Number(child.y || 0),
    }
  }
  return positions
}
