import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import NodeInspector from './NodeInspector'
import type { OverlayState, TwinGraphResponse } from '../types'

const emptyGraph: TwinGraphResponse = {
  nodes: [],
  edges: [],
  page: 0,
  limit: 100,
  total_nodes: 0,
}

const graphWithNode: TwinGraphResponse = {
  nodes: [
    { id: 'n1', natural_key: 'service:auth', kind: 'CONTAINER', name: 'AuthService', meta: { loc: 500 } },
  ],
  edges: [],
  page: 0,
  limit: 100,
  total_nodes: 1,
}

const defaultOverlay: OverlayState = {
  mode: 'none',
  runtimeByNodeKey: {},
  riskByNodeKey: {},
  loadedAt: null,
}

describe('NodeInspector', () => {
  it('shows placeholder when no node is selected', () => {
    render(
      <NodeInspector
        selectedNodeId=""
        graph={emptyGraph}
        neighborhood={emptyGraph}
        neighborhoodState="idle"
        neighborhoodError=""
        overlay={defaultOverlay}
        onClearSelection={vi.fn()}
      />,
    )
    expect(screen.getByText('Node inspector')).toBeInTheDocument()
    expect(screen.getByText(/Select a node/)).toBeInTheDocument()
  })

  it('shows placeholder when node id not found in graph', () => {
    render(
      <NodeInspector
        selectedNodeId="nonexistent"
        graph={emptyGraph}
        neighborhood={emptyGraph}
        neighborhoodState="idle"
        neighborhoodError=""
        overlay={defaultOverlay}
        onClearSelection={vi.fn()}
      />,
    )
    expect(screen.getByText(/Select a node/)).toBeInTheDocument()
  })

  it('renders node details when a node is selected', () => {
    render(
      <NodeInspector
        selectedNodeId="n1"
        graph={graphWithNode}
        neighborhood={emptyGraph}
        neighborhoodState="idle"
        neighborhoodError=""
        overlay={defaultOverlay}
        onClearSelection={vi.fn()}
      />,
    )
    expect(screen.getByText('AuthService')).toBeInTheDocument()
    expect(screen.getByText('CONTAINER')).toBeInTheDocument()
  })

  it('renders Clear button and handles click', async () => {
    const onClearSelection = vi.fn()
    render(
      <NodeInspector
        selectedNodeId="n1"
        graph={graphWithNode}
        neighborhood={emptyGraph}
        neighborhoodState="idle"
        neighborhoodError=""
        overlay={defaultOverlay}
        onClearSelection={onClearSelection}
      />,
    )
    const clearBtn = screen.getByRole('button', { name: 'Clear' })
    await userEvent.click(clearBtn)
    expect(onClearSelection).toHaveBeenCalledOnce()
  })
})
