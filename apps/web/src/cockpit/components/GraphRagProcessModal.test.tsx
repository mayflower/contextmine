/**
 * Tests for GraphRagProcessModal rendering + pure helper functions.
 */
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { GraphRagProcessDetailPayload, ViewScenario } from '../types'

vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn().mockResolvedValue({ svg: '<svg></svg>', bindFunctions: undefined }),
  },
}))

vi.mock('../utils/mermaidUtils', () => ({
  renderMermaidSvg: vi.fn(),
}))

import GraphRagProcessModal from './GraphRagProcessModal'

// Replicated from GraphRagProcessModal.tsx (lines 15-37)
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

const scenario: ViewScenario = {
  id: 's1',
  collection_id: 'c1',
  name: 'Main',
  version: 1,
  is_as_is: true,
  base_scenario_id: null,
}

// --- Pure helper tests ---

describe('toMermaid', () => {
  it('generates flowchart TD header', () => {
    const detail: GraphRagProcessDetailPayload = {
      collection_id: 'c1',
      scenario,
      process: {
        id: 'p1',
        label: 'Process 1',
        process_type: 'intra_community',
        step_count: 0,
        community_ids: [],
        entry_node_id: '',
        terminal_node_id: '',
      },
      steps: [],
      edges: [],
    }
    expect(toMermaid(detail)).toBe('flowchart TD')
  })

  it('generates nodes from steps', () => {
    const detail: GraphRagProcessDetailPayload = {
      collection_id: 'c1',
      scenario,
      process: {
        id: 'p1',
        label: 'Process 1',
        process_type: 'intra_community',
        step_count: 2,
        community_ids: [],
        entry_node_id: 'n1',
        terminal_node_id: 'n2',
      },
      steps: [
        { step: 1, node_id: 'n1', node_name: 'Start', node_kind: 'CONTAINER', node_natural_key: 'start' },
        { step: 2, node_id: 'n2', node_name: 'End', node_kind: 'CONTAINER', node_natural_key: 'end' },
      ],
      edges: [
        { id: 'e1', source_node_id: 'n1', target_node_id: 'n2', kind: 'CALLS', meta: {} },
      ],
    }

    const result = toMermaid(detail)
    expect(result).toContain('flowchart TD')
    expect(result).toContain('n1["1. Start"]')
    expect(result).toContain('n2["2. End"]')
    expect(result).toContain('n1 --> n2')
  })

  it('sanitizes special characters in node IDs', () => {
    const detail: GraphRagProcessDetailPayload = {
      collection_id: 'c1',
      scenario,
      process: {
        id: 'p1',
        label: 'Process 1',
        process_type: 'intra_community',
        step_count: 1,
        community_ids: [],
        entry_node_id: 'node-with.dots',
        terminal_node_id: 'node-with.dots',
      },
      steps: [
        { step: 1, node_id: 'node-with.dots', node_name: 'Special', node_kind: 'FILE', node_natural_key: 'x' },
      ],
      edges: [],
    }

    const result = toMermaid(detail)
    expect(result).toContain('node_with_dots["1. Special"]')
  })

  it('replaces quotes in labels', () => {
    const detail: GraphRagProcessDetailPayload = {
      collection_id: 'c1',
      scenario,
      process: {
        id: 'p1',
        label: 'P',
        process_type: 'intra_community',
        step_count: 1,
        community_ids: [],
        entry_node_id: 'n1',
        terminal_node_id: 'n1',
      },
      steps: [
        { step: 1, node_id: 'n1', node_name: 'Say "hello"', node_kind: 'FILE', node_natural_key: 'x' },
      ],
      edges: [],
    }

    const result = toMermaid(detail)
    expect(result).toContain("1. Say 'hello'")
  })

  it('skips self-referencing edges', () => {
    const detail: GraphRagProcessDetailPayload = {
      collection_id: 'c1',
      scenario,
      process: {
        id: 'p1',
        label: 'P',
        process_type: 'intra_community',
        step_count: 1,
        community_ids: [],
        entry_node_id: 'n1',
        terminal_node_id: 'n1',
      },
      steps: [
        { step: 1, node_id: 'n1', node_name: 'A', node_kind: 'FILE', node_natural_key: 'x' },
      ],
      edges: [
        { id: 'e1', source_node_id: 'n1', target_node_id: 'n1', kind: 'SELF', meta: {} },
      ],
    }

    const result = toMermaid(detail)
    expect(result).not.toContain('-->')
  })
})

// --- Rendering tests ---

const mockDetail: GraphRagProcessDetailPayload = {
  collection_id: 'c1',
  scenario,
  process: {
    id: 'p1',
    label: 'Authentication Flow',
    process_type: 'cross_community',
    step_count: 3,
    community_ids: ['c1', 'c2'],
    entry_node_id: 'n1',
    terminal_node_id: 'n3',
  },
  steps: [
    { step: 1, node_id: 'n1', node_name: 'Login Handler', node_kind: 'SYMBOL', node_natural_key: 'sym:login' },
    { step: 2, node_id: 'n2', node_name: 'Token Validator', node_kind: 'SYMBOL', node_natural_key: 'sym:validate' },
    { step: 3, node_id: 'n3', node_name: 'Session Creator', node_kind: 'SYMBOL', node_natural_key: 'sym:session' },
  ],
  edges: [
    { id: 'e1', source_node_id: 'n1', target_node_id: 'n2', kind: 'CALLS', meta: {} },
    { id: 'e2', source_node_id: 'n2', target_node_id: 'n3', kind: 'CALLS', meta: {} },
  ],
}

function makeProps(overrides: Partial<Parameters<typeof GraphRagProcessModal>[0]> = {}) {
  return {
    detail: mockDetail,
    focused: false,
    onClose: vi.fn(),
    onSelectNodeId: vi.fn(),
    onToggleFocus: vi.fn(),
    ...overrides,
  }
}

describe('GraphRagProcessModal rendering', () => {
  it('renders the modal dialog', () => {
    render(<GraphRagProcessModal {...makeProps()} />)
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    expect(screen.getByLabelText('Process flow')).toBeInTheDocument()
  })

  it('renders the process label as heading', () => {
    render(<GraphRagProcessModal {...makeProps()} />)
    expect(screen.getByText('Authentication Flow')).toBeInTheDocument()
  })

  it('renders process type and step count', () => {
    render(<GraphRagProcessModal {...makeProps()} />)
    expect(screen.getByText(/Cross-community process/)).toBeInTheDocument()
    expect(screen.getByText(/Steps: 3/)).toBeInTheDocument()
  })

  it('renders close button and calls onClose', async () => {
    const onClose = vi.fn()
    render(<GraphRagProcessModal {...makeProps({ onClose })} />)
    await userEvent.click(screen.getByText('Close'))
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('renders Focus in graph button when not focused', () => {
    render(<GraphRagProcessModal {...makeProps({ focused: false })} />)
    expect(screen.getByText('Focus in graph')).toBeInTheDocument()
  })

  it('renders Clear focus button when focused', () => {
    render(<GraphRagProcessModal {...makeProps({ focused: true })} />)
    expect(screen.getByText('Clear focus')).toBeInTheDocument()
  })

  it('calls onToggleFocus when focus button is clicked', async () => {
    const onToggleFocus = vi.fn()
    render(<GraphRagProcessModal {...makeProps({ onToggleFocus })} />)
    await userEvent.click(screen.getByText('Focus in graph'))
    expect(onToggleFocus).toHaveBeenCalledOnce()
  })

  it('renders Copy Mermaid button', () => {
    render(<GraphRagProcessModal {...makeProps()} />)
    expect(screen.getByText('Copy Mermaid')).toBeInTheDocument()
  })

  it('renders step buttons', () => {
    render(<GraphRagProcessModal {...makeProps()} />)
    expect(screen.getByText('1. Login Handler')).toBeInTheDocument()
    expect(screen.getByText('2. Token Validator')).toBeInTheDocument()
    expect(screen.getByText('3. Session Creator')).toBeInTheDocument()
  })

  it('calls onSelectNodeId when a step is clicked', async () => {
    const onSelectNodeId = vi.fn()
    render(<GraphRagProcessModal {...makeProps({ onSelectNodeId })} />)
    await userEvent.click(screen.getByText('1. Login Handler'))
    expect(onSelectNodeId).toHaveBeenCalledWith('n1')
  })

  it('renders intra-community process type', () => {
    const intraDetail = {
      ...mockDetail,
      process: { ...mockDetail.process, process_type: 'intra_community' as const },
    }
    render(<GraphRagProcessModal {...makeProps({ detail: intraDetail })} />)
    expect(screen.getByText(/Intra-community process/)).toBeInTheDocument()
  })
})
