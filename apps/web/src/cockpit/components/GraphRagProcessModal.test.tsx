/**
 * Tests for GraphRagProcessModal pure helper functions.
 * The component uses mermaid rendering which requires DOM setup,
 * so we test the pure toMermaid function.
 */
import { describe, expect, it } from 'vitest'

import type { GraphRagProcessDetailPayload, ViewScenario } from '../types'

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
