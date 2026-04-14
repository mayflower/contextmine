/**
 * Tests for C4DiffView rendering + pure helper functions.
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { MermaidPayload, ViewScenario } from '../types'

vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn().mockResolvedValue({ svg: '<svg></svg>', bindFunctions: undefined }),
  },
}))

vi.mock('../utils/mermaidUtils', () => ({
  renderMermaid: vi.fn().mockResolvedValue(undefined),
}))

vi.mock('../flags', () => ({
  cockpitFlags: {
    c4RenderedDiff: false,
  },
}))

import C4DiffView from './C4DiffView'

const mockScenario: ViewScenario = {
  id: 's1',
  collection_id: 'c1',
  name: 'Base',
  version: 1,
  is_as_is: true,
  base_scenario_id: null,
}

const singleMermaid: MermaidPayload = {
  collection_id: 'c1',
  scenario: mockScenario,
  mode: 'single',
  content: 'C4Context\n  System(api, "API Gateway")\n  System(db, "Database")',
  warnings: [],
}

const compareMermaid: MermaidPayload = {
  collection_id: 'c1',
  scenario: mockScenario,
  mode: 'compare',
  as_is: 'C4Context\n  System(api, "API")\n  System(db, "DB")',
  to_be: 'C4Context\n  System(api, "API")\n  System(db, "DB")\n  System(cache, "Cache")',
  warnings: ['Partial diagram'],
  as_is_warnings: ['AS-IS had issues'],
  to_be_warnings: ['TO-BE had issues'],
}

function makeProps(overrides: Partial<Parameters<typeof C4DiffView>[0]> = {}) {
  return {
    mermaid: singleMermaid,
    state: 'ready' as const,
    error: '',
    onRetry: vi.fn(),
    ...overrides,
  }
}

// --- Pure helper tests (replicated from component) ---

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

describe('extractElementIds', () => {
  it('extracts System ids', () => {
    const ids = extractElementIds('C4Context\n  System(api, "API")')
    expect(ids.has('api')).toBe(true)
  })

  it('extracts multiple element types', () => {
    const source = 'Container(web, "Web")\nPerson(user, "User")\nSystemDb(db, "DB")'
    const ids = extractElementIds(source)
    expect(ids.has('web')).toBe(true)
    expect(ids.has('user')).toBe(true)
    expect(ids.has('db')).toBe(true)
  })

  it('returns empty set for non-matching content', () => {
    const ids = extractElementIds('just some text')
    expect(ids.size).toBe(0)
  })
})

describe('withSemanticClasses', () => {
  it('adds classDef and class lines for added', () => {
    const result = withSemanticClasses('source', new Set(['cache']), 'added')
    expect(result).toContain('classDef added')
    expect(result).toContain('class cache added;')
  })

  it('adds classDef and class lines for removed', () => {
    const result = withSemanticClasses('source', new Set(['old']), 'removed')
    expect(result).toContain('classDef removed')
    expect(result).toContain('class old removed;')
  })

  it('returns source unchanged when ids set is empty', () => {
    expect(withSemanticClasses('source', new Set(), 'added')).toBe('source')
  })

  it('returns source unchanged when source is empty', () => {
    expect(withSemanticClasses('  ', new Set(['x']), 'added')).toBe('  ')
  })
})

// --- Rendering tests ---

describe('C4DiffView rendering', () => {
  it('renders the C4 panel with header', () => {
    render(<C4DiffView {...makeProps()} />)
    expect(screen.getByText('Mermaid C4 diff')).toBeInTheDocument()
  })

  it('renders source toggle button', () => {
    render(<C4DiffView {...makeProps()} />)
    expect(screen.getByText('Show source')).toBeInTheDocument()
  })

  it('renders zoom control', () => {
    render(<C4DiffView {...makeProps()} />)
    expect(screen.getByText('Zoom')).toBeInTheDocument()
  })

  it('renders single mode content as pre', () => {
    render(<C4DiffView {...makeProps()} />)
    const preElements = screen.getByRole('tabpanel').querySelectorAll('pre')
    expect(preElements.length).toBeGreaterThan(0)
  })

  it('renders compare mode with AS-IS and TO-BE columns', () => {
    render(<C4DiffView {...makeProps({ mermaid: compareMermaid })} />)
    expect(screen.getByText('AS-IS')).toBeInTheDocument()
    expect(screen.getByText('TO-BE')).toBeInTheDocument()
    expect(screen.getByText('Baseline')).toBeInTheDocument()
    expect(screen.getByText('Target')).toBeInTheDocument()
  })

  it('renders warnings in compare mode', () => {
    render(<C4DiffView {...makeProps({ mermaid: compareMermaid })} />)
    expect(screen.getByText(/Degraded or truncated diagram output/)).toBeInTheDocument()
    expect(screen.getByText('Partial diagram')).toBeInTheDocument()
    expect(screen.getByText('AS-IS had issues')).toBeInTheDocument()
    expect(screen.getByText('TO-BE had issues')).toBeInTheDocument()
  })

  it('renders loading skeleton when state is loading and no data', () => {
    render(<C4DiffView {...makeProps({ state: 'loading', mermaid: null })} />)
    expect(screen.queryByText('Mermaid C4 diff')).not.toBeInTheDocument()
  })

  it('renders error state when state is error and no data', () => {
    render(<C4DiffView {...makeProps({ state: 'error', error: 'Failed to fetch', mermaid: null })} />)
    expect(screen.getByText('C4 diagram request failed')).toBeInTheDocument()
    expect(screen.getByText('Failed to fetch')).toBeInTheDocument()
  })

  it('source toggle button is disabled when c4RenderedDiff is false', () => {
    render(<C4DiffView {...makeProps()} />)
    const btn = screen.getByText('Show source')
    expect(btn).toBeDisabled()
  })

  it('renders the description text for source compare mode', () => {
    render(<C4DiffView {...makeProps()} />)
    expect(screen.getByText('AS-IS and TO-BE source compare.')).toBeInTheDocument()
  })
})
