/**
 * Tests for ArchitectureView rendering + pure helper functions.
 */
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type {
  Arc42DriftPayload,
  Arc42ViewPayload,
  ErmViewPayload,
  PortsAdaptersPayload,
  ViewScenario,
} from '../types'

vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn().mockResolvedValue({ svg: '<svg></svg>', bindFunctions: undefined }),
  },
}))

vi.mock('../utils/mermaidUtils', () => ({
  renderMermaid: vi.fn().mockResolvedValue(undefined),
}))

import ArchitectureView from './ArchitectureView'

const mockScenario: ViewScenario = {
  id: 's1',
  collection_id: 'c1',
  name: 'Base',
  version: 1,
  is_as_is: true,
  base_scenario_id: null,
}

const arc42Payload: Arc42ViewPayload = {
  collection_id: 'c1',
  scenario: mockScenario,
  artifact: { id: 'a1', name: 'arc42', kind: 'ARC42', cached: false },
  section: null,
  arc42: {
    title: 'ContextMine',
    generated_at: '2025-06-01T12:00:00Z',
    sections: {
      '01_introduction': 'ContextMine is a documentation indexing system.',
      '02_constraints': 'Must run on PostgreSQL.',
      '03_context': '',
    },
    markdown: '# ContextMine\n...',
    warnings: ['Missing runtime data'],
    confidence_summary: { total: 10, avg: 0.75, by_source: {} },
    section_coverage: { '01_introduction': true, '02_constraints': true, '03_context': false },
  },
  facts_hash: 'abc123def456ghi789',
  facts_count: 42,
  ports_adapters_count: 5,
  warnings: [],
}

const portsPayload: PortsAdaptersPayload = {
  collection_id: 'c1',
  scenario: mockScenario,
  summary: { total: 3, inbound: 2, outbound: 1 },
  filters: { direction: null, container: null },
  items: [
    { fact_id: 'f1', direction: 'inbound', port_name: 'REST /api/auth', adapter_name: 'FastAPI router', container: 'API Gateway', component: 'auth_router', protocol: 'HTTP', source: 'deterministic', confidence: 0.95, attributes: {}, evidence: [] },
    { fact_id: 'f2', direction: 'inbound', port_name: 'REST /api/billing', adapter_name: 'FastAPI router', container: 'API Gateway', component: 'billing_router', protocol: 'HTTP', source: 'hybrid', confidence: 0.8, attributes: {}, evidence: [] },
    { fact_id: 'f3', direction: 'outbound', port_name: 'PostgreSQL', adapter_name: 'SQLAlchemy', container: 'Database', component: 'db_client', protocol: 'TCP', source: 'deterministic', confidence: 0.99, attributes: {}, evidence: [] },
  ],
  warnings: [],
}

const driftPayload: Arc42DriftPayload = {
  collection_id: 'c1',
  scenario: mockScenario,
  baseline_scenario: null,
  generated_at: '2025-06-01T12:00:00Z',
  current_hash: 'abc123',
  baseline_hash: 'def456',
  summary: { total: 2, by_type: { added: 1, removed: 1 }, severity: 'medium' },
  deltas: [
    { delta_type: 'added', subject: 'cache-service', detail: 'New cache service added', confidence: 0.9 },
    { delta_type: 'removed', subject: 'legacy-auth', detail: 'Legacy auth service removed', confidence: 0.85 },
  ],
  warnings: [],
}

const ermPayload: ErmViewPayload = {
  collection_id: 'c1',
  scenario: mockScenario,
  summary: { tables: 2, columns: 8, foreign_keys: 1, has_mermaid: true },
  tables: [
    {
      id: 't1', natural_key: 'users', name: 'users', description: 'User accounts', column_count: 4,
      primary_keys: ['id'],
      columns: [
        { id: 'col1', natural_key: 'users.id', name: 'id', table: 'users', type: 'uuid', nullable: false, primary_key: true, foreign_key: null },
        { id: 'col2', natural_key: 'users.email', name: 'email', table: 'users', type: 'varchar', nullable: false, primary_key: false, foreign_key: null },
      ],
    },
    {
      id: 't2', natural_key: 'orders', name: 'orders', description: 'Orders table', column_count: 4,
      primary_keys: ['id'],
      columns: [
        { id: 'col3', natural_key: 'orders.id', name: 'id', table: 'orders', type: 'uuid', nullable: false, primary_key: true, foreign_key: null },
        { id: 'col4', natural_key: 'orders.user_id', name: 'user_id', table: 'orders', type: 'uuid', nullable: false, primary_key: false, foreign_key: 'users.id' },
      ],
    },
  ],
  foreign_keys: [
    { id: 'fk1', fk_name: 'fk_user', source_table: 'orders', source_column: 'user_id', target_table: 'users', target_column: 'id', source_column_node_id: 'col4', target_column_node_id: 'col1' },
  ],
  mermaid: { artifact_id: 'erd1', name: 'ERD', content: 'erDiagram\n  users ||--o{ orders : has', meta: {} },
  warnings: [],
}

function makeProps(overrides: Partial<Parameters<typeof ArchitectureView>[0]> = {}) {
  return {
    state: 'ready' as const,
    error: '',
    arc42: arc42Payload,
    portsAdapters: portsPayload,
    drift: driftPayload,
    erm: ermPayload,
    panelErrors: { arc42: '', ports: '', drift: '', erm: '' },
    actions: {
      reindexState: 'idle' as const,
      reindexMessage: '',
      regenerateState: 'idle' as const,
      regenerateMessage: '',
    },
    onReindex: vi.fn().mockResolvedValue(true),
    onRegenerateArc42: vi.fn().mockResolvedValue(true),
    onRetry: vi.fn(),
    ...overrides,
  }
}

// --- Pure helper tests (replicated from component) ---

function sectionLabel(sectionKey: string): string {
  const normalized = sectionKey.replace(/^\d+_/, '').replaceAll('_', ' ').trim()
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
  if (value === null || value === undefined || Number.isNaN(value)) return 'n/a'
  return `${Math.round(value * 100)}%`
}

describe('sectionLabel', () => {
  it('strips numeric prefix and capitalizes', () => {
    expect(sectionLabel('01_introduction')).toBe('Introduction')
  })

  it('handles multi-word sections', () => {
    expect(sectionLabel('03_system_scope_and_context')).toBe('System Scope And Context')
  })

  it('handles sections without number prefix', () => {
    expect(sectionLabel('glossary')).toBe('Glossary')
  })
})

describe('shortHash', () => {
  it('returns n/a for undefined', () => {
    expect(shortHash(undefined)).toBe('n/a')
  })

  it('returns full hash if short', () => {
    expect(shortHash('abc123')).toBe('abc123')
  })

  it('truncates long hashes', () => {
    expect(shortHash('abc123def456ghi789')).toBe('abc123de...ghi789')
  })
})

describe('toPercentage', () => {
  it('returns n/a for null', () => {
    expect(toPercentage(null)).toBe('n/a')
  })

  it('returns n/a for undefined', () => {
    expect(toPercentage(undefined)).toBe('n/a')
  })

  it('returns n/a for NaN', () => {
    expect(toPercentage(NaN)).toBe('n/a')
  })

  it('converts decimal to percentage', () => {
    expect(toPercentage(0.75)).toBe('75%')
  })

  it('handles zero', () => {
    expect(toPercentage(0)).toBe('0%')
  })

  it('handles 1.0', () => {
    expect(toPercentage(1)).toBe('100%')
  })
})

// --- Rendering tests ---

describe('ArchitectureView rendering', () => {
  it('renders the architecture panel with header', () => {
    render(<ArchitectureView {...makeProps()} />)
    expect(screen.getByText('Architecture intelligence')).toBeInTheDocument()
  })

  it('renders delta count in description', () => {
    render(<ArchitectureView {...makeProps()} />)
    expect(screen.getByText(/2 deltas/)).toBeInTheDocument()
  })

  it('renders action buttons', () => {
    render(<ArchitectureView {...makeProps()} />)
    expect(screen.getByText('Reindex data')).toBeInTheDocument()
    expect(screen.getByText('Regenerate arc42')).toBeInTheDocument()
  })

  it('shows reindexing text when reindex is loading', () => {
    render(<ArchitectureView {...makeProps({
      actions: { reindexState: 'loading', reindexMessage: 'Processing...', regenerateState: 'idle', regenerateMessage: '' },
    })} />)
    expect(screen.getByText('Reindexing...')).toBeInTheDocument()
    expect(screen.getByText(/Processing.../)).toBeInTheDocument()
  })

  it('renders KPI cards', () => {
    render(<ArchitectureView {...makeProps()} />)
    expect(screen.getByText('arc42 section coverage')).toBeInTheDocument()
    expect(screen.getByText('Facts hash')).toBeInTheDocument()
  })

  it('renders tab navigation', () => {
    render(<ArchitectureView {...makeProps()} />)
    expect(screen.getByRole('tab', { name: 'arc42' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Ports/Adapters' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'ERD' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Drift' })).toBeInTheDocument()
  })

  // arc42 tab (default)
  it('renders arc42 sections navigation by default', () => {
    render(<ArchitectureView {...makeProps()} />)
    expect(screen.getByText('arc42 sections')).toBeInTheDocument()
    // Introduction appears in both nav and heading
    expect(screen.getAllByText('Introduction').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('Constraints')).toBeInTheDocument()
  })

  it('renders arc42 section content', () => {
    render(<ArchitectureView {...makeProps()} />)
    expect(screen.getByText('ContextMine is a documentation indexing system.')).toBeInTheDocument()
  })

  // Ports tab
  it('renders ports/adapters when ports tab is clicked', async () => {
    render(<ArchitectureView {...makeProps()} />)
    await userEvent.click(screen.getByRole('tab', { name: 'Ports/Adapters' }))
    expect(screen.getByText('Ports & adapters map')).toBeInTheDocument()
    expect(screen.getByText('Inbound ports')).toBeInTheDocument()
    expect(screen.getByText('Outbound ports')).toBeInTheDocument()
  })

  it('renders port cards with names and protocols', async () => {
    render(<ArchitectureView {...makeProps()} />)
    await userEvent.click(screen.getByRole('tab', { name: 'Ports/Adapters' }))
    expect(screen.getByText('REST /api/auth')).toBeInTheDocument()
    expect(screen.getByText('PostgreSQL')).toBeInTheDocument()
  })

  // ERD tab
  it('renders ERM section when ERD tab is clicked', async () => {
    render(<ArchitectureView {...makeProps()} />)
    await userEvent.click(screen.getByRole('tab', { name: 'ERD' }))
    expect(screen.getByText('ERM data model')).toBeInTheDocument()
    expect(screen.getByText(/2 tables/)).toBeInTheDocument()
  })

  it('renders ERM table cards', async () => {
    render(<ArchitectureView {...makeProps()} />)
    await userEvent.click(screen.getByRole('tab', { name: 'ERD' }))
    expect(screen.getByText('users')).toBeInTheDocument()
    expect(screen.getByText('orders')).toBeInTheDocument()
  })

  it('renders ERM column names', async () => {
    render(<ArchitectureView {...makeProps()} />)
    await userEvent.click(screen.getByRole('tab', { name: 'ERD' }))
    expect(screen.getByText('email')).toBeInTheDocument()
    expect(screen.getByText('user_id')).toBeInTheDocument()
  })

  // Drift tab
  it('renders drift report when drift tab is clicked', async () => {
    render(<ArchitectureView {...makeProps()} />)
    await userEvent.click(screen.getByRole('tab', { name: 'Drift' }))
    expect(screen.getByText('Advisory drift report')).toBeInTheDocument()
  })

  it('renders drift deltas', async () => {
    render(<ArchitectureView {...makeProps()} />)
    await userEvent.click(screen.getByRole('tab', { name: 'Drift' }))
    expect(screen.getByText('New cache service added')).toBeInTheDocument()
    expect(screen.getByText('Legacy auth service removed')).toBeInTheDocument()
    expect(screen.getByText('cache-service')).toBeInTheDocument()
    expect(screen.getByText('legacy-auth')).toBeInTheDocument()
  })

  // Empty states
  it('renders empty arc42 state', () => {
    render(<ArchitectureView {...makeProps({
      arc42: { ...arc42Payload, arc42: { ...arc42Payload.arc42, sections: {} } },
    })} />)
    expect(screen.getByText('No arc42 sections available')).toBeInTheDocument()
  })

  it('renders empty ports state', async () => {
    render(<ArchitectureView {...makeProps({ portsAdapters: { ...portsPayload, items: [] } })} />)
    await userEvent.click(screen.getByRole('tab', { name: 'Ports/Adapters' }))
    expect(screen.getByText('No ports/adapters found')).toBeInTheDocument()
  })

  it('renders empty drift state', async () => {
    render(<ArchitectureView {...makeProps({ drift: { ...driftPayload, deltas: [] } })} />)
    await userEvent.click(screen.getByRole('tab', { name: 'Drift' }))
    expect(screen.getByText('No architecture drift detected')).toBeInTheDocument()
  })

  it('renders empty ERM table state', async () => {
    render(<ArchitectureView {...makeProps({ erm: { ...ermPayload, tables: [], mermaid: null } })} />)
    await userEvent.click(screen.getByRole('tab', { name: 'ERD' }))
    expect(screen.getByText('No ERD diagram available')).toBeInTheDocument()
    expect(screen.getByText('No ERM tables found')).toBeInTheDocument()
  })

  // Panel errors
  it('renders arc42 panel error', () => {
    render(<ArchitectureView {...makeProps({ panelErrors: { arc42: 'arc42 failed', ports: '', drift: '', erm: '' } })} />)
    expect(screen.getByText('arc42 failed')).toBeInTheDocument()
  })

  it('renders ports panel error', async () => {
    render(<ArchitectureView {...makeProps({ panelErrors: { arc42: '', ports: 'Ports failed', drift: '', erm: '' } })} />)
    await userEvent.click(screen.getByRole('tab', { name: 'Ports/Adapters' }))
    expect(screen.getByText('Ports failed')).toBeInTheDocument()
  })

  // Warnings
  it('renders extraction warnings', () => {
    render(<ArchitectureView {...makeProps({
      arc42: { ...arc42Payload, warnings: ['Weak evidence'] },
    })} />)
    expect(screen.getByText('Extraction warnings')).toBeInTheDocument()
    expect(screen.getByText('Weak evidence')).toBeInTheDocument()
  })

  // Loading/error
  it('renders loading skeleton when loading and no data', () => {
    render(<ArchitectureView {...makeProps({
      state: 'loading',
      arc42: null,
      portsAdapters: null,
      drift: null,
      erm: null,
    })} />)
    expect(screen.queryByText('Architecture intelligence')).not.toBeInTheDocument()
  })

  it('renders error state when error and no data', () => {
    render(<ArchitectureView {...makeProps({
      state: 'error',
      error: 'Server error',
      arc42: null,
      portsAdapters: null,
      drift: null,
      erm: null,
    })} />)
    expect(screen.getByText('Architecture view request failed')).toBeInTheDocument()
  })

  it('calls onReindex when reindex button is clicked', async () => {
    const onReindex = vi.fn().mockResolvedValue(true)
    render(<ArchitectureView {...makeProps({ onReindex })} />)
    await userEvent.click(screen.getByText('Reindex data'))
    expect(onReindex).toHaveBeenCalledOnce()
  })

  it('calls onRegenerateArc42 when regenerate button is clicked', async () => {
    const onRegenerateArc42 = vi.fn().mockResolvedValue(true)
    render(<ArchitectureView {...makeProps({ onRegenerateArc42 })} />)
    await userEvent.click(screen.getByText('Regenerate arc42'))
    expect(onRegenerateArc42).toHaveBeenCalledOnce()
  })
})
