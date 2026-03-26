/**
 * Tests for EvolutionView rendering + pure helper functions.
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type {
  FitnessFunctionsPayload,
  InvestmentUtilizationPayload,
  KnowledgeIslandsPayload,
  TemporalCouplingPayload,
  ViewScenario,
} from '../types'

vi.mock('reactflow', () => {
  const Background = () => <div data-testid="rf-background" />
  const Controls = () => <div data-testid="rf-controls" />
  const ReactFlowComponent = (props: Record<string, unknown>) => (
    <div data-testid="reactflow" aria-label={props['aria-label'] as string} />
  )
  return {
    default: ReactFlowComponent,
    ReactFlow: ReactFlowComponent,
    Background,
    Controls,
  }
})

import EvolutionView from './EvolutionView'

const mockScenario: ViewScenario = {
  id: 's1',
  collection_id: 'c1',
  name: 'Base',
  version: 1,
  is_as_is: true,
  base_scenario_id: null,
}

const investmentPayload: InvestmentUtilizationPayload = {
  collection_id: 'c1',
  scenario: mockScenario,
  status: 'ready',
  reason: 'ok',
  entity_level: 'container',
  window_days: 90,
  summary: {
    total_entities: 3,
    coverage_entity_ratio: 0.8,
    utilization_available: true,
    quadrants: { strength: 1, overinvestment: 1, efficient_core: 1, opportunity_or_retire: 0 },
  },
  items: [
    { entity_key: 'auth', label: 'Auth Service', size: 5000, investment_score: 0.8, utilization_score: 0.6, coverage_avg: 0.75, change_frequency_avg: 3.2, churn_avg: 150, quadrant: 'strength' },
    { entity_key: 'billing', label: 'Billing Service', size: 3000, investment_score: 0.3, utilization_score: 0.9, coverage_avg: 0.5, change_frequency_avg: 1.1, churn_avg: 50, quadrant: 'efficient_core' },
  ],
  warnings: ['Coverage data is partial'],
}

const knowledgePayload: KnowledgeIslandsPayload = {
  collection_id: 'c1',
  scenario: mockScenario,
  status: 'ready',
  reason: 'ok',
  entity_level: 'container',
  ownership_threshold: 0.7,
  window_days: 180,
  summary: {
    files: 120,
    entities: 4,
    bus_factor_global: 2,
    single_owner_files: 30,
    churn_p75: 200,
  },
  entities: [
    { entity_key: 'auth', label: 'Auth', files: 40, bus_factor: 1, dominant_owner: 'alice', dominant_share: 0.85, single_owner_ratio: 0.6 },
    { entity_key: 'billing', label: 'Billing', files: 30, bus_factor: 3, dominant_owner: 'bob', dominant_share: 0.4, single_owner_ratio: 0.1 },
  ],
  at_risk_files: [],
  warnings: [],
}

const couplingPayload: TemporalCouplingPayload = {
  collection_id: 'c1',
  scenario: mockScenario,
  status: 'ready',
  reason: 'ok',
  entity_level: 'container',
  window_days: 90,
  min_jaccard: 0.3,
  max_edges: 50,
  summary: { nodes: 2, edges: 1, cross_boundary_edges: 1, avg_jaccard: 0.45 },
  graph: {
    nodes: [
      { id: 'cn1', key: 'auth', label: 'Auth', entity_level: 'container' },
      { id: 'cn2', key: 'billing', label: 'Billing', entity_level: 'container' },
    ],
    edges: [
      { id: 'ce1', source: 'cn1', target: 'cn2', co_change_count: 10, source_change_count: 20, target_change_count: 15, ratio_source_to_target: 0.5, ratio_target_to_source: 0.67, jaccard: 0.45, cross_boundary: true },
    ],
  },
  warnings: [],
}

const fitnessPayload: FitnessFunctionsPayload = {
  collection_id: 'c1',
  scenario: mockScenario,
  status: 'ready',
  reason: 'ok',
  include_resolved: false,
  window_days: 30,
  summary: { rules: 1, violations: 2, open: 2, resolved: 0, highest_severity: 'high' },
  rules: [
    { rule_id: 'CYCLIC_DEP', finding_type: 'cyclic_dependency', count: 2, open: 2, resolved: 0, highest_severity: 'high' },
  ],
  violations: [
    { id: 'v1', rule_id: 'CYCLIC_DEP', finding_type: 'cyclic_dependency', severity: 'high', confidence: 'high', status: 'open', subject: 'auth->billing->auth', message: 'Cyclic dependency detected', filename: 'src/auth.py', line_number: 42, created_at: '2025-01-01', updated_at: '2025-01-01', meta: {} },
    { id: 'v2', rule_id: 'CYCLIC_DEP', finding_type: 'cyclic_dependency', severity: 'medium', confidence: 'medium', status: 'open', subject: 'billing->gateway', message: 'Potential cycle', filename: 'src/billing.py', line_number: 15, created_at: '2025-01-02', updated_at: '2025-01-02', meta: {} },
  ],
  warnings: [],
}

function makeProps(overrides: Partial<Parameters<typeof EvolutionView>[0]> = {}) {
  return {
    state: 'ready' as const,
    error: '',
    investmentUtilization: investmentPayload,
    knowledgeIslands: knowledgePayload,
    temporalCoupling: couplingPayload,
    fitnessFunctions: fitnessPayload,
    panelErrors: { investment: '', knowledge: '', coupling: '', fitness: '' },
    onRetry: vi.fn(),
    ...overrides,
  }
}

// --- Pure helper tests (replicated from component) ---

function severityClass(severity: string): string {
  const normalized = severity.trim().toLowerCase()
  if (normalized === 'critical' || normalized === 'high') return 'risk-high'
  if (normalized === 'medium') return 'risk-medium'
  return 'risk-low'
}

function bubbleColor(quadrant: string): string {
  if (quadrant === 'strength') return '#166534'
  if (quadrant === 'overinvestment') return '#b91c1c'
  if (quadrant === 'efficient_core') return '#1d4ed8'
  if (quadrant === 'opportunity_or_retire') return '#92400e'
  return '#6b7280'
}

describe('severityClass', () => {
  it('returns risk-high for critical', () => expect(severityClass('critical')).toBe('risk-high'))
  it('returns risk-high for high', () => expect(severityClass('high')).toBe('risk-high'))
  it('returns risk-medium for medium', () => expect(severityClass('medium')).toBe('risk-medium'))
  it('returns risk-low for low', () => expect(severityClass('low')).toBe('risk-low'))
  it('returns risk-low for unknown', () => expect(severityClass('unknown')).toBe('risk-low'))
  it('handles whitespace', () => expect(severityClass('  HIGH  ')).toBe('risk-high'))
  it('handles case insensitivity', () => {
    expect(severityClass('CRITICAL')).toBe('risk-high')
    expect(severityClass('Medium')).toBe('risk-medium')
  })
})

describe('bubbleColor', () => {
  it('returns green for strength', () => expect(bubbleColor('strength')).toBe('#166534'))
  it('returns red for overinvestment', () => expect(bubbleColor('overinvestment')).toBe('#b91c1c'))
  it('returns blue for efficient_core', () => expect(bubbleColor('efficient_core')).toBe('#1d4ed8'))
  it('returns brown for opportunity_or_retire', () => expect(bubbleColor('opportunity_or_retire')).toBe('#92400e'))
  it('returns gray for unknown', () => {
    expect(bubbleColor('other')).toBe('#6b7280')
    expect(bubbleColor('')).toBe('#6b7280')
  })
})

// --- Rendering tests ---

describe('EvolutionView rendering', () => {
  it('renders the evolution panel with header', () => {
    render(<EvolutionView {...makeProps()} />)
    expect(screen.getByText('Evolution analytics')).toBeInTheDocument()
  })

  it('renders description text', () => {
    render(<EvolutionView {...makeProps()} />)
    expect(screen.getByText(/Investment vs utilization, ownership, temporal coupling, and fitness/)).toBeInTheDocument()
  })

  it('renders investment vs utilization section', () => {
    render(<EvolutionView {...makeProps()} />)
    expect(screen.getByText('Investment vs utilization')).toBeInTheDocument()
    expect(screen.getByText(/Bubble size = LOC/)).toBeInTheDocument()
  })

  it('renders investment scatter chart', () => {
    render(<EvolutionView {...makeProps()} />)
    expect(screen.getByLabelText('Investment utilization chart')).toBeInTheDocument()
  })

  it('renders investment warnings', () => {
    render(<EvolutionView {...makeProps()} />)
    expect(screen.getByText('Coverage data is partial')).toBeInTheDocument()
  })

  it('renders knowledge islands section', () => {
    render(<EvolutionView {...makeProps()} />)
    expect(screen.getByText('Knowledge islands')).toBeInTheDocument()
    expect(screen.getByText(/Bus-factor and ownership/)).toBeInTheDocument()
  })

  it('renders knowledge islands KPIs', () => {
    render(<EvolutionView {...makeProps()} />)
    expect(screen.getByText('Global bus factor')).toBeInTheDocument()
    expect(screen.getByText('Single-owner files')).toBeInTheDocument()
    expect(screen.getByText('Entities analyzed')).toBeInTheDocument()
  })

  it('renders knowledge islands table', () => {
    render(<EvolutionView {...makeProps()} />)
    expect(screen.getByText('Entity')).toBeInTheDocument()
    expect(screen.getByText('Bus factor')).toBeInTheDocument()
    expect(screen.getByText('Dominant owner')).toBeInTheDocument()
    expect(screen.getByText('Auth')).toBeInTheDocument()
    expect(screen.getByText('alice')).toBeInTheDocument()
  })

  it('renders temporal coupling section with graph', () => {
    render(<EvolutionView {...makeProps()} />)
    expect(screen.getByText('Temporal coupling')).toBeInTheDocument()
    expect(screen.getByText(/Cross-boundary edges are highlighted/)).toBeInTheDocument()
    expect(screen.getByTestId('reactflow')).toBeInTheDocument()
  })

  it('renders fitness functions section', () => {
    render(<EvolutionView {...makeProps()} />)
    expect(screen.getByText('Fitness functions')).toBeInTheDocument()
    expect(screen.getByText(/Persisted advisory findings/)).toBeInTheDocument()
  })

  it('renders fitness rule chips', () => {
    render(<EvolutionView {...makeProps()} />)
    // CYCLIC_DEP appears in both chip and table rows
    expect(screen.getAllByText(/CYCLIC_DEP/).length).toBeGreaterThanOrEqual(1)
  })

  it('renders fitness violations table', () => {
    render(<EvolutionView {...makeProps()} />)
    expect(screen.getByText('Rule')).toBeInTheDocument()
    expect(screen.getByText('Severity')).toBeInTheDocument()
    expect(screen.getByText('Message')).toBeInTheDocument()
    expect(screen.getByText('Cyclic dependency detected')).toBeInTheDocument()
  })

  it('renders empty investment state', () => {
    render(<EvolutionView {...makeProps({ investmentUtilization: { ...investmentPayload, items: [] } })} />)
    expect(screen.getByText('No investment data available')).toBeInTheDocument()
  })

  it('renders empty knowledge islands state', () => {
    render(<EvolutionView {...makeProps({ knowledgeIslands: { ...knowledgePayload, entities: [] } })} />)
    expect(screen.getByText('No ownership data available')).toBeInTheDocument()
  })

  it('renders empty coupling graph state', () => {
    render(<EvolutionView {...makeProps({ temporalCoupling: { ...couplingPayload, graph: { nodes: [], edges: [] } } })} />)
    expect(screen.getByText('No coupling graph available')).toBeInTheDocument()
  })

  it('renders empty fitness state', () => {
    render(<EvolutionView {...makeProps({ fitnessFunctions: { ...fitnessPayload, rules: [], violations: [] } })} />)
    expect(screen.getByText('No fitness findings available')).toBeInTheDocument()
  })

  it('renders panel errors for each section', () => {
    render(<EvolutionView {...makeProps({
      panelErrors: {
        investment: 'Investment API failed',
        knowledge: 'Knowledge API failed',
        coupling: 'Coupling API failed',
        fitness: 'Fitness API failed',
      },
    })} />)
    expect(screen.getByText('Investment API failed')).toBeInTheDocument()
    expect(screen.getByText('Knowledge API failed')).toBeInTheDocument()
    expect(screen.getByText('Coupling API failed')).toBeInTheDocument()
    expect(screen.getByText('Fitness API failed')).toBeInTheDocument()
  })

  it('renders loading skeleton when loading with no data', () => {
    render(<EvolutionView {...makeProps({
      state: 'loading',
      investmentUtilization: null,
      knowledgeIslands: null,
      temporalCoupling: null,
      fitnessFunctions: null,
    })} />)
    expect(screen.queryByText('Evolution analytics')).not.toBeInTheDocument()
  })

  it('renders error state when error with no data', () => {
    render(<EvolutionView {...makeProps({
      state: 'error',
      error: 'Server error',
      investmentUtilization: null,
      knowledgeIslands: null,
      temporalCoupling: null,
      fitnessFunctions: null,
    })} />)
    expect(screen.getByText('Evolution view request failed')).toBeInTheDocument()
  })
})
