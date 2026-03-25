import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import RebuildReadinessView from './RebuildReadinessView'
import type { RebuildReadinessPayload, ViewScenario } from '../types'

const scenario: ViewScenario = {
  id: 's1',
  collection_id: 'c1',
  name: 'Main',
  version: 1,
  is_as_is: true,
  base_scenario_id: null,
}

function makePayload(overrides?: Partial<RebuildReadinessPayload>): RebuildReadinessPayload {
  return {
    collection_id: 'c1',
    scenario,
    projection: 'rebuild_readiness',
    score: 72,
    summary: {
      interface_test_coverage: 0.85,
      flow_evidence_density: 0.6,
      ui_to_endpoint_traceability: 0.7,
      critical_inferred_only_count: 3,
      total_nodes: 100,
      total_edges: 200,
    },
    known_gaps: ['Missing API docs'],
    critical_inferred_only: [],
    evidence_handles: [],
    behavioral_layers_status: 'complete',
    last_behavioral_materialized_at: null,
    deep_warnings: [],
    ...overrides,
  }
}

describe('RebuildReadinessView', () => {
  const defaultProps = {
    state: 'ready' as const,
    error: '',
    onRetry: vi.fn(),
  }

  it('renders heading and score when payload is available', () => {
    render(<RebuildReadinessView {...defaultProps} payload={makePayload()} />)
    expect(screen.getByText('Rebuild readiness')).toBeInTheDocument()
    expect(screen.getByText('72')).toBeInTheDocument()
    expect(screen.getByText('Overall score')).toBeInTheDocument()
  })

  it('renders percentage values from summary', () => {
    render(<RebuildReadinessView {...defaultProps} payload={makePayload()} />)
    expect(screen.getByText('85%')).toBeInTheDocument()
    expect(screen.getByText('70%')).toBeInTheDocument()
  })

  it('renders loading skeleton when loading with no data', () => {
    const { container } = render(
      <RebuildReadinessView {...defaultProps} payload={null} state="loading" />,
    )
    expect(container.querySelectorAll('.cockpit2-skeleton-card')).toHaveLength(2)
  })

  it('renders error state when error with no data', () => {
    render(
      <RebuildReadinessView {...defaultProps} payload={null} state="error" error="Failed" />,
    )
    expect(screen.getByText('Readiness request failed')).toBeInTheDocument()
  })

  it('shows behavioral layer status', () => {
    render(<RebuildReadinessView {...defaultProps} payload={makePayload()} />)
    expect(screen.getByText(/complete/)).toBeInTheDocument()
  })

  it('shows SCIP status', () => {
    render(
      <RebuildReadinessView
        {...defaultProps}
        payload={makePayload({ scip_status: 'ready' })}
      />,
    )
    expect(screen.getByText(/SCIP: ready/)).toBeInTheDocument()
  })
})
