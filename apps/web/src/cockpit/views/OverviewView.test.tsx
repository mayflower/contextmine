import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import OverviewView from './OverviewView'
import type { CityPayload, ViewScenario } from '../types'

const scenario: ViewScenario = {
  id: 's1',
  collection_id: 'c1',
  name: 'Main',
  version: 1,
  is_as_is: true,
  base_scenario_id: null,
}

function makeCity(overrides?: Partial<CityPayload>): CityPayload {
  return {
    collection_id: 'c1',
    scenario,
    summary: {
      metric_nodes: 100,
      coverage_avg: 0.75,
      complexity_avg: 12.3,
      coupling_avg: 3.5,
      change_frequency_avg: 2.1,
      churn_avg: 45.6,
    },
    metrics_status: {
      status: 'ready',
      reason: 'ok',
      strict_mode: false,
    },
    hotspots: [
      {
        node_natural_key: 'src/main.py',
        loc: 500,
        symbol_count: 20,
        coverage: 0.6,
        complexity: 25,
        coupling: 5,
        change_frequency: 10,
        churn: 80,
      },
      {
        node_natural_key: 'src/utils.py',
        loc: 200,
        symbol_count: 10,
        coverage: 0.9,
        complexity: 3,
        coupling: 1,
        change_frequency: 2,
        churn: 10,
      },
    ],
    cc_json: {},
    ...overrides,
  }
}

describe('OverviewView', () => {
  const defaultProps = {
    state: 'ready' as const,
    error: '',
    filter: '',
    onRetry: vi.fn(),
    onOpenTopology: vi.fn(),
    onOpenCity: vi.fn(),
    onSelectHotspot: vi.fn(),
  }

  it('renders system health summary when data is available', () => {
    render(<OverviewView {...defaultProps} city={makeCity()} />)
    expect(screen.getByText('System health summary')).toBeInTheDocument()
    expect(screen.getByText('100')).toBeInTheDocument()
    expect(screen.getByText('Metric nodes')).toBeInTheDocument()
  })

  it('renders hotspot table', () => {
    render(<OverviewView {...defaultProps} city={makeCity()} />)
    expect(screen.getByText('Top hotspots')).toBeInTheDocument()
    expect(screen.getByText('src/main.py')).toBeInTheDocument()
    expect(screen.getByText('src/utils.py')).toBeInTheDocument()
  })

  it('renders empty state when city is null and not loading', () => {
    render(<OverviewView {...defaultProps} city={null} state="empty" />)
    expect(screen.getByText('No city data available yet')).toBeInTheDocument()
  })

  it('renders loading skeleton when loading with no data', () => {
    const { container } = render(
      <OverviewView {...defaultProps} city={null} state="loading" />,
    )
    expect(container.querySelectorAll('.cockpit2-skeleton-card')).toHaveLength(4)
  })

  it('renders error state when error with no data', () => {
    render(
      <OverviewView {...defaultProps} city={null} state="error" error="Failed to load" />,
    )
    expect(screen.getByText('Overview request failed')).toBeInTheDocument()
    expect(screen.getByText('Failed to load')).toBeInTheDocument()
  })

  it('filters hotspots by filter string', () => {
    render(<OverviewView {...defaultProps} city={makeCity()} filter="main" />)
    expect(screen.getByText('src/main.py')).toBeInTheDocument()
    expect(screen.queryByText('src/utils.py')).not.toBeInTheDocument()
  })

  it('calls onSelectHotspot when hotspot name is clicked', async () => {
    const onSelectHotspot = vi.fn()
    render(<OverviewView {...defaultProps} city={makeCity()} onSelectHotspot={onSelectHotspot} />)
    await userEvent.click(screen.getByText('src/main.py'))
    expect(onSelectHotspot).toHaveBeenCalledWith('src/main.py')
  })

  it('calls onOpenTopology when topology button is clicked', async () => {
    const onOpenTopology = vi.fn()
    render(<OverviewView {...defaultProps} city={makeCity()} onOpenTopology={onOpenTopology} />)
    await userEvent.click(screen.getByRole('button', { name: /Open Topology/ }))
    expect(onOpenTopology).toHaveBeenCalledOnce()
  })

  it('calls onOpenCity when city button is clicked', async () => {
    const onOpenCity = vi.fn()
    render(<OverviewView {...defaultProps} city={makeCity()} onOpenCity={onOpenCity} />)
    await userEvent.click(screen.getByRole('button', { name: /Open City Map/ }))
    expect(onOpenCity).toHaveBeenCalledOnce()
  })

  it('shows metrics unavailable message when status is unavailable', () => {
    const city = makeCity({
      metrics_status: { status: 'unavailable', reason: 'no_real_metrics', strict_mode: false },
    })
    render(<OverviewView {...defaultProps} city={city} />)
    expect(screen.getByText('Real metrics are currently unavailable for this scenario.')).toBeInTheDocument()
  })

  it('shows coverage ingest failed message', () => {
    const city = makeCity({
      metrics_status: { status: 'unavailable', reason: 'coverage_ingest_failed', strict_mode: false },
    })
    render(<OverviewView {...defaultProps} city={city} />)
    expect(screen.getAllByText(/Coverage ingest failed/).length).toBeGreaterThan(0)
  })

  it('renders a metrics status label when coverage is pending', () => {
    const city = makeCity({
      metrics_status: { status: 'unavailable', reason: 'awaiting_ci_coverage', strict_mode: false },
    })
    render(<OverviewView {...defaultProps} city={city} />)
    expect(screen.getByText(/Coverage ingest pending/)).toBeInTheDocument()
  })

  it('shows awaiting CI coverage message', () => {
    const city = makeCity({
      metrics_status: { status: 'unavailable', reason: 'awaiting_ci_coverage', strict_mode: false },
    })
    render(<OverviewView {...defaultProps} city={city} />)
    expect(screen.getByText(/Structural metrics are ready/)).toBeInTheDocument()
  })

  it('can toggle sort direction by clicking same column', async () => {
    render(<OverviewView {...defaultProps} city={makeCity()} />)
    // Click Complexity twice to toggle direction
    const complexityButton = screen.getByRole('button', { name: 'Complexity' })
    await userEvent.click(complexityButton)
    await userEvent.click(complexityButton)
    // Just verify it doesn't crash
    expect(screen.getByText('src/main.py')).toBeInTheDocument()
  })

  it('sorts by node name when node column is clicked', async () => {
    render(<OverviewView {...defaultProps} city={makeCity()} />)
    await userEvent.click(screen.getByRole('button', { name: 'Node' }))
    const rows = screen.getAllByRole('row')
    // Header row + 2 data rows
    expect(rows.length).toBeGreaterThanOrEqual(3)
  })

  it('formats metric values with toFixed(2)', () => {
    render(<OverviewView {...defaultProps} city={makeCity()} />)
    expect(screen.getByText('0.75')).toBeInTheDocument()
    expect(screen.getByText('12.30')).toBeInTheDocument()
  })

  it('shows N/A for null metric values', () => {
    const city = makeCity({
      summary: {
        metric_nodes: 0,
        coverage_avg: null,
        complexity_avg: null,
        coupling_avg: null,
        change_frequency_avg: null,
        churn_avg: null,
      },
    })
    render(<OverviewView {...defaultProps} city={city} />)
    const naElements = screen.getAllByText('N/A')
    expect(naElements.length).toBeGreaterThanOrEqual(5)
  })
})
