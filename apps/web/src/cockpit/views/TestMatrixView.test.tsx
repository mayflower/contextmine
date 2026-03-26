/**
 * Tests for TestMatrixView rendering.
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { TestMatrixPayload, ViewScenario } from '../types'

import TestMatrixView from './TestMatrixView'

const mockScenario: ViewScenario = {
  id: 's1',
  collection_id: 'c1',
  name: 'Base',
  version: 1,
  is_as_is: true,
  base_scenario_id: null,
}

const populatedPayload: TestMatrixPayload = {
  collection_id: 'c1',
  scenario: mockScenario,
  projection: 'test_matrix',
  entity_level: 'test_case',
  summary: {
    test_cases: 3,
    test_suites: 2,
    test_fixtures: 1,
    matrix_rows: 3,
  },
  matrix: [
    {
      test_case_id: 'tc1',
      test_case_key: 'test_auth_login',
      test_case_name: 'test_auth_login',
      covers_symbols: ['authenticate', 'validate_token'],
      validates_rules: ['AUTH_REQUIRED'],
      verifies_flows: ['login_flow'],
      fixtures: ['test_user'],
      evidence_ids: ['ev1'],
    },
    {
      test_case_id: 'tc2',
      test_case_key: 'test_billing_charge',
      test_case_name: 'test_billing_charge',
      covers_symbols: ['create_charge', 'apply_discount'],
      validates_rules: [],
      verifies_flows: [],
      fixtures: ['test_customer', 'mock_payment'],
      evidence_ids: ['ev2'],
    },
    {
      test_case_id: 'tc3',
      test_case_key: 'test_healthcheck',
      test_case_name: 'test_healthcheck',
      covers_symbols: [],
      validates_rules: [],
      verifies_flows: [],
      fixtures: [],
      evidence_ids: [],
    },
  ],
  warnings: [],
  graph: { nodes: [], edges: [], page: 0, limit: 0, total_nodes: 0 },
}

function makeProps(overrides: Partial<Parameters<typeof TestMatrixView>[0]> = {}) {
  return {
    state: 'ready' as const,
    error: '',
    payload: populatedPayload,
    onRetry: vi.fn(),
    ...overrides,
  }
}

describe('TestMatrixView rendering', () => {
  it('renders the test matrix panel with header', () => {
    render(<TestMatrixView {...makeProps()} />)
    expect(screen.getByText('Test matrix')).toBeInTheDocument()
  })

  it('displays test case and suite counts', () => {
    render(<TestMatrixView {...makeProps()} />)
    expect(screen.getByText(/Cases: 3/)).toBeInTheDocument()
    expect(screen.getByText(/Suites: 2/)).toBeInTheDocument()
  })

  it('renders table with column headers', () => {
    render(<TestMatrixView {...makeProps()} />)
    expect(screen.getByText('Test case')).toBeInTheDocument()
    expect(screen.getByText('Symbols')).toBeInTheDocument()
    expect(screen.getByText('Rules')).toBeInTheDocument()
    expect(screen.getByText('Flows')).toBeInTheDocument()
    expect(screen.getByText('Fixtures')).toBeInTheDocument()
  })

  it('renders test case names in table', () => {
    render(<TestMatrixView {...makeProps()} />)
    expect(screen.getByText('test_auth_login')).toBeInTheDocument()
    expect(screen.getByText('test_billing_charge')).toBeInTheDocument()
    expect(screen.getByText('test_healthcheck')).toBeInTheDocument()
  })

  it('renders symbol names in table cells', () => {
    render(<TestMatrixView {...makeProps()} />)
    expect(screen.getByText(/authenticate, validate_token/)).toBeInTheDocument()
    expect(screen.getByText(/create_charge, apply_discount/)).toBeInTheDocument()
  })

  it('renders rules in table cells', () => {
    render(<TestMatrixView {...makeProps()} />)
    expect(screen.getByText('AUTH_REQUIRED')).toBeInTheDocument()
  })

  it('renders fixtures in table cells', () => {
    render(<TestMatrixView {...makeProps()} />)
    expect(screen.getByText('test_user')).toBeInTheDocument()
    expect(screen.getByText(/test_customer, mock_payment/)).toBeInTheDocument()
  })

  it('renders dash for empty symbol/rule/flow/fixture cells', () => {
    render(<TestMatrixView {...makeProps()} />)
    // test_healthcheck has empty arrays for all
    const rows = screen.getAllByRole('row')
    const healthcheckRow = rows.find((row) => row.textContent?.includes('test_healthcheck'))
    expect(healthcheckRow?.textContent).toContain('\u2014')
  })

  it('renders empty state when no matrix rows', () => {
    render(<TestMatrixView {...makeProps({ payload: { ...populatedPayload, matrix: [] } })} />)
    expect(screen.getByText('No test matrix rows')).toBeInTheDocument()
    expect(screen.getByText(/Behavioral extraction may still be pending/)).toBeInTheDocument()
  })

  it('renders loading skeleton when loading with no data', () => {
    render(<TestMatrixView {...makeProps({ state: 'loading', payload: null })} />)
    expect(screen.queryByText('Test matrix')).not.toBeInTheDocument()
  })

  it('renders error state when error with no data', () => {
    render(<TestMatrixView {...makeProps({ state: 'error', error: 'API error', payload: null })} />)
    expect(screen.getByText('Test matrix request failed')).toBeInTheDocument()
    expect(screen.getByText('API error')).toBeInTheDocument()
  })

  it('shows zero counts when payload has zero summary', () => {
    const zeroPayload = {
      ...populatedPayload,
      summary: { test_cases: 0, test_suites: 0, test_fixtures: 0, matrix_rows: 0 },
      matrix: [],
    }
    render(<TestMatrixView {...makeProps({ payload: zeroPayload })} />)
    expect(screen.getByText(/Cases: 0/)).toBeInTheDocument()
    expect(screen.getByText(/Suites: 0/)).toBeInTheDocument()
  })
})
