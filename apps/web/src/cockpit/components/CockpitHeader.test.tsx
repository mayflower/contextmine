import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import CockpitHeader from './CockpitHeader'

describe('CockpitHeader', () => {
  const defaultProps = {
    selection: {
      collectionId: 'col-1',
      scenarioId: 'sc-1',
      layer: 'code_controlflow' as const,
      view: 'overview' as const,
    },
    scenario: null,
    activeState: 'idle' as const,
    activeUpdatedAt: null,
  }

  it('renders the Architecture Cockpit heading', () => {
    render(<CockpitHeader {...defaultProps} />)
    expect(screen.getByText('Architecture Cockpit')).toBeInTheDocument()
  })

  it('displays view chip with formatted view name', () => {
    render(<CockpitHeader {...defaultProps} />)
    expect(screen.getByText('View: overview')).toBeInTheDocument()
  })

  it('shows "Not selected" when no scenario', () => {
    render(<CockpitHeader {...defaultProps} scenario={null} />)
    expect(screen.getByText('Scenario: Not selected')).toBeInTheDocument()
  })

  it('shows scenario name when provided', () => {
    render(
      <CockpitHeader
        {...defaultProps}
        scenario={{ id: 'sc-1', name: 'Test Scenario', version: 2, is_as_is: true }}
      />,
    )
    expect(screen.getByText('Scenario: Test Scenario')).toBeInTheDocument()
    expect(screen.getByText('Version: v2')).toBeInTheDocument()
    expect(screen.getByText('Mode: AS-IS')).toBeInTheDocument()
  })

  it('shows TO-BE for non-as-is scenarios', () => {
    render(
      <CockpitHeader
        {...defaultProps}
        scenario={{ id: 'sc-1', name: 'New', version: 1, is_as_is: false }}
      />,
    )
    expect(screen.getByText('Mode: TO-BE')).toBeInTheDocument()
  })

  it('displays Idle status', () => {
    render(<CockpitHeader {...defaultProps} />)
    expect(screen.getByText('Status: Idle')).toBeInTheDocument()
  })

  it('displays Loading status', () => {
    render(<CockpitHeader {...defaultProps} activeState="loading" />)
    expect(screen.getByText('Status: Loading')).toBeInTheDocument()
  })

  it('displays Ready status', () => {
    render(<CockpitHeader {...defaultProps} activeState="ready" />)
    expect(screen.getByText('Status: Ready')).toBeInTheDocument()
  })

  it('displays Error status', () => {
    render(<CockpitHeader {...defaultProps} activeState="error" />)
    expect(screen.getByText('Status: Error')).toBeInTheDocument()
  })

  it('displays Empty status', () => {
    render(<CockpitHeader {...defaultProps} activeState="empty" />)
    expect(screen.getByText('Status: Empty')).toBeInTheDocument()
  })

  it('displays "Not loaded yet" when no updatedAt', () => {
    render(<CockpitHeader {...defaultProps} />)
    expect(screen.getByText('Updated: Not loaded yet')).toBeInTheDocument()
  })
})
