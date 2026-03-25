import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import CockpitTabs from './CockpitTabs'
import { COCKPIT_VIEWS } from '../types'

describe('CockpitTabs', () => {
  it('renders all view tabs', () => {
    render(<CockpitTabs activeView="overview" onViewChange={vi.fn()} />)
    const tabs = screen.getAllByRole('tab')
    expect(tabs).toHaveLength(COCKPIT_VIEWS.length)
  })

  it('marks the active tab with aria-selected=true', () => {
    render(<CockpitTabs activeView="topology" onViewChange={vi.fn()} />)
    const topologyTab = screen.getByRole('tab', { name: 'Topology' })
    expect(topologyTab).toHaveAttribute('aria-selected', 'true')
  })

  it('marks non-active tabs with aria-selected=false', () => {
    render(<CockpitTabs activeView="topology" onViewChange={vi.fn()} />)
    const overviewTab = screen.getByRole('tab', { name: 'Overview' })
    expect(overviewTab).toHaveAttribute('aria-selected', 'false')
  })

  it('calls onViewChange when a tab is clicked', async () => {
    const onViewChange = vi.fn()
    render(<CockpitTabs activeView="overview" onViewChange={onViewChange} />)
    await userEvent.click(screen.getByRole('tab', { name: 'City' }))
    expect(onViewChange).toHaveBeenCalledWith('city')
  })

  it('renders a tablist with proper aria-label', () => {
    render(<CockpitTabs activeView="overview" onViewChange={vi.fn()} />)
    expect(screen.getByRole('tablist')).toHaveAttribute('aria-label', 'Architecture Cockpit views')
  })
})
