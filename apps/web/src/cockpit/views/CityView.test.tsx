/**
 * Tests for CityView rendering.
 */
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import CityView from './CityView'

function makeProps(overrides: Partial<Parameters<typeof CityView>[0]> = {}) {
  return {
    state: 'ready' as const,
    error: '',
    embedUrl: 'https://codecharta.example.com/embed?id=123',
    projection: 'architecture' as const,
    entityLevel: 'domain' as const,
    onProjectionChange: vi.fn(),
    onEntityLevelChange: vi.fn(),
    onReload: vi.fn(),
    onOpenOverview: vi.fn(),
    ...overrides,
  }
}

describe('CityView rendering', () => {
  it('renders the city panel with header', () => {
    render(<CityView {...makeProps()} />)
    expect(screen.getByText('Inline Code City')).toBeInTheDocument()
    expect(screen.getByText(/Explore architecture and file-level maps/)).toBeInTheDocument()
  })

  it('renders projection select', () => {
    render(<CityView {...makeProps()} />)
    expect(screen.getByText('Projection')).toBeInTheDocument()
    expect(screen.getByText('Architecture')).toBeInTheDocument()
    expect(screen.getByText('Code file')).toBeInTheDocument()
  })

  it('renders entity level select', () => {
    render(<CityView {...makeProps()} />)
    expect(screen.getByText('Entity level')).toBeInTheDocument()
    expect(screen.getByText('Domain')).toBeInTheDocument()
    expect(screen.getByText('Container')).toBeInTheDocument()
    expect(screen.getByText('Component')).toBeInTheDocument()
  })

  it('disables entity level select when projection is code_file', () => {
    render(<CityView {...makeProps({ projection: 'code_file' })} />)
    const selects = screen.getAllByRole('combobox')
    // The entity level select is the second one
    const entitySelect = selects.find((s) => s.closest('label')?.textContent?.includes('Entity level'))
    expect(entitySelect).toBeDisabled()
  })

  it('enables entity level select when projection is architecture', () => {
    render(<CityView {...makeProps({ projection: 'architecture' })} />)
    const selects = screen.getAllByRole('combobox')
    const entitySelect = selects.find((s) => s.closest('label')?.textContent?.includes('Entity level'))
    expect(entitySelect).not.toBeDisabled()
  })

  it('renders Reload map button', () => {
    render(<CityView {...makeProps()} />)
    expect(screen.getByText('Reload map')).toBeInTheDocument()
  })

  it('shows Loading text when state is loading', () => {
    render(<CityView {...makeProps({ state: 'loading' })} />)
    expect(screen.getByText('Loading\u2026')).toBeInTheDocument()
  })

  it('disables reload button when loading', () => {
    render(<CityView {...makeProps({ state: 'loading' })} />)
    expect(screen.getByText('Loading\u2026')).toBeDisabled()
  })

  it('renders Open Overview button', () => {
    render(<CityView {...makeProps()} />)
    expect(screen.getByText('Open Overview')).toBeInTheDocument()
  })

  it('renders iframe with embed URL', () => {
    render(<CityView {...makeProps()} />)
    const iframe = screen.getByTitle('CodeCharta city view')
    expect(iframe).toBeInTheDocument()
    expect(iframe).toHaveAttribute('src', 'https://codecharta.example.com/embed?id=123')
  })

  it('renders error message', () => {
    render(<CityView {...makeProps({ error: 'City gen failed' })} />)
    expect(screen.getByText('City gen failed')).toBeInTheDocument()
  })

  it('renders empty state when no embed URL and not loading', () => {
    render(<CityView {...makeProps({ embedUrl: '' })} />)
    expect(screen.getByText('No city view available yet')).toBeInTheDocument()
    expect(screen.getByText('Generate the map from the selected scenario.')).toBeInTheDocument()
  })

  it('renders skeleton when loading with no embed URL', () => {
    render(<CityView {...makeProps({ state: 'loading', embedUrl: '' })} />)
    // Skeleton cards are rendered
    expect(screen.queryByTitle('CodeCharta city view')).not.toBeInTheDocument()
  })

  it('calls onReload when Reload map button is clicked', async () => {
    const onReload = vi.fn()
    render(<CityView {...makeProps({ onReload })} />)
    await userEvent.click(screen.getByText('Reload map'))
    expect(onReload).toHaveBeenCalledOnce()
  })

  it('calls onOpenOverview when Open Overview button is clicked', async () => {
    const onOpenOverview = vi.fn()
    render(<CityView {...makeProps({ onOpenOverview })} />)
    await userEvent.click(screen.getByText('Open Overview'))
    expect(onOpenOverview).toHaveBeenCalledOnce()
  })

  it('calls onProjectionChange when projection is changed', async () => {
    const onProjectionChange = vi.fn()
    render(<CityView {...makeProps({ onProjectionChange })} />)
    const selects = screen.getAllByRole('combobox')
    const projSelect = selects.find((s) => s.closest('label')?.textContent?.includes('Projection'))!
    await userEvent.selectOptions(projSelect, 'code_file')
    expect(onProjectionChange).toHaveBeenCalledWith('code_file')
  })
})
