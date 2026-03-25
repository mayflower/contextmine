import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import ViewShell from './ViewShell'

describe('ViewShell', () => {
  it('renders loading skeletons when state=loading and no data', () => {
    const { container } = render(
      <ViewShell
        state="loading"
        error={null}
        panelId="test-panel"
        title="Test"
        hasData={false}
      >
        <p>Content</p>
      </ViewShell>,
    )
    expect(screen.getByRole('tabpanel')).toBeInTheDocument()
    expect(container.querySelectorAll('.cockpit2-skeleton-card')).toHaveLength(3)
    expect(screen.queryByText('Content')).not.toBeInTheDocument()
  })

  it('renders custom skeletonCount', () => {
    const { container } = render(
      <ViewShell
        state="loading"
        error={null}
        panelId="test-panel"
        title="Test"
        hasData={false}
        skeletonCount={5}
      >
        <p>Content</p>
      </ViewShell>,
    )
    expect(container.querySelectorAll('.cockpit2-skeleton-card')).toHaveLength(5)
  })

  it('renders tall skeletons when skeletonTall=true', () => {
    const { container } = render(
      <ViewShell
        state="loading"
        error={null}
        panelId="test-panel"
        title="Test"
        hasData={false}
        skeletonTall
        skeletonCount={2}
      >
        <p>Content</p>
      </ViewShell>,
    )
    expect(container.querySelectorAll('.cockpit2-skeleton-card.tall')).toHaveLength(2)
  })

  it('renders custom skeleton node when provided', () => {
    render(
      <ViewShell
        state="loading"
        error={null}
        panelId="test-panel"
        title="Test"
        hasData={false}
        skeleton={<div data-testid="custom-skeleton">Custom</div>}
      >
        <p>Content</p>
      </ViewShell>,
    )
    expect(screen.getByTestId('custom-skeleton')).toBeInTheDocument()
  })

  it('renders error state with message and retry button', async () => {
    const onRetry = vi.fn()
    render(
      <ViewShell
        state="error"
        error="Something broke"
        panelId="test-panel"
        title="Test"
        hasData={false}
        onRetry={onRetry}
      >
        <p>Content</p>
      </ViewShell>,
    )
    expect(screen.getByText('Test request failed')).toBeInTheDocument()
    expect(screen.getByText('Something broke')).toBeInTheDocument()

    const retryBtn = screen.getByRole('button', { name: 'Retry' })
    await userEvent.click(retryBtn)
    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('renders error state without retry when onRetry not provided', () => {
    render(
      <ViewShell
        state="error"
        error="fail"
        panelId="test-panel"
        title="Test"
        hasData={false}
      >
        <p>Content</p>
      </ViewShell>,
    )
    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument()
  })

  it('renders children when state=ready', () => {
    render(
      <ViewShell
        state="ready"
        error={null}
        panelId="test-panel"
        title="Test"
        hasData={true}
      >
        <p>Visible Content</p>
      </ViewShell>,
    )
    expect(screen.getByText('Visible Content')).toBeInTheDocument()
  })

  it('renders inline error with children when hasData=true and error exists', () => {
    render(
      <ViewShell
        state="error"
        error="Partial error"
        panelId="test-panel"
        title="Test"
        hasData={true}
      >
        <p>Stale Content</p>
      </ViewShell>,
    )
    expect(screen.getByText('Partial error')).toBeInTheDocument()
    expect(screen.getByText('Stale Content')).toBeInTheDocument()
  })

  it('renders children during loading if hasData=true', () => {
    render(
      <ViewShell
        state="loading"
        error={null}
        panelId="test-panel"
        title="Test"
        hasData={true}
      >
        <p>Already loaded content</p>
      </ViewShell>,
    )
    expect(screen.getByText('Already loaded content')).toBeInTheDocument()
  })
})
