import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import ExportsView from './ExportsView'

describe('ExportsView', () => {
  const defaultProps = {
    exportFormat: 'cc_json' as const,
    exportProjection: 'architecture' as const,
    exportState: 'ready' as const,
    exportError: '',
    exportContent: '{"test": true}',
    onFormatChange: vi.fn(),
    onProjectionChange: vi.fn(),
    onGenerate: vi.fn(),
    onCopy: vi.fn(),
    onDownload: vi.fn(),
  }

  it('renders the exports heading', () => {
    render(<ExportsView {...defaultProps} />)
    expect(screen.getByText('Visualization exports')).toBeInTheDocument()
  })

  it('renders format select with all export formats', () => {
    render(<ExportsView {...defaultProps} />)
    expect(screen.getByText('Format')).toBeInTheDocument()
    expect(screen.getByText('CodeCharta (cc.json)')).toBeInTheDocument()
    expect(screen.getByText('Mermaid C4')).toBeInTheDocument()
  })

  it('renders projection select', () => {
    render(<ExportsView {...defaultProps} />)
    expect(screen.getByText('Projection')).toBeInTheDocument()
  })

  it('has tabpanel role', () => {
    render(<ExportsView {...defaultProps} />)
    expect(screen.getByRole('tabpanel')).toBeInTheDocument()
  })
})
