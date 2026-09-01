import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Link, MemoryRouter, Route, Routes, useLocation } from 'react-router'
import { describe, expect, it } from 'vitest'

import { legacyPageLocation, routeLocation } from './routing'

function RoutedFixture() {
  const location = useLocation()
  return <>
    <nav><Link to="/collections">Collections</Link><Link to="/runs">Runs</Link></nav>
    <output data-testid="location">{location.pathname}{location.search}</output>
    <Routes>
      <Route path="/" element={<h1>Dashboard</h1>} />
      <Route path="/collections" element={<h1>Collections page</h1>} />
      <Route path="/runs" element={<h1>Runs page</h1>} />
      <Route path="/cockpit" element={<h1>Cockpit page</h1>} />
    </Routes>
  </>
}

describe('application routing', () => {
  it('renders a deep-linked page directly', () => {
    render(<MemoryRouter initialEntries={['/runs']}><RoutedFixture /></MemoryRouter>)
    expect(screen.getByRole('heading', { name: 'Runs page' })).toBeInTheDocument()
  })

  it('updates the rendered page and browser location through navigation', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/']}><RoutedFixture /></MemoryRouter>)
    await user.click(screen.getByRole('link', { name: 'Collections' }))
    expect(screen.getByRole('heading', { name: 'Collections page' })).toBeInTheDocument()
    expect(screen.getByTestId('location')).toHaveTextContent('/collections')
    await user.click(screen.getByRole('link', { name: 'Runs' }))
    expect(screen.getByRole('heading', { name: 'Runs page' })).toBeInTheDocument()
  })

  it('translates a legacy page query exactly once and preserves unrelated params', () => {
    expect(legacyPageLocation('/', '?page=collections&theme=dark')).toEqual({
      pathname: '/collections',
      search: 'theme=dark',
    })
    expect(legacyPageLocation('/collections', '?theme=dark')).toBeNull()
  })

  it('preserves cockpit deep-link state and removes it on another page', () => {
    expect(routeLocation('cockpit', '?query=ports', {
      collectionId: 'c1', scenarioId: 's1', view: 'architecture', layer: 'domain_container',
    })).toEqual({
      pathname: '/cockpit',
      search: 'query=ports&collection=c1&scenario=s1&view=architecture&layer=domain_container',
    })
    expect(routeLocation('runs', '?collection=c1&scenario=s1&view=architecture&theme=dark')).toEqual({
      pathname: '/runs',
      search: 'theme=dark',
    })
  })
})
