import { lazy, Suspense, useEffect, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Navigate, NavLink, Route, Routes, useLocation, useNavigate } from 'react-router'

import './App.css'
import { api, apiData } from './api/client'
import CockpitPage from './cockpit/CockpitPage'
import { legacyPageLocation, routeLocation } from './routing'

const DashboardPage = lazy(() => import('./features/dashboard/DashboardPage'))
const CollectionsPage = lazy(() => import('./features/collections/CollectionsPage'))
const RunsPage = lazy(() => import('./features/runs/RunsPage'))
const GITHUB_REPO = 'https://github.com/mayflower/contextmine'

function CockpitRoute() {
  const navigate = useNavigate()
  const location = useLocation()
  const collectionsQuery = useQuery({
    queryKey: ['collections'],
    queryFn: ({ signal }) => apiData(api.GET('/api/collections', { signal })),
  })
  return <CockpitPage
    collections={(collectionsQuery.data ?? []).map((collection) => ({ id: collection.id, name: collection.name }))}
    onOpenCollections={() => navigate(routeLocation('collections', location.search))}
    onOpenRuns={() => navigate(routeLocation('runs', location.search))}
  />
}

function LoginPage() {
  return <div className="app login-page"><div className="login-container">
    <img src="/logo-md.png" alt="ContextMine" className="login-logo" />
    <h1>ContextMine</h1>
    <p className="login-subtitle">Documentation & Code Indexing with MCP</p>
    <button className="login-button" onClick={() => { globalThis.location.href = '/api/auth/login' }}>
      <svg viewBox="0 0 16 16" width="20" height="20" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" /></svg>
      Sign in with GitHub
    </button>
  </div></div>
}

function App() {
  const location = useLocation()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const authQuery = useQuery({
    queryKey: ['auth', 'me'],
    queryFn: ({ signal }) => apiData(api.GET('/api/auth/me', { signal })),
    retry: false,
  })
  const user = authQuery.data

  useEffect(() => {
    const legacy = legacyPageLocation(location.pathname, location.search)
    if (legacy) navigate(legacy, { replace: true })
  }, [location.pathname, location.search, navigate])

  if (authQuery.isLoading) return null
  if (!user) return <LoginPage />

  const logout = async () => {
    await apiData(api.GET('/api/auth/logout'))
    queryClient.setQueryData(['auth', 'me'], null)
  }
  const closeMenu = () => setMobileMenuOpen(false)

  return <div className="app">
    <header className="header">
      <div className="header-left">
        <button className="mobile-menu-toggle" onClick={() => setMobileMenuOpen((open) => !open)} aria-label="Toggle menu"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">{mobileMenuOpen ? <path d="M18 6L6 18M6 6l12 12" /> : <path d="M4 6h16M4 12h16M4 18h16" />}</svg></button>
        <img src="/logo-dark-sm.png" alt="ContextMine" className="header-logo" />
        <h1>ContextMine</h1>
        <button className="header-cta" onClick={() => { navigate(routeLocation('collections', location.search)); closeMenu() }}><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14" /></svg>Add Source</button>
      </div>
      <div className="header-right"><div className="user-info">{user.avatar_url && <img src={user.avatar_url} alt={user.github_login} className="avatar" />}<span className="username">{user.name || user.github_login}</span></div><button className="logout-button" onClick={logout}>Logout</button></div>
    </header>

    <nav className={`sidebar ${mobileMenuOpen ? 'open' : ''}`}><ul>
      <li><NavLink to="/" end onClick={closeMenu}>Dashboard</NavLink></li>
      <li><NavLink to={routeLocation('cockpit', location.search)} onClick={closeMenu}>Architecture Cockpit</NavLink></li>
      <li><NavLink to="/collections" onClick={closeMenu}>Collections</NavLink></li>
      <li><NavLink to="/runs" onClick={closeMenu}>Runs</NavLink></li>
    </ul></nav>

    <main className="content"><Suspense fallback={<p className="loading-text">Loading...</p>}><Routes>
      <Route path="/" element={<DashboardPage />} />
      <Route path="/collections" element={<CollectionsPage />} />
      <Route path="/runs" element={<RunsPage />} />
      <Route path="/cockpit" element={<CockpitRoute />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes></Suspense></main>

    <footer className="footer"><div className="footer-left"><span>ContextMine by</span><a href="https://mayflower.de" target="_blank" rel="noopener noreferrer">Mayflower</a></div><div className="footer-links"><a href={GITHUB_REPO} target="_blank" rel="noopener noreferrer">GitHub</a><a href={`${GITHUB_REPO}/blob/main/README.md`} target="_blank" rel="noopener noreferrer">Docs</a><a href={`${GITHUB_REPO}/issues`} target="_blank" rel="noopener noreferrer">Issues</a></div></footer>
  </div>
}

export default App
