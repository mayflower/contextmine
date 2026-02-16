import { useEffect, useState } from 'react'
import './App.css'
import CockpitPage from './cockpit/CockpitPage'
import type { CockpitLayer, CockpitView } from './cockpit/types'

interface HealthStatus {
  status: string
}

interface User {
  id: string
  github_login: string
  name: string | null
  avatar_url: string | null
}

interface Collection {
  id: string
  slug: string
  name: string
  visibility: 'global' | 'private'
  owner_id: string
  owner_github_login: string
  created_at: string
  is_owner: boolean
  member_count: number
}

interface CollectionMember {
  user_id: string
  github_login: string
  name: string | null
  avatar_url: string | null
  is_owner: boolean
}

interface CollectionInvite {
  github_login: string
  created_at: string
}

interface Source {
  id: string
  collection_id: string
  type: 'github' | 'web'
  url: string
  config: Record<string, string>
  enabled: boolean
  schedule_interval_minutes: number
  next_run_at: string | null
  last_run_at: string | null
  created_at: string
  document_count: number
  deploy_key_fingerprint: string | null
}

interface SyncRunStats {
  // GitHub sync stats
  files_scanned?: number
  files_indexed?: number
  files_skipped?: number
  files_deleted?: number
  // Web sync stats
  pages_crawled?: number
  pages_skipped?: number
  // Common stats
  docs_created?: number
  docs_updated?: number
  docs_deleted?: number
  chunks_created?: number
  chunks_deleted?: number
  chunks_embedded?: number
  embedding_tokens_used?: number
  // GitHub specific
  commit_sha?: string
  previous_sha?: string | null
}

interface SyncRun {
  id: string
  source_id: string
  started_at: string
  finished_at: string | null
  status: 'running' | 'success' | 'failed'
  stats: SyncRunStats | null
  error: string | null
}

type Page = 'dashboard' | 'collections' | 'runs' | 'cockpit'

const GITHUB_REPO = 'https://github.com/mayflower/contextmine'
const VALID_PAGES: Page[] = ['dashboard', 'collections', 'runs', 'cockpit']
const DEFAULT_COCKPIT_VIEW: CockpitView = 'overview'
const DEFAULT_COCKPIT_LAYER: CockpitLayer = 'domain_container'

interface CockpitNavigationOptions {
  collectionId?: string
  scenarioId?: string
  view?: CockpitView
  layer?: CockpitLayer
}

function parseInitialPage(): Page {
  if (typeof window === 'undefined') {
    return 'dashboard'
  }

  const params = new URLSearchParams(window.location.search)
  const rawPage = params.get('page')
  if (!rawPage) {
    return 'dashboard'
  }

  return VALID_PAGES.includes(rawPage as Page) ? (rawPage as Page) : 'dashboard'
}

function updatePageQuery(page: Page, cockpitOptions?: CockpitNavigationOptions): void {
  if (typeof window === 'undefined') {
    return
  }

  const params = new URLSearchParams(window.location.search)
  params.set('page', page)

  if (page === 'cockpit') {
    const nextView = cockpitOptions?.view
    const nextLayer = cockpitOptions?.layer

    if (cockpitOptions?.collectionId) {
      params.set('collection', cockpitOptions.collectionId)
    } else if (cockpitOptions && !cockpitOptions.collectionId) {
      params.delete('collection')
    }

    if (cockpitOptions?.scenarioId) {
      params.set('scenario', cockpitOptions.scenarioId)
    } else if (cockpitOptions && !cockpitOptions.scenarioId) {
      params.delete('scenario')
    }

    if (nextView) {
      params.set('view', nextView)
    } else if (!params.get('view')) {
      params.set('view', DEFAULT_COCKPIT_VIEW)
    }

    if (nextLayer) {
      params.set('layer', nextLayer)
    } else if (!params.get('layer')) {
      params.set('layer', DEFAULT_COCKPIT_LAYER)
    }
  } else {
    params.delete('collection')
    params.delete('scenario')
    params.delete('view')
    params.delete('layer')
  }

  const nextQuery = params.toString()
  const nextUrl = nextQuery ? `${window.location.pathname}?${nextQuery}` : window.location.pathname
  window.history.replaceState({}, '', nextUrl)
}

/**
 * Format a source URL for display.
 * GitHub URLs: owner/repo
 * Web docs: hostname/path (e.g., langchain-ai.github.io/langgraph)
 */
function formatSourceUrl(url: string): string {
  if (url.startsWith('https://github.com/')) {
    return url.replace('https://github.com/', '').split('/').slice(0, 2).join('/')
  }
  try {
    const parsed = new URL(url)
    const path = parsed.pathname.replace(/\/$/, '') // Remove trailing slash
    if (path && path !== '/') {
      return parsed.hostname + path
    }
    return parsed.hostname
  } catch {
    return url
  }
}

interface ContextSource {
  uri: string
  title: string
  file_path: string | null
}

interface ContextResult {
  markdown: string
  query: string
  chunks_used: number
  sources: ContextSource[]
}

interface DashboardStats {
  collections: number
  sources: number
  documents: number
  chunks: number
  embedded_chunks: number
  runs_by_status: Record<string, number>
  recent_runs: {
    id: string
    status: string
    started_at: string | null
    finished_at: string | null
    source_url: string
  }[]
}

interface FlowRunProgress {
  total: number
  completed: number
  failed: number
  running: number
  pending: number
  current_task: string | null
  percent: number
}

interface FlowRun {
  id: string
  name: string
  flow_id: string
  state_type: string
  state_name: string
  start_time: string | null
  end_time: string | null
  parameters: {
    source_id?: string
    source_url?: string
  }
  total_run_time: number | null
  progress?: FlowRunProgress
}

interface PrefectFlowRuns {
  active: FlowRun[]
  recent: FlowRun[]
  error?: string
}

function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [user, setUser] = useState<User | null>(null)
  const [authLoading, setAuthLoading] = useState(true)
  const [currentPage, setCurrentPage] = useState<Page>(() => parseInitialPage())

  // Collections state
  const [collections, setCollections] = useState<Collection[]>([])
  const [collectionsLoading, setCollectionsLoading] = useState(false)
  const [newCollectionName, setNewCollectionName] = useState('')
  const [newCollectionSlug, setNewCollectionSlug] = useState('')
  const [newCollectionVisibility, setNewCollectionVisibility] = useState<'global' | 'private'>('private')
  const [selectedCollection, setSelectedCollection] = useState<Collection | null>(null)
  const [collectionMembers, setCollectionMembers] = useState<CollectionMember[]>([])
  const [collectionInvites, setCollectionInvites] = useState<CollectionInvite[]>([])
  const [shareGithubLogin, setShareGithubLogin] = useState('')
  const [shareError, setShareError] = useState<string | null>(null)
  // New UX state
  const [expandedCollections, setExpandedCollections] = useState<Set<string>>(new Set())
  const [collectionSources, setCollectionSources] = useState<Record<string, Source[]>>({})
  const [collectionSourcesLoading, setCollectionSourcesLoading] = useState<Set<string>>(new Set())
  const [sharePopoverCollection, setSharePopoverCollection] = useState<Collection | null>(null)
  const [editingCollection, setEditingCollection] = useState<Collection | null>(null)
  const [editCollectionName, setEditCollectionName] = useState('')
  const [editCollectionVisibility, setEditCollectionVisibility] = useState<'global' | 'private'>('private')
  const [editCollectionLoading, setEditCollectionLoading] = useState(false)

  // Sources state
  const [sources, setSources] = useState<Source[]>([])
  const [newSourceType, setNewSourceType] = useState<'github' | 'web'>('github')
  const [newSourceUrl, setNewSourceUrl] = useState('')
  const [newSourceEnabled, setNewSourceEnabled] = useState(true)
  const [newSourceInterval, setNewSourceInterval] = useState(60)
  const [sourceError, setSourceError] = useState<string | null>(null)
  // Deploy key state
  const [selectedSource, setSelectedSource] = useState<Source | null>(null)
  const [deployKeyInput, setDeployKeyInput] = useState('')
  const [deployKeyError, setDeployKeyError] = useState<string | null>(null)
  const [deployKeyLoading, setDeployKeyLoading] = useState(false)
  // Edit source state
  const [editingSource, setEditingSource] = useState<Source | null>(null)
  const [editSourceEnabled, setEditSourceEnabled] = useState(true)
  const [editSourceInterval, setEditSourceInterval] = useState(60)
  const [editSourceMaxPages, setEditSourceMaxPages] = useState(100)
  const [editSourceError, setEditSourceError] = useState<string | null>(null)
  const [editSourceLoading, setEditSourceLoading] = useState(false)
  // Sync status state
  const [syncingSources, setSyncingSources] = useState<Set<string>>(new Set())

  // Runs state
  const [runsCollection, setRunsCollection] = useState<Collection | null>(null)
  const [runsSources, setRunsSources] = useState<Source[]>([])
  const [selectedRunSource, setSelectedRunSource] = useState<Source | null>(null)
  const [runs, setRuns] = useState<SyncRun[]>([])
  const [runsLoading, setRunsLoading] = useState(false)
  const [prefectFlowRuns, setPrefectFlowRuns] = useState<PrefectFlowRuns | null>(null)

  // Query state
  const [queryText, setQueryText] = useState('')
  const [queryCollection, setQueryCollection] = useState<Collection | null>(null)
  const [queryResult, setQueryResult] = useState<ContextResult | null>(null)
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryError, setQueryError] = useState<string | null>(null)
  const [queryMode, setQueryMode] = useState<'quick' | 'deep'>('quick')
  // Deep research state
  const [researchStep, setResearchStep] = useState<string | null>(null)
  const [researchCitations, setResearchCitations] = useState<string[]>([])
  const [researchRunId, setResearchRunId] = useState<string | null>(null)

  // Mobile menu state
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  // Check authentication status
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await fetch('/api/auth/me', { credentials: 'include' })
        if (response.ok) {
          const userData = await response.json()
          setUser(userData)
        }
      } catch {
        // Not authenticated
      } finally {
        setAuthLoading(false)
      }
    }
    checkAuth()
  }, [])

  useEffect(() => {
    const onPopState = () => {
      setCurrentPage(parseInitialPage())
    }

    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  // Check health status
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const response = await fetch('/api/health')
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        const data = await response.json()
        setHealth(data)
        setHealthError(null)
      } catch (err) {
        setHealthError(err instanceof Error ? err.message : 'Unknown error')
        setHealth(null)
      } finally {
        setLoading(false)
      }
    }

    fetchHealth()
    const interval = setInterval(fetchHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  // Fetch dashboard stats
  const fetchStats = async () => {
    try {
      const response = await fetch('/api/stats')
      if (response.ok) {
        const data = await response.json()
        if (!data.error) {
          setStats(data)
        }
      }
    } catch {
      // Silently fail - stats are optional
    }
  }

  // Load stats when switching to Dashboard page
  useEffect(() => {
    if (currentPage === 'dashboard' && user) {
      fetchStats()
    }
  }, [currentPage, user])

  const handleLogin = () => {
    window.location.href = '/api/auth/login'
  }

  const handleLogout = async () => {
    await fetch('/api/auth/logout', { credentials: 'include' })
    setUser(null)
  }

  // Fetch collections
  const fetchCollections = async () => {
    setCollectionsLoading(true)
    try {
      const response = await fetch('/api/collections', { credentials: 'include' })
      if (response.ok) {
        const data = await response.json()
        setCollections(data)
      }
    } catch {
      // Error fetching collections
    } finally {
      setCollectionsLoading(false)
    }
  }

  // Load collections when switching to Collections page
  useEffect(() => {
    if (currentPage === 'collections' && user) {
      fetchCollections()
    }
  }, [currentPage, user])

  // Fetch collection members, invites, and sources
  const fetchCollectionDetails = async (collection: Collection) => {
    try {
      const membersRes = await fetch(`/api/collections/${collection.id}/members`, { credentials: 'include' })
      if (membersRes.ok) {
        const members = await membersRes.json()
        setCollectionMembers(members)
      }

      if (collection.is_owner) {
        const invitesRes = await fetch(`/api/collections/${collection.id}/invites`, { credentials: 'include' })
        if (invitesRes.ok) {
          const invites = await invitesRes.json()
          setCollectionInvites(invites)
        }

        // Fetch sources for owners
        const sourcesRes = await fetch(`/api/collections/${collection.id}/sources`, { credentials: 'include' })
        if (sourcesRes.ok) {
          const sourcesData = await sourcesRes.json()
          setSources(sourcesData)
        }
      } else {
        setCollectionInvites([])
        setSources([])
      }
    } catch {
      // Error fetching collection details
    }
  }

  const handleSelectCollection = (collection: Collection) => {
    setSelectedCollection(collection)
    setShareError(null)
    setShareGithubLogin('')
    setSourceError(null)
    setNewSourceUrl('')
    setSelectedSource(null)
    fetchCollectionDetails(collection)
  }

  const handleDeleteCollection = async (collection: Collection) => {
    if (!confirm(`Are you sure you want to delete "${collection.name}"? This will delete all sources, documents, and chunks. This action cannot be undone.`)) {
      return
    }

    try {
      const response = await fetch(`/api/collections/${collection.id}`, {
        method: 'DELETE',
        credentials: 'include',
      })
      if (response.ok) {
        setSelectedCollection(null)
        fetchCollections()
      } else {
        const error = await response.json()
        alert(error.detail || 'Failed to delete collection')
      }
    } catch {
      alert('Failed to delete collection')
    }
  }

  const handleCreateCollection = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newCollectionName.trim() || !newCollectionSlug.trim()) {
      return
    }

    try {
      const response = await fetch('/api/collections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          name: newCollectionName,
          slug: newCollectionSlug,
          visibility: newCollectionVisibility,
        }),
      })
      if (response.ok) {
        const createdCollection = await response.json()
        setNewCollectionName('')
        setNewCollectionSlug('')
        setNewCollectionVisibility('private')
        // Refresh collection list and auto-select the new collection
        await fetchCollections()
        // Select the newly created collection to open its detail view
        handleSelectCollection(createdCollection)
      } else {
        const error = await response.json()
        alert(error.detail || 'Failed to create collection')
      }
    } catch (err) {
      alert('Failed to create collection: ' + (err instanceof Error ? err.message : 'Unknown error'))
    }
  }

  const handleUnshare = async (identifier: string) => {
    if (!selectedCollection) return

    try {
      const response = await fetch(`/api/collections/${selectedCollection.id}/share/${identifier}`, {
        method: 'DELETE',
        credentials: 'include',
      })
      if (response.ok) {
        fetchCollectionDetails(selectedCollection)
        fetchCollections()
      }
    } catch {
      // Error unsharing
    }
  }



  const handleCreateSource = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedCollection || !newSourceUrl.trim()) return

    try {
      const response = await fetch(`/api/collections/${selectedCollection.id}/sources`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          type: newSourceType,
          url: newSourceUrl,
          enabled: newSourceEnabled,
          schedule_interval_minutes: newSourceInterval,
        }),
      })
      if (response.ok) {
        setNewSourceUrl('')
        setNewSourceType('github')
        setNewSourceEnabled(true)
        setNewSourceInterval(60)
        setSourceError(null)
        fetchCollectionDetails(selectedCollection)
      } else {
        const error = await response.json()
        setSourceError(error.detail || 'Failed to create source')
      }
    } catch {
      setSourceError('Failed to create source')
    }
  }

  const handleDeleteSource = async (sourceId: string) => {
    if (!selectedCollection) return

    try {
      const response = await fetch(`/api/sources/${sourceId}`, {
        method: 'DELETE',
        credentials: 'include',
      })
      if (response.ok) {
        fetchCollectionDetails(selectedCollection)
      }
    } catch {
      // Error deleting source
    }
  }

  const handleSyncNow = async (sourceId: string) => {
    // Mark as syncing
    setSyncingSources(prev => new Set(prev).add(sourceId))

    try {
      const response = await fetch(`/api/sources/${sourceId}/sync-now`, {
        method: 'POST',
        credentials: 'include',
      })
      if (response.ok) {
        // Poll for sync completion
        const pollInterval = setInterval(async () => {
          try {
            const runsRes = await fetch(`/api/runs?source_id=${sourceId}&limit=1`, {
              credentials: 'include',
            })
            if (runsRes.ok) {
              const runs = await runsRes.json()
              if (runs.length > 0 && runs[0].status !== 'running') {
                // Sync finished
                clearInterval(pollInterval)
                setSyncingSources(prev => {
                  const next = new Set(prev)
                  next.delete(sourceId)
                  return next
                })
                if (selectedCollection) {
                  fetchCollectionDetails(selectedCollection)
                }
              }
            }
          } catch {
            // Polling error, keep trying
          }
        }, 2000)

        // Safety timeout after 5 minutes
        setTimeout(() => {
          clearInterval(pollInterval)
          setSyncingSources(prev => {
            const next = new Set(prev)
            next.delete(sourceId)
            return next
          })
        }, 300000)
      }
    } catch {
      setSyncingSources(prev => {
        const next = new Set(prev)
        next.delete(sourceId)
        return next
      })
    }
  }

  // Fetch sources for a specific collection (for expandable rows)
  const fetchCollectionSources = async (collectionId: string) => {
    setCollectionSourcesLoading(prev => new Set(prev).add(collectionId))
    try {
      const response = await fetch(`/api/collections/${collectionId}/sources`, { credentials: 'include' })
      if (response.ok) {
        const data = await response.json()
        setCollectionSources(prev => ({ ...prev, [collectionId]: data }))
      }
    } catch {
      // Error fetching sources
    } finally {
      setCollectionSourcesLoading(prev => {
        const next = new Set(prev)
        next.delete(collectionId)
        return next
      })
    }
  }

  // Toggle collection expansion
  const handleToggleExpand = async (collection: Collection) => {
    const isExpanded = expandedCollections.has(collection.id)
    if (isExpanded) {
      setExpandedCollections(prev => {
        const next = new Set(prev)
        next.delete(collection.id)
        return next
      })
    } else {
      setExpandedCollections(prev => new Set(prev).add(collection.id))
      // Fetch sources if not already loaded
      if (!collectionSources[collection.id]) {
        await fetchCollectionSources(collection.id)
      }
      // Also fetch members/invites for share popover
      if (!collectionMembers.length || selectedCollection?.id !== collection.id) {
        fetchCollectionDetails(collection)
      }
    }
  }

  // Calculate sync status for a collection based on its sources
  const getCollectionSyncStatus = (collectionId: string): 'success' | 'failed' | 'syncing' | 'never' | 'unknown' => {
    const sources = collectionSources[collectionId]
    if (!sources || sources.length === 0) return 'never'

    const hasSyncing = sources.some(s => syncingSources.has(s.id))
    if (hasSyncing) return 'syncing'

    // Check if any source has never been synced
    const hasNeverSynced = sources.some(s => !s.last_run_at)
    if (hasNeverSynced && sources.every(s => !s.last_run_at)) return 'never'

    // For now, show success if sources exist and have been synced
    // TODO: Track last run status per source
    return 'success'
  }

  // Handle collection edit
  const handleStartEditCollection = (collection: Collection) => {
    setEditingCollection(collection)
    setEditCollectionName(collection.name)
    setEditCollectionVisibility(collection.visibility)
  }

  const handleCancelEditCollection = () => {
    setEditingCollection(null)
  }

  const handleSaveCollection = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!editingCollection) return

    setEditCollectionLoading(true)
    try {
      const response = await fetch(`/api/collections/${editingCollection.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          name: editCollectionName,
          visibility: editCollectionVisibility,
        }),
      })
      if (response.ok) {
        await fetchCollections()
        setEditingCollection(null)
      } else {
        const error = await response.json()
        alert(error.detail || 'Failed to update collection')
      }
    } catch {
      alert('Failed to update collection')
    } finally {
      setEditCollectionLoading(false)
    }
  }

  // Handle share popover
  const handleOpenSharePopover = (collection: Collection, e: React.MouseEvent) => {
    e.stopPropagation()
    setSharePopoverCollection(collection)
    setShareGithubLogin('')
    setShareError(null)
    // Fetch members/invites
    fetchCollectionDetails(collection)
  }

  const handleCloseSharePopover = () => {
    setSharePopoverCollection(null)
    setShareError(null)
  }

  const handleShareFromPopover = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!sharePopoverCollection || !shareGithubLogin.trim()) return

    try {
      const response = await fetch(`/api/collections/${sharePopoverCollection.id}/share`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ github_login: shareGithubLogin }),
      })
      if (response.ok) {
        setShareGithubLogin('')
        setShareError(null)
        fetchCollectionDetails(sharePopoverCollection)
        fetchCollections()
      } else {
        const error = await response.json()
        setShareError(error.detail || 'Failed to share')
      }
    } catch {
      setShareError('Failed to share collection')
    }
  }

  // Get total sources and docs count for a collection
  const getCollectionStats = (collectionId: string) => {
    const sources = collectionSources[collectionId] || []
    const sourceCount = sources.length
    const docCount = sources.reduce((sum, s) => sum + s.document_count, 0)
    return { sourceCount, docCount }
  }

  const handleSetDeployKey = async (e?: React.FormEvent | React.MouseEvent) => {
    if (e) e.preventDefault()
    if (!selectedSource || !deployKeyInput.trim()) return

    setDeployKeyLoading(true)
    setDeployKeyError(null)

    try {
      const response = await fetch(`/api/sources/${selectedSource.id}/deploy-key`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ private_key: deployKeyInput }),
      })
      if (response.ok) {
        const data = await response.json()
        const updatedSource = { ...selectedSource, deploy_key_fingerprint: data.fingerprint }
        // Update the source in sources list
        setSources(sources.map(s =>
          s.id === selectedSource.id ? updatedSource : s
        ))
        // Update in collectionSources
        setCollectionSources(prev => {
          const updated = { ...prev }
          for (const collId of Object.keys(updated)) {
            updated[collId] = updated[collId].map(s =>
              s.id === selectedSource.id ? updatedSource : s
            )
          }
          return updated
        })
        setSelectedSource(updatedSource)
        setDeployKeyInput('')
      } else {
        const error = await response.json()
        setDeployKeyError(error.detail || 'Failed to set deploy key')
      }
    } catch {
      setDeployKeyError('Failed to set deploy key')
    } finally {
      setDeployKeyLoading(false)
    }
  }

  const handleDeleteDeployKey = async () => {
    if (!selectedSource) return

    setDeployKeyLoading(true)
    setDeployKeyError(null)

    try {
      const response = await fetch(`/api/sources/${selectedSource.id}/deploy-key`, {
        method: 'DELETE',
        credentials: 'include',
      })
      if (response.ok) {
        const updatedSource = { ...selectedSource, deploy_key_fingerprint: null }
        // Update the source in sources list
        setSources(sources.map(s =>
          s.id === selectedSource.id ? updatedSource : s
        ))
        // Update in collectionSources
        setCollectionSources(prev => {
          const updated = { ...prev }
          for (const collId of Object.keys(updated)) {
            updated[collId] = updated[collId].map(s =>
              s.id === selectedSource.id ? updatedSource : s
            )
          }
          return updated
        })
        setSelectedSource(updatedSource)
      } else {
        const error = await response.json()
        setDeployKeyError(error.detail || 'Failed to delete deploy key')
      }
    } catch {
      setDeployKeyError('Failed to delete deploy key')
    } finally {
      setDeployKeyLoading(false)
    }
  }

  const handleEditSource = (source: Source) => {
    setEditingSource(source)
    setEditSourceEnabled(source.enabled)
    setEditSourceInterval(source.schedule_interval_minutes)
    setEditSourceMaxPages(Number(source.config?.max_pages) || 100)
    setEditSourceError(null)
  }

  const handleCancelEditSource = () => {
    setEditingSource(null)
    setEditSourceError(null)
  }

  const handleSaveSource = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!editingSource) return

    setEditSourceLoading(true)
    setEditSourceError(null)

    try {
      const body: { enabled?: boolean; schedule_interval_minutes?: number; max_pages?: number } = {
        enabled: editSourceEnabled,
        schedule_interval_minutes: editSourceInterval,
      }
      if (editingSource.type === 'web') {
        body.max_pages = editSourceMaxPages
      }

      const response = await fetch(`/api/sources/${editingSource.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(body),
      })
      if (response.ok) {
        const updated = await response.json()
        // Update sources list
        setSources(sources.map(s => s.id === updated.id ? updated : s))
        // Update collectionSources
        setCollectionSources(prev => {
          const result = { ...prev }
          for (const collId of Object.keys(result)) {
            result[collId] = result[collId].map(s =>
              s.id === updated.id ? updated : s
            )
          }
          return result
        })
        setEditingSource(null)
      } else {
        const error = await response.json()
        setEditSourceError(error.detail || 'Failed to update source')
      }
    } catch {
      setEditSourceError('Failed to update source')
    } finally {
      setEditSourceLoading(false)
    }
  }

  // Fetch sources for runs page
  const fetchRunsSources = async (collection: Collection) => {
    try {
      const response = await fetch(`/api/collections/${collection.id}/sources`, { credentials: 'include' })
      if (response.ok) {
        const data = await response.json()
        setRunsSources(data)
      }
    } catch {
      // Error fetching sources
    }
  }

  // Fetch runs for a source
  const fetchRuns = async (source: Source) => {
    setRunsLoading(true)
    try {
      const response = await fetch(`/api/runs?source_id=${source.id}`, { credentials: 'include' })
      if (response.ok) {
        const data = await response.json()
        setRuns(data)
      }
    } catch {
      // Error fetching runs
    } finally {
      setRunsLoading(false)
    }
  }

  // Fetch Prefect flow runs
  const fetchPrefectFlowRuns = async () => {
    try {
      const response = await fetch('/api/prefect/flow-runs', { credentials: 'include' })
      if (response.ok) {
        const data = await response.json()
        setPrefectFlowRuns(data)
      }
    } catch {
      // Error fetching Prefect flow runs
    }
  }

  // Load collections and Prefect runs when switching to runs page
  useEffect(() => {
    if (currentPage === 'runs' && user) {
      fetchCollections()
      fetchPrefectFlowRuns()
      // Refresh selected source's runs if one is selected
      if (selectedRunSource) {
        fetchRuns(selectedRunSource)
      }
      // Poll for active runs and selected source runs every 5 seconds
      const interval = setInterval(() => {
        fetchPrefectFlowRuns()
        if (selectedRunSource) {
          fetchRuns(selectedRunSource)
        }
      }, 5000)
      return () => clearInterval(interval)
    }
  }, [currentPage, user, selectedRunSource])

  // Load collections for dashboard (needed for query form)
  useEffect(() => {
    if ((currentPage === 'dashboard' || currentPage === 'cockpit') && user) {
      fetchCollections()
    }
  }, [currentPage, user])

  // Handle query submission with SSE streaming
  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!queryText.trim()) return

    setQueryLoading(true)
    setQueryError(null)
    setQueryResult(null)
    setResearchStep(null)
    setResearchCitations([])
    setResearchRunId(null)

    // Route to appropriate handler based on mode
    if (queryMode === 'deep') {
      await handleDeepResearch()
    } else {
      await handleQuickQuery()
    }
  }

  // Quick query - uses semantic search + LLM synthesis
  const handleQuickQuery = async () => {
    try {
      const response = await fetch('/api/context/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          query: queryText,
          collection_id: queryCollection?.id || null,
          max_chunks: 5,
          max_tokens: 2000,
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        setQueryError(error.detail || 'Failed to generate context')
        setQueryLoading(false)
        return
      }

      const reader = response.body?.getReader()
      if (!reader) {
        setQueryError('Streaming not supported')
        setQueryLoading(false)
        return
      }

      const decoder = new TextDecoder()
      let buffer = ''
      let markdown = ''
      let metadata: { query: string; chunks_used: number; sources: ContextSource[] } | null = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Parse SSE events from buffer
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer

        let eventType = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7)
          } else if (line.startsWith('data: ')) {
            const data = line.slice(6)
            try {
              const parsed = JSON.parse(data)

              if (eventType === 'metadata') {
                metadata = {
                  query: parsed.query,
                  chunks_used: parsed.chunks_used,
                  sources: parsed.sources.map((s: { uri: string; title: string; file_path?: string }) => ({
                    uri: s.uri,
                    title: s.title,
                    file_path: s.file_path || null,
                  })),
                }
                // Initialize result with metadata
                setQueryResult({
                  markdown: '',
                  query: metadata.query,
                  chunks_used: metadata.chunks_used,
                  sources: metadata.sources,
                })
              } else if (eventType === 'content') {
                markdown += parsed.text
                if (metadata) {
                  setQueryResult({
                    markdown,
                    query: metadata.query,
                    chunks_used: metadata.chunks_used,
                    sources: metadata.sources,
                  })
                }
              } else if (eventType === 'error') {
                setQueryError(parsed.error || 'Stream error')
              }
            } catch {
              // Ignore JSON parse errors
            }
          }
        }
      }

      setQueryLoading(false)
    } catch {
      setQueryError('Failed to generate context')
      setQueryLoading(false)
    }
  }

  // Deep research - uses multi-step agent
  const handleDeepResearch = async () => {
    try {
      const response = await fetch('/api/context/research/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          question: queryText,
          budget: 10,
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        setQueryError(error.detail || 'Failed to start research')
        setQueryLoading(false)
        return
      }

      const reader = response.body?.getReader()
      if (!reader) {
        setQueryError('Streaming not supported')
        setQueryLoading(false)
        return
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Parse SSE events from buffer
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep incomplete line in buffer

        let eventType = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7)
          } else if (line.startsWith('data: ')) {
            const data = line.slice(6)
            try {
              const parsed = JSON.parse(data)

              if (eventType === 'step') {
                setResearchStep(parsed.description || `Step ${parsed.step}`)
              } else if (eventType === 'answer') {
                setQueryResult({
                  markdown: parsed.text,
                  query: queryText,
                  chunks_used: 0,
                  sources: [],
                })
              } else if (eventType === 'citations') {
                setResearchCitations(parsed.citations || [])
                setResearchRunId(parsed.run_id)
                // Update result with citation count
                setQueryResult(prev => prev ? {
                  ...prev,
                  chunks_used: parsed.steps_used || 0,
                } : null)
              } else if (eventType === 'error') {
                setQueryError(parsed.error || 'Research error')
              }
            } catch {
              // Ignore JSON parse errors
            }
          }
        }
      }

      setQueryLoading(false)
      setResearchStep(null)
    } catch {
      setQueryError('Failed to run research')
      setQueryLoading(false)
      setResearchStep(null)
    }
  }

  const handleSelectRunsCollection = (collection: Collection) => {
    setRunsCollection(collection)
    setSelectedRunSource(null)
    setRuns([])
    fetchRunsSources(collection)
  }

  const handleSelectRunSource = (source: Source) => {
    setSelectedRunSource(source)
    fetchRuns(source)
  }

  const navigateToPage = (page: Page, cockpitOptions?: CockpitNavigationOptions) => {
    updatePageQuery(page, cockpitOptions)
    setCurrentPage(page)
    setMobileMenuOpen(false)
  }

  const openCockpitForCollection = (collection?: Collection | null) => {
    navigateToPage('cockpit', {
      collectionId: collection?.id,
      view: DEFAULT_COCKPIT_VIEW,
      layer: DEFAULT_COCKPIT_LAYER,
    })
  }

  // Show login page if not authenticated
  if (!authLoading && !user) {
    return (
      <div className="app login-page">
        <div className="login-container">
          <img src="/logo-md.png" alt="ContextMine" className="login-logo" />
          <h1>ContextMine</h1>
          <p className="login-subtitle">Documentation & Code Indexing with MCP</p>
          <button className="login-button" onClick={handleLogin}>
            <svg viewBox="0 0 16 16" width="20" height="20" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            Sign in with GitHub
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <button
            className="mobile-menu-toggle"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle menu"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              {mobileMenuOpen ? (
                <path d="M18 6L6 18M6 6l12 12" />
              ) : (
                <path d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
          <img src="/logo-dark-sm.png" alt="ContextMine" className="header-logo" />
          <h1>ContextMine</h1>
          <button
            className="header-cta"
            onClick={() => navigateToPage('collections')}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 5v14M5 12h14" />
            </svg>
            Add Source
          </button>
        </div>
        {user && (
          <div className="header-right">
            <div className="user-info">
              {user.avatar_url && (
                <img src={user.avatar_url} alt={user.github_login} className="avatar" />
              )}
              <span className="username">{user.name || user.github_login}</span>
            </div>
            <button className="logout-button" onClick={handleLogout}>
              Logout
            </button>
          </div>
        )}
      </header>

      <nav className={`sidebar ${mobileMenuOpen ? 'open' : ''}`}>
        <ul>
          <li className={currentPage === 'dashboard' ? 'active' : ''} onClick={() => navigateToPage('dashboard')}>Dashboard</li>
          <li className={currentPage === 'cockpit' ? 'active' : ''} onClick={() => navigateToPage('cockpit', { view: DEFAULT_COCKPIT_VIEW, layer: DEFAULT_COCKPIT_LAYER })}>Architecture Cockpit</li>
          <li className={currentPage === 'collections' ? 'active' : ''} onClick={() => navigateToPage('collections')}>Collections</li>
          <li className={currentPage === 'runs' ? 'active' : ''} onClick={() => navigateToPage('runs')}>Runs</li>
        </ul>
      </nav>

      <main className="content">
        {currentPage === 'dashboard' && (
          <>
          <section className="card welcome-card">
            <img src="/logo-512.png" alt="ContextMine" className="welcome-logo" />
            <div className="welcome-content">
              <h2>Your AI's Knowledge Base</h2>
              <p className="welcome-tagline">
                Give Claude, Cursor, and any MCP-compatible AI assistant instant access to your documentation and codebase.
              </p>
              <div className="features-grid">
                <div className="feature-item">
                  <span className="feature-icon">🔍</span>
                  <div className="feature-text">
                    <strong>Semantic Search</strong>
                    <span>Hybrid FTS + vector search with smart ranking finds exactly what you need</span>
                  </div>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">🔄</span>
                  <div className="feature-text">
                    <strong>Auto-Sync</strong>
                    <span>GitHub repos and web docs stay current with scheduled incremental updates</span>
                  </div>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">🔌</span>
                  <div className="feature-text">
                    <strong>MCP Native</strong>
                    <span>Works with Claude Desktop, Cursor, VS Code, Cline, and Claude Code CLI</span>
                  </div>
                </div>
                <div className="feature-item">
                  <span className="feature-icon">🧠</span>
                  <div className="feature-text">
                    <strong>Deep Research</strong>
                    <span>AI agent investigates complex questions across your entire knowledge base</span>
                  </div>
                </div>
              </div>
            </div>
          </section>
          <div className="dashboard-grid">
            <div className="dashboard-left">
              <section className="card stats-card">
                <h2>Index Statistics</h2>
                <div className="stats-grid">
                  <div className="stat-item">
                    <span className="stat-value">{stats?.collections ?? '-'}</span>
                    <span className="stat-label">Collections</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{stats?.sources ?? '-'}</span>
                    <span className="stat-label">Sources</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{stats?.documents ?? '-'}</span>
                    <span className="stat-label">Documents</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{stats?.chunks ?? '-'}</span>
                    <span className="stat-label">Chunks</span>
                  </div>
                </div>
                <div className="stats-bar">
                  <div className="stats-bar-label">
                    <span>Embeddings</span>
                    <span>{stats ? `${stats.embedded_chunks} / ${stats.chunks}` : '-'}</span>
                  </div>
                  <div className="stats-bar-track">
                    <div
                      className="stats-bar-fill"
                      style={{ width: stats && stats.chunks > 0 ? `${(stats.embedded_chunks / stats.chunks) * 100}%` : '0%' }}
                    />
                  </div>
                </div>
              </section>

              <section className="card">
                <h2>System Status</h2>
                <div className="status-row">
                  <span className="label">API</span>
                  {loading ? (
                    <span className="status loading">Checking...</span>
                  ) : healthError ? (
                    <span className="status error">Error</span>
                  ) : (
                    <span className={`status ${health?.status === 'ok' ? 'ok' : 'error'}`}>
                      {health?.status === 'ok' ? 'Healthy' : 'Unhealthy'}
                    </span>
                  )}
                </div>
                <div className="status-row">
                  <span className="label">Sync Runs</span>
                  <span className="status-counts">
                    <span className="status-count ok">{stats?.runs_by_status?.success ?? 0} completed</span>
                    <span className="status-count warning">{stats?.runs_by_status?.running ?? 0} running</span>
                    <span className="status-count error">{stats?.runs_by_status?.failed ?? 0} failed</span>
                  </span>
                </div>
              </section>

              <section className="card">
                <h2>Query Documentation</h2>
                <div className="query-mode-toggle">
                  <button
                    className={`mode-button ${queryMode === 'quick' ? 'active' : ''}`}
                    onClick={() => setQueryMode('quick')}
                    type="button"
                  >
                    Quick Search
                  </button>
                  <button
                    className={`mode-button ${queryMode === 'deep' ? 'active' : ''}`}
                    onClick={() => setQueryMode('deep')}
                    type="button"
                  >
                    Deep Research
                  </button>
                </div>
                <p className="mode-description">
                  {queryMode === 'quick'
                    ? 'Fast semantic search with LLM-synthesized answer from indexed documentation.'
                    : 'Multi-step AI agent that searches, reads code, and investigates complex questions.'}
                </p>
                <form onSubmit={handleQuery} className="query-form">
                  {queryMode === 'quick' && (
                    <div className="form-row">
                      <select
                        value={queryCollection?.id || ''}
                        onChange={(e) => {
                          const coll = collections.find(c => c.id === e.target.value) || null
                          setQueryCollection(coll)
                        }}
                        className="collection-select"
                      >
                        <option value="">All accessible collections</option>
                        {collections.map((collection) => (
                          <option key={collection.id} value={collection.id}>
                            {collection.name}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                  <div className="form-row">
                    <textarea
                      placeholder={queryMode === 'quick'
                        ? "Enter your query... (e.g., 'How do I use the authentication API?')"
                        : "Enter a complex question... (e.g., 'How does the error handling work in the API layer?')"}
                      value={queryText}
                      onChange={(e) => setQueryText(e.target.value)}
                      className="query-input"
                      rows={3}
                    />
                  </div>
                  <button type="submit" className="query-button" disabled={queryLoading || !queryText.trim()}>
                    {queryLoading
                      ? (researchStep || (queryMode === 'deep' ? 'Researching...' : 'Generating...'))
                      : (queryMode === 'deep' ? 'Start Research' : 'Generate Context')}
                  </button>
                </form>
                {queryError && <p className="query-error">{queryError}</p>}
              </section>

              {queryResult && (
                <>
                  <section className="card">
                    <h2>Result</h2>
                    <div className="query-meta">
                      {queryMode === 'quick' ? (
                        <span>Used {queryResult.chunks_used} chunks from {queryResult.sources.length} sources</span>
                      ) : (
                        <span>Research completed in {queryResult.chunks_used} steps with {researchCitations.length} citations</span>
                      )}
                    </div>
                    <div className="markdown-content">
                      <pre className="markdown-raw">{queryResult.markdown}</pre>
                    </div>
                  </section>

                  {queryMode === 'quick' && queryResult.sources.length > 0 && (
                    <section className="card">
                      <h2>Sources</h2>
                      <ul className="sources-list">
                        {queryResult.sources.map((source, index) => (
                          <li key={index} className="source-item">
                            <a href={source.uri} target="_blank" rel="noopener noreferrer">
                              {source.title}
                            </a>
                            {source.file_path && (
                              <span className="file-path">{source.file_path}</span>
                            )}
                          </li>
                        ))}
                      </ul>
                    </section>
                  )}

                  {queryMode === 'deep' && researchCitations.length > 0 && (
                    <section className="card">
                      <h2>Evidence Citations</h2>
                      <ul className="sources-list citations-list">
                        {researchCitations.map((citation, index) => (
                          <li key={index} className="source-item citation-item">
                            <code>{citation}</code>
                          </li>
                        ))}
                      </ul>
                      {researchRunId && (
                        <p className="note">Run ID: {researchRunId}</p>
                      )}
                    </section>
                  )}
                </>
              )}
            </div>

            <div className="dashboard-right">
              <section className="card cockpit-cta-card">
                <div className="cockpit-cta-copy">
                  <h2>Architecture Cockpit</h2>
                  <p>Inspect extracted views across Overview, Topology, Deep Dive, C4 Diff, and Exports.</p>
                </div>
                <button
                  type="button"
                  className="cockpit-cta-button"
                  onClick={() => openCockpitForCollection(collections[0] || null)}
                >
                  Open Cockpit
                </button>
              </section>

              <section className="card coverage-ingest-card">
                <h2>GitHub Actions Coverage Ingest</h2>
                <p className="note">
                  Coverage reports are pushed from CI. ContextMine validates commit SHA and applies coverage
                  asynchronously to Twin metrics.
                </p>

                <ol className="coverage-ingest-steps">
                  <li>Get the source ID from Collections → Source details.</li>
                  <li>Rotate a source ingest token once (owner session required).</li>
                  <li>Store the token as GitHub secret <code>CONTEXTMINE_INGEST_TOKEN</code>.</li>
                  <li>Post coverage reports from GitHub Actions after tests.</li>
                </ol>

                <h3>Rotate Token (run in browser console while logged in)</h3>
                <pre className="config-block">{`await fetch("/api/sources/<SOURCE_ID>/metrics/coverage-ingest-token/rotate", {
  method: "POST",
  credentials: "include"
}).then((r) => r.json())`}</pre>

                <h3>GitHub Actions Snippet</h3>
                <pre className="config-block">{`- name: Push coverage to ContextMine
  if: always()
  env:
    CONTEXTMINE_URL: ${window.location.origin}
    CONTEXTMINE_SOURCE_ID: \${{ secrets.CONTEXTMINE_SOURCE_ID }}
    CONTEXTMINE_INGEST_TOKEN: \${{ secrets.CONTEXTMINE_INGEST_TOKEN }}
  run: |
    curl --fail-with-body \\
      -X POST "$CONTEXTMINE_URL/api/sources/$CONTEXTMINE_SOURCE_ID/metrics/coverage-ingest" \\
      -H "X-ContextMine-Ingest-Token: $CONTEXTMINE_INGEST_TOKEN" \\
      -F "commit_sha=\${{ github.sha }}" \\
      -F "branch=\${{ github.ref_name }}" \\
      -F "workflow_run_id=\${{ github.run_id }}" \\
      -F "provider=github_actions" \\
      -F "reports=@coverage/lcov.info" \\
      -F "reports=@coverage/coverage.xml"`}</pre>
              </section>

              <section className="card">
                <h2>MCP Setup</h2>
                <p className="note">Connect your AI assistant to ContextMine. Authentication is handled via GitHub OAuth - you'll be prompted to login when first connecting.</p>

                <h3>Claude Code (CLI)</h3>
                <code className="usage-example">claude mcp add contextmine {window.location.origin}/mcp</code>

                <h3>Claude Desktop</h3>
                <p className="note">Add to <code>claude_desktop_config.json</code>:</p>
                <pre className="config-block">{`{
  "mcpServers": {
    "contextmine": {
      "url": "${window.location.origin}/mcp"
    }
  }
}`}</pre>

                <h3>Cursor</h3>
                <p className="note">Add to <code>~/.cursor/mcp.json</code>:</p>
                <pre className="config-block">{`{
  "mcpServers": {
    "contextmine": {
      "url": "${window.location.origin}/mcp"
    }
  }
}`}</pre>

                <h3>VS Code</h3>
                <p className="note">Add to <code>.vscode/mcp.json</code> or user settings:</p>
                <pre className="config-block">{`{
  "mcp": {
    "servers": {
      "contextmine": {
        "url": "${window.location.origin}/mcp"
      }
    }
  }
}`}</pre>

                <h3>Cline</h3>
                <p className="note">Add to <code>cline_mcp_settings.json</code>:</p>
                <pre className="config-block">{`{
  "mcpServers": {
    "contextmine": {
      "url": "${window.location.origin}/mcp"
    }
  }
}`}</pre>
              </section>
            </div>
          </div>
          </>
        )}


        {currentPage === 'collections' && (
          <>
            <section className="card collections-overview">
              <div className="collections-header">
                <h2>Collections</h2>
                <button
                  className="create-button-inline"
                  onClick={() => setSelectedCollection({ id: 'new' } as Collection)}
                >
                  + New Collection
                </button>
              </div>

              {/* Create Collection Form (inline) */}
              {selectedCollection?.id === 'new' && (
                <div className="create-collection-inline">
                  <form onSubmit={handleCreateCollection} className="collection-form-inline">
                    <div className="form-row">
                      <input
                        type="text"
                        placeholder="Collection name"
                        value={newCollectionName}
                        onChange={(e) => setNewCollectionName(e.target.value)}
                        className="collection-input"
                        autoFocus
                      />
                      <input
                        type="text"
                        placeholder="slug"
                        value={newCollectionSlug}
                        onChange={(e) => setNewCollectionSlug(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '-'))}
                        className="collection-input slug-input"
                      />
                      <select
                        value={newCollectionVisibility}
                        onChange={(e) => setNewCollectionVisibility(e.target.value as 'global' | 'private')}
                        className="visibility-select"
                      >
                        <option value="private">Private</option>
                        <option value="global">Global</option>
                      </select>
                      <button type="submit" className="create-button">Create</button>
                      <button type="button" className="cancel-button" onClick={() => setSelectedCollection(null)}>Cancel</button>
                    </div>
                  </form>
                </div>
              )}

              {collectionsLoading ? (
                <p className="loading-text">Loading collections...</p>
              ) : collections.length === 0 ? (
                <div className="empty-state">
                  <p>No collections yet</p>
                  <p className="note">Collections organize your documentation and code sources.</p>
                  <button className="create-button" onClick={() => setSelectedCollection({ id: 'new' } as Collection)}>
                    Create Your First Collection
                  </button>
                </div>
              ) : (
                <div className="collections-list">
                  {collections.map((collection) => {
                    const isExpanded = expandedCollections.has(collection.id)
                    const isLoading = collectionSourcesLoading.has(collection.id)
                    const sources = collectionSources[collection.id] || []
                    const { sourceCount, docCount } = getCollectionStats(collection.id)
                    const syncStatus = getCollectionSyncStatus(collection.id)
                    const isEditing = editingCollection?.id === collection.id

                    return (
                      <div key={collection.id} className={`collection-row ${isExpanded ? 'expanded' : ''}`}>
                        {/* Collection Header Row */}
                        <div className="collection-header-row" onClick={() => handleToggleExpand(collection)}>
                          <button className="expand-toggle" aria-label={isExpanded ? 'Collapse' : 'Expand'}>
                            {isExpanded ? '▼' : '▶'}
                          </button>

                          <div className="collection-info">
                            {isEditing ? (
                              <form onSubmit={handleSaveCollection} className="edit-collection-form" onClick={e => e.stopPropagation()}>
                                <input
                                  type="text"
                                  value={editCollectionName}
                                  onChange={(e) => setEditCollectionName(e.target.value)}
                                  className="edit-name-input"
                                  autoFocus
                                />
                                <select
                                  value={editCollectionVisibility}
                                  onChange={(e) => setEditCollectionVisibility(e.target.value as 'global' | 'private')}
                                  className="edit-visibility-select"
                                >
                                  <option value="private">Private</option>
                                  <option value="global">Global</option>
                                </select>
                                <button type="submit" disabled={editCollectionLoading} className="save-btn">
                                  {editCollectionLoading ? '...' : 'Save'}
                                </button>
                                <button type="button" onClick={handleCancelEditCollection} className="cancel-btn">Cancel</button>
                              </form>
                            ) : (
                              <>
                                <span className="collection-name">{collection.name}</span>
                                <span className={`visibility-badge ${collection.visibility}`}>
                                  {collection.visibility === 'private' ? '🔒' : '🌐'} {collection.visibility}
                                </span>
                              </>
                            )}
                          </div>

                          <div className="collection-stats">
                            <span className="stat">{sourceCount > 0 ? `${sourceCount} source${sourceCount !== 1 ? 's' : ''}` : 'No sources'}</span>
                            <span className="stat">{docCount > 0 ? `${docCount} docs` : ''}</span>
                            {sourceCount > 0 && (
                              <span className={`sync-status status-${syncStatus}`}>
                                {syncStatus === 'success' && '●'}
                                {syncStatus === 'syncing' && '◐'}
                                {syncStatus === 'failed' && '●'}
                                {syncStatus === 'never' && '○'}
                              </span>
                            )}
                          </div>

                          <div className="collection-actions-inline" onClick={e => e.stopPropagation()}>
                            <button
                              className="action-btn cockpit-btn"
                              onClick={(e) => {
                                e.stopPropagation()
                                openCockpitForCollection(collection)
                              }}
                              title="Open in Architecture Cockpit"
                            >
                              Open in Cockpit
                            </button>
                            {collection.is_owner && (
                              <>
                                <button
                                  className="action-btn edit-btn"
                                  onClick={(e) => { e.stopPropagation(); handleStartEditCollection(collection); }}
                                  title="Edit collection"
                                >
                                  ⚙️
                                </button>
                                <button
                                  className="action-btn share-btn"
                                  onClick={(e) => handleOpenSharePopover(collection, e)}
                                  title="Share collection"
                                >
                                  🔗
                                </button>
                                <button
                                  className="action-btn delete-btn"
                                  onClick={(e) => { e.stopPropagation(); handleDeleteCollection(collection); }}
                                  title="Delete collection"
                                >
                                  🗑️
                                </button>
                              </>
                            )}
                            {!collection.is_owner && (
                              <span className="owner-label">by @{collection.owner_github_login}</span>
                            )}
                          </div>
                        </div>

                        {/* Share Popover */}
                        {sharePopoverCollection?.id === collection.id && (
                          <div className="share-popover" onClick={e => e.stopPropagation()}>
                            <div className="popover-header">
                              <h4>Share "{collection.name}"</h4>
                              <button className="close-btn" onClick={handleCloseSharePopover}>×</button>
                            </div>
                            <div className="popover-content">
                              {collectionMembers.length > 0 && (
                                <div className="members-mini">
                                  <span className="label">Members:</span>
                                  {collectionMembers.map(m => (
                                    <span key={m.user_id} className="member-chip">
                                      @{m.github_login}
                                      {m.is_owner && <span className="owner-tag">owner</span>}
                                      {!m.is_owner && (
                                        <button className="remove-chip" onClick={() => handleUnshare(m.user_id)}>×</button>
                                      )}
                                    </span>
                                  ))}
                                </div>
                              )}
                              {collectionInvites.length > 0 && (
                                <div className="invites-mini">
                                  <span className="label">Pending:</span>
                                  {collectionInvites.map(i => (
                                    <span key={i.github_login} className="member-chip pending">
                                      @{i.github_login}
                                      <button className="remove-chip" onClick={() => handleUnshare(i.github_login)}>×</button>
                                    </span>
                                  ))}
                                </div>
                              )}
                              <form onSubmit={handleShareFromPopover} className="share-form-mini">
                                <input
                                  type="text"
                                  placeholder="GitHub username"
                                  value={shareGithubLogin}
                                  onChange={(e) => setShareGithubLogin(e.target.value)}
                                  className="share-input-mini"
                                />
                                <button type="submit" className="add-btn">+ Add</button>
                              </form>
                              {shareError && <p className="share-error-mini">{shareError}</p>}
                            </div>
                          </div>
                        )}

                        {/* Expanded Sources Section */}
                        {isExpanded && (
                          <div className="collection-sources-section">
                            {isLoading ? (
                              <p className="loading-text">Loading sources...</p>
                            ) : (
                              <>
                                {collection.is_owner && (
                                  <div className="add-source-inline">
                                    <form onSubmit={(e) => { e.preventDefault(); setSelectedCollection(collection); handleCreateSource(e); }} className="source-form-inline">
                                      <select
                                        value={newSourceType}
                                        onChange={(e) => setNewSourceType(e.target.value as 'github' | 'web')}
                                        className="source-type-select"
                                      >
                                        <option value="github">GitHub</option>
                                        <option value="web">Web</option>
                                      </select>
                                      <input
                                        type="text"
                                        placeholder={newSourceType === 'github' ? 'https://github.com/owner/repo' : 'https://docs.example.com/'}
                                        value={selectedCollection?.id === collection.id ? newSourceUrl : ''}
                                        onChange={(e) => { setSelectedCollection(collection); setNewSourceUrl(e.target.value); }}
                                        onFocus={() => setSelectedCollection(collection)}
                                        className="source-url-input"
                                      />
                                      <button type="submit" className="add-source-btn" disabled={!newSourceUrl.trim() || selectedCollection?.id !== collection.id}>
                                        Add Source
                                      </button>
                                    </form>
                                    {sourceError && selectedCollection?.id === collection.id && (
                                      <p className="source-error-inline">{sourceError}</p>
                                    )}
                                  </div>
                                )}

                                {sources.length === 0 ? (
                                  <p className="no-sources-text">No sources yet. Add a GitHub repo or documentation URL above.</p>
                                ) : (
                                  <div className="sources-list">
                                    {sources.map(source => {
                                      const isEditingThis = editingSource?.id === source.id
                                      const isManagingKey = selectedSource?.id === source.id

                                      return (
                                        <div key={source.id} className={`source-row ${!source.enabled ? 'disabled' : ''} ${isEditingThis || isManagingKey ? 'editing' : ''}`}>
                                          {isEditingThis ? (
                                            /* Inline Edit Form */
                                            <div className="source-edit-inline">
                                              <div className="edit-row">
                                                <span className={`source-type-badge ${source.type}`}>{source.type}</span>
                                                <span className="source-url-static">{formatSourceUrl(source.url)}</span>
                                              </div>
                                              <div className="edit-row">
                                                <label className="checkbox-inline">
                                                  <input
                                                    type="checkbox"
                                                    checked={editSourceEnabled}
                                                    onChange={(e) => setEditSourceEnabled(e.target.checked)}
                                                  />
                                                  Enabled
                                                </label>
                                                <label className="select-inline">
                                                  Interval:
                                                  <select
                                                    value={editSourceInterval}
                                                    onChange={(e) => setEditSourceInterval(Number(e.target.value))}
                                                  >
                                                    <option value={15}>15 min</option>
                                                    <option value={30}>30 min</option>
                                                    <option value={60}>Hourly</option>
                                                    <option value={120}>2 hours</option>
                                                    <option value={360}>6 hours</option>
                                                    <option value={720}>12 hours</option>
                                                    <option value={1440}>Daily</option>
                                                  </select>
                                                </label>
                                                {source.type === 'web' && (
                                                  <label className="input-inline">
                                                    Max pages:
                                                    <input
                                                      type="number"
                                                      min={1}
                                                      max={1000}
                                                      value={editSourceMaxPages}
                                                      onChange={(e) => setEditSourceMaxPages(Number(e.target.value))}
                                                    />
                                                  </label>
                                                )}
                                              </div>
                                              <div className="edit-actions">
                                                <button
                                                  onClick={(e) => { e.preventDefault(); handleSaveSource(e as React.FormEvent); fetchCollectionSources(collection.id); }}
                                                  className="save-btn"
                                                  disabled={editSourceLoading}
                                                >
                                                  {editSourceLoading ? 'Saving...' : 'Save'}
                                                </button>
                                                <button onClick={handleCancelEditSource} className="cancel-btn">Cancel</button>
                                              </div>
                                              {editSourceError && <p className="inline-error">{editSourceError}</p>}
                                            </div>
                                          ) : isManagingKey ? (
                                            /* Inline Deploy Key Management */
                                            <div className="source-key-inline">
                                              <div className="edit-row">
                                                <span className={`source-type-badge ${source.type}`}>{source.type}</span>
                                                <span className="source-url-static">{formatSourceUrl(source.url)}</span>
                                                <button onClick={() => setSelectedSource(null)} className="close-inline">×</button>
                                              </div>
                                              {source.deploy_key_fingerprint ? (
                                                <div className="key-info">
                                                  <span className="key-status has">🔑 Key configured</span>
                                                  <code className="fingerprint">{source.deploy_key_fingerprint}</code>
                                                  <button
                                                    onClick={handleDeleteDeployKey}
                                                    className="remove-key-btn"
                                                    disabled={deployKeyLoading}
                                                  >
                                                    {deployKeyLoading ? 'Removing...' : 'Remove Key'}
                                                  </button>
                                                </div>
                                              ) : (
                                                <div className="key-form">
                                                  <p className="key-hint">Paste SSH private key for private repo access:</p>
                                                  <textarea
                                                    placeholder="-----BEGIN OPENSSH PRIVATE KEY-----"
                                                    value={deployKeyInput}
                                                    onChange={(e) => setDeployKeyInput(e.target.value)}
                                                    rows={4}
                                                  />
                                                  <button
                                                    onClick={handleSetDeployKey}
                                                    className="save-key-btn"
                                                    disabled={deployKeyLoading || !deployKeyInput.trim()}
                                                  >
                                                    {deployKeyLoading ? 'Saving...' : 'Save Key'}
                                                  </button>
                                                </div>
                                              )}
                                              {deployKeyError && <p className="inline-error">{deployKeyError}</p>}
                                            </div>
                                          ) : (
                                            /* Normal Source Row */
                                            <>
                                              <span className={`source-type-badge ${source.type}`}>{source.type}</span>
                                              <a href={source.url} target="_blank" rel="noopener noreferrer" className="source-url">
                                                {formatSourceUrl(source.url)}
                                              </a>
                                              <span className="source-docs">{source.document_count} docs</span>
                                              <span className="source-last-sync">
                                                {source.last_run_at
                                                  ? `Synced ${new Date(source.last_run_at).toLocaleDateString()}`
                                                  : 'Never synced'}
                                              </span>
                                              {collection.is_owner && (
                                                <div className="source-actions">
                                                  {source.type === 'github' && (
                                                    <button
                                                      onClick={() => { setSelectedSource(source); setDeployKeyInput(''); setDeployKeyError(null); }}
                                                      className={`key-btn ${source.deploy_key_fingerprint ? 'has-key' : ''}`}
                                                      title={source.deploy_key_fingerprint ? 'Manage deploy key' : 'Add deploy key'}
                                                    >
                                                      {source.deploy_key_fingerprint ? '🔑' : '🔐'}
                                                    </button>
                                                  )}
                                                  <button
                                                    onClick={() => handleSyncNow(source.id)}
                                                    className={`sync-btn ${syncingSources.has(source.id) ? 'syncing' : ''}`}
                                                    disabled={syncingSources.has(source.id)}
                                                  >
                                                    {syncingSources.has(source.id) ? 'Syncing...' : 'Sync'}
                                                  </button>
                                                  <button
                                                    onClick={() => { setSelectedCollection(collection); setSources(sources); handleEditSource(source); }}
                                                    className="edit-btn"
                                                  >
                                                    Edit
                                                  </button>
                                                  <button
                                                    onClick={() => { setSelectedCollection(collection); handleDeleteSource(source.id); fetchCollectionSources(collection.id); }}
                                                    className="delete-btn"
                                                  >
                                                    ×
                                                  </button>
                                                </div>
                                              )}
                                            </>
                                          )}
                                        </div>
                                      )
                                    })}
                                  </div>
                                )}
                              </>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </section>
          </>
        )}

        {currentPage === 'runs' && (
          <>
            {/* Active Runs from Prefect */}
            <section className="card active-runs-section">
              <h2>Active Runs</h2>
              {prefectFlowRuns?.error ? (
                <p className="note error">Prefect connection error: {prefectFlowRuns.error}</p>
              ) : prefectFlowRuns?.active && prefectFlowRuns.active.length > 0 ? (
                <div className="active-runs-grid">
                  {prefectFlowRuns.active.map((run) => (
                    <div key={run.id} className={`active-run-card state-${run.state_type.toLowerCase()}`}>
                      <div className="run-header">
                        <span className={`state-badge ${run.state_type.toLowerCase()}`}>
                          {run.state_name}
                        </span>
                        <span className="run-name">{run.name}</span>
                      </div>
                      {run.parameters.source_url && (
                        <div className="run-source">
                          {formatSourceUrl(run.parameters.source_url)}
                        </div>
                      )}
                      {run.progress && (
                        <div className="run-progress">
                          <div className="progress-bar">
                            <div
                              className="progress-fill"
                              style={{ width: `${run.progress.percent}%` }}
                            />
                          </div>
                          <div className="progress-stats">
                            <span>{run.progress.completed}/{run.progress.total} tasks</span>
                            {run.progress.current_task && (
                              <span className="current-task">{run.progress.current_task}</span>
                            )}
                          </div>
                        </div>
                      )}
                      {run.start_time && (
                        <div className="run-started">
                          Started {new Date(run.start_time).toLocaleTimeString()}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="note">No active runs. Runs will appear here when syncs are in progress.</p>
              )}
            </section>

            {/* Recent Runs from Prefect */}
            {prefectFlowRuns?.recent && prefectFlowRuns.recent.length > 0 && (
              <section className="card">
                <h2>Recent Runs (Prefect)</h2>
                <table className="runs-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Source</th>
                      <th>Status</th>
                      <th>Started</th>
                      <th>Duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {prefectFlowRuns.recent.slice(0, 10).map((run) => {
                      const startTime = run.start_time ? new Date(run.start_time) : null
                      const endTime = run.end_time ? new Date(run.end_time) : null
                      const durationMs = startTime && endTime ? endTime.getTime() - startTime.getTime() : null
                      const durationStr = durationMs !== null
                        ? durationMs < 1000
                          ? `${durationMs}ms`
                          : durationMs < 60000
                            ? `${(durationMs / 1000).toFixed(1)}s`
                            : `${Math.floor(durationMs / 60000)}m ${Math.floor((durationMs % 60000) / 1000)}s`
                        : '-'

                      return (
                        <tr key={run.id} className={`state-${run.state_type.toLowerCase()}`}>
                          <td>{run.name}</td>
                          <td className="source-cell">
                            {run.parameters.source_url ? formatSourceUrl(run.parameters.source_url) : '-'}
                          </td>
                          <td>
                            <span className={`state-badge ${run.state_type.toLowerCase()}`}>
                              {run.state_name}
                            </span>
                          </td>
                          <td>{startTime ? startTime.toLocaleString() : '-'}</td>
                          <td>{durationStr}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </section>
            )}

            <section className="card">
              <h2>Run History</h2>
              <div className="run-filters">
                <div className="filter-group">
                  <label>Collection:</label>
                  <select
                    value={runsCollection?.id || ''}
                    onChange={(e) => {
                      const coll = collections.find(c => c.id === e.target.value) || null
                      if (coll) {
                        handleSelectRunsCollection(coll)
                      } else {
                        setRunsCollection(null)
                        setRunsSources([])
                        setSelectedRunSource(null)
                        setRuns([])
                      }
                    }}
                    className="filter-select"
                  >
                    <option value="">Select collection...</option>
                    {collections.map((collection) => (
                      <option key={collection.id} value={collection.id}>
                        {collection.name}
                      </option>
                    ))}
                  </select>
                </div>
                {runsCollection && (
                  <div className="filter-group">
                    <label>Source:</label>
                    <select
                      value={selectedRunSource?.id || ''}
                      onChange={(e) => {
                        const src = runsSources.find(s => s.id === e.target.value) || null
                        if (src) {
                          handleSelectRunSource(src)
                        } else {
                          setSelectedRunSource(null)
                          setRuns([])
                        }
                      }}
                      className="filter-select"
                    >
                      <option value="">Select source...</option>
                      {runsSources.map((source) => (
                        <option key={source.id} value={source.id}>
                          [{source.type}] {formatSourceUrl(source.url)}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>

              {!runsCollection && (
                <p className="note">Select a collection and source to view run history.</p>
              )}
              {runsCollection && !selectedRunSource && runsSources.length > 0 && (
                <p className="note">Select a source to view its run history.</p>
              )}
              {runsCollection && runsSources.length === 0 && (
                <p className="note">No sources in this collection.</p>
              )}
            </section>

            {selectedRunSource && (
              <section className="card">
                <h2>Runs: {formatSourceUrl(selectedRunSource.url)}</h2>
                {runsLoading ? (
                  <p>Loading runs...</p>
                ) : runs.length === 0 ? (
                  <p className="note">No runs yet for this source.</p>
                ) : (
                  <table className="runs-table">
                    <thead>
                      <tr>
                        <th>Started</th>
                        <th>Finished</th>
                        <th>Duration</th>
                        <th>Status</th>
                        <th>Stats</th>
                        <th>Error</th>
                      </tr>
                    </thead>
                    <tbody>
                      {runs.map((run) => {
                        const startDate = new Date(run.started_at)
                        const endDate = run.finished_at ? new Date(run.finished_at) : null
                        const durationMs = endDate ? endDate.getTime() - startDate.getTime() : null
                        const durationStr = durationMs !== null
                          ? durationMs < 1000
                            ? `${durationMs}ms`
                            : durationMs < 60000
                              ? `${(durationMs / 1000).toFixed(1)}s`
                              : `${Math.floor(durationMs / 60000)}m ${Math.floor((durationMs % 60000) / 1000)}s`
                          : '-'

                        return (
                          <tr key={run.id} className={`run-${run.status}`}>
                            <td>{startDate.toLocaleString()}</td>
                            <td>{endDate ? endDate.toLocaleString() : '-'}</td>
                            <td>{durationStr}</td>
                            <td>
                              <span className={`status ${run.status === 'success' ? 'ok' : run.status === 'failed' ? 'error' : 'loading'}`}>
                                {run.status}
                              </span>
                            </td>
                            <td className="stats-cell">
                              {run.stats ? (
                                <div className="stats-grid">
                                  {run.stats.docs_created !== undefined && (
                                    <span className="stat-item">
                                      <span className="stat-value created">+{run.stats.docs_created}</span>
                                      <span className="stat-label">docs</span>
                                    </span>
                                  )}
                                  {run.stats.docs_updated !== undefined && run.stats.docs_updated > 0 && (
                                    <span className="stat-item">
                                      <span className="stat-value updated">~{run.stats.docs_updated}</span>
                                      <span className="stat-label">updated</span>
                                    </span>
                                  )}
                                  {run.stats.docs_deleted !== undefined && run.stats.docs_deleted > 0 && (
                                    <span className="stat-item">
                                      <span className="stat-value deleted">-{run.stats.docs_deleted}</span>
                                      <span className="stat-label">deleted</span>
                                    </span>
                                  )}
                                  {run.stats.chunks_embedded !== undefined && (
                                    <span className="stat-item">
                                      <span className="stat-value">{run.stats.chunks_embedded}</span>
                                      <span className="stat-label">chunks</span>
                                    </span>
                                  )}
                                  {run.stats.files_scanned !== undefined && (
                                    <span className="stat-item" title={`Indexed: ${run.stats.files_indexed}, Skipped: ${run.stats.files_skipped}`}>
                                      <span className="stat-value">{run.stats.files_scanned}</span>
                                      <span className="stat-label">files</span>
                                    </span>
                                  )}
                                  {run.stats.pages_crawled !== undefined && (
                                    <span className="stat-item">
                                      <span className="stat-value">{run.stats.pages_crawled}</span>
                                      <span className="stat-label">pages</span>
                                    </span>
                                  )}
                                </div>
                              ) : '-'}
                            </td>
                            <td className="error-cell">
                              {run.error ? (
                                <span className="error-text" title={run.error}>
                                  {run.error.length > 50 ? run.error.substring(0, 50) + '...' : run.error}
                                </span>
                              ) : '-'}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                )}
              </section>
            )}
          </>
        )}

        {currentPage === 'cockpit' && (
          <CockpitPage
            collections={collections.map((c) => ({ id: c.id, name: c.name }))}
            onOpenCollections={() => navigateToPage('collections')}
            onOpenRuns={() => navigateToPage('runs')}
          />
        )}
      </main>

      <footer className="footer">
        <div className="footer-left">
          <span>ContextMine by</span>
          <a href="https://mayflower.de" target="_blank" rel="noopener noreferrer">Mayflower</a>
        </div>
        <div className="footer-links">
          <a href={GITHUB_REPO} target="_blank" rel="noopener noreferrer">GitHub</a>
          <a href={`${GITHUB_REPO}/blob/main/README.md`} target="_blank" rel="noopener noreferrer">Docs</a>
          <a href={`${GITHUB_REPO}/issues`} target="_blank" rel="noopener noreferrer">Issues</a>
        </div>
      </footer>
    </div>
  )
}

export default App
