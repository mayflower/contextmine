import { useEffect, useState } from 'react'
import './App.css'

interface HealthStatus {
  status: string
}

interface User {
  id: string
  github_login: string
  name: string | null
  avatar_url: string | null
}

interface MCPToken {
  id: string
  name: string
  token?: string // Only present on creation
  created_at: string
  last_used_at: string | null
  revoked_at: string | null
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

type Page = 'dashboard' | 'query' | 'collections' | 'runs'

const GITHUB_REPO = 'https://github.com/mayflower/contextmine'

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
  const [currentPage, setCurrentPage] = useState<Page>('dashboard')

  // MCP Tokens state
  const [tokens, setTokens] = useState<MCPToken[]>([])
  const [tokensLoading, setTokensLoading] = useState(false)
  const [newTokenName, setNewTokenName] = useState('')
  const [createdToken, setCreatedToken] = useState<string | null>(null)

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

  // Fetch MCP tokens
  const fetchTokens = async () => {
    setTokensLoading(true)
    try {
      const response = await fetch('/api/mcp-tokens', { credentials: 'include' })
      if (response.ok) {
        const data = await response.json()
        setTokens(data)
      }
    } catch {
      // Error fetching tokens
    } finally {
      setTokensLoading(false)
    }
  }

  // Load tokens when switching to Dashboard page
  useEffect(() => {
    if (currentPage === 'dashboard' && user) {
      fetchTokens()
    }
  }, [currentPage, user])

  const handleCreateToken = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newTokenName.trim()) return

    try {
      const response = await fetch('/api/mcp-tokens', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ name: newTokenName }),
      })
      if (response.ok) {
        const data = await response.json()
        setCreatedToken(data.token)
        setNewTokenName('')
        fetchTokens()
      }
    } catch {
      // Error creating token
    }
  }

  const handleRevokeToken = async (tokenId: string) => {
    try {
      const response = await fetch(`/api/mcp-tokens/${tokenId}`, {
        method: 'DELETE',
        credentials: 'include',
      })
      if (response.ok) {
        fetchTokens()
      }
    } catch {
      // Error revoking token
    }
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

  const handleShareCollection = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedCollection || !shareGithubLogin.trim()) return

    try {
      const response = await fetch(`/api/collections/${selectedCollection.id}/share`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ github_login: shareGithubLogin }),
      })
      if (response.ok) {
        setShareGithubLogin('')
        setShareError(null)
        fetchCollectionDetails(selectedCollection)
        fetchCollections()
      } else {
        const error = await response.json()
        setShareError(error.detail || 'Failed to share')
      }
    } catch {
      setShareError('Failed to share collection')
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

  const handleSelectSource = (source: Source) => {
    if (selectedSource?.id === source.id) {
      setSelectedSource(null)
    } else {
      setSelectedSource(source)
      setDeployKeyInput('')
      setDeployKeyError(null)
    }
  }

  const handleSetDeployKey = async (e: React.FormEvent) => {
    e.preventDefault()
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
        // Update the source in the list with the new fingerprint
        setSources(sources.map(s =>
          s.id === selectedSource.id
            ? { ...s, deploy_key_fingerprint: data.fingerprint }
            : s
        ))
        setSelectedSource({ ...selectedSource, deploy_key_fingerprint: data.fingerprint })
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
        // Update the source in the list
        setSources(sources.map(s =>
          s.id === selectedSource.id
            ? { ...s, deploy_key_fingerprint: null }
            : s
        ))
        setSelectedSource({ ...selectedSource, deploy_key_fingerprint: null })
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
        setSources(sources.map(s => s.id === updated.id ? updated : s))
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

  // Load collections for query page
  useEffect(() => {
    if (currentPage === 'query' && user) {
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
          <img src="/logo-sm.png" alt="ContextMine" className="header-logo" />
          <h1>ContextMine</h1>
          <span className="subtitle">Admin Console</span>
          <button
            className="header-cta"
            onClick={() => setCurrentPage('collections')}
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
          <li className={currentPage === 'dashboard' ? 'active' : ''} onClick={() => { setCurrentPage('dashboard'); setMobileMenuOpen(false); }}>Dashboard</li>
          <li className={currentPage === 'query' ? 'active' : ''} onClick={() => { setCurrentPage('query'); setMobileMenuOpen(false); }}>Query</li>
          <li className={currentPage === 'collections' ? 'active' : ''} onClick={() => { setCurrentPage('collections'); setMobileMenuOpen(false); }}>Collections</li>
          <li className={currentPage === 'runs' ? 'active' : ''} onClick={() => { setCurrentPage('runs'); setMobileMenuOpen(false); }}>Runs</li>
        </ul>
      </nav>

      <main className="content">
        {currentPage === 'dashboard' && (
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
                <h2>Recent Sync Runs</h2>
                {stats?.recent_runs && stats.recent_runs.length > 0 ? (
                  <table className="runs-table compact">
                    <thead>
                      <tr>
                        <th>Source</th>
                        <th>Status</th>
                        <th>Started</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stats.recent_runs.slice(0, 5).map((run) => (
                        <tr key={run.id}>
                          <td className="source-cell" title={run.source_url}>
                            {formatSourceUrl(run.source_url)}
                          </td>
                          <td>
                            <span className={`status-badge ${run.status}`}>
                              {run.status}
                            </span>
                          </td>
                          <td>{run.started_at ? new Date(run.started_at).toLocaleString() : '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p className="note">No sync runs yet.</p>
                )}
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
            </div>

            <div className="dashboard-right">
              <section className="card">
                <h2>MCP Connection</h2>
                <div className="mcp-endpoint">
                  <code>{window.location.origin}/mcp</code>
                </div>

                <h3>Access Tokens</h3>
                <form onSubmit={handleCreateToken} className="token-form">
                  <input
                    type="text"
                    placeholder="Token name (e.g., 'Claude Desktop')"
                    value={newTokenName}
                    onChange={(e) => setNewTokenName(e.target.value)}
                    className="token-input"
                  />
                  <button type="submit" className="create-button">Create</button>
                </form>
                {createdToken && (
                  <div className="token-created">
                    <p><strong>Token created!</strong> Copy it now:</p>
                    <code className="token-value">{createdToken}</code>
                    <button onClick={() => {
                      navigator.clipboard.writeText(createdToken)
                    }} className="copy-button">Copy</button>
                    <button onClick={() => setCreatedToken(null)} className="dismiss-button">×</button>
                  </div>
                )}

                {tokensLoading ? (
                  <p>Loading...</p>
                ) : tokens.length === 0 ? (
                  <p className="note">No tokens yet.</p>
                ) : (
                  <table className="tokens-table compact">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Last Used</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {tokens.filter(t => !t.revoked_at).map((token) => (
                        <tr key={token.id}>
                          <td>{token.name}</td>
                          <td>{token.last_used_at ? new Date(token.last_used_at).toLocaleDateString() : 'Never'}</td>
                          <td>
                            <button
                              onClick={() => handleRevokeToken(token.id)}
                              className="revoke-button small"
                            >
                              Revoke
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </section>

              <section className="card">
                <h2>Claude Code Setup</h2>
                <p className="note">Add ContextMine to Claude Code with this command:</p>
                <code className="usage-example">claude mcp add --transport http -H "Authorization: Bearer YOUR_TOKEN" contextmine {window.location.origin}/mcp</code>
              </section>
            </div>
          </div>
        )}

        {currentPage === 'query' && (
          <>
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
          </>
        )}


        {currentPage === 'collections' && (
          <>
            <section className="card">
              <h2>Create New Collection</h2>
              <form onSubmit={handleCreateCollection} className="collection-form">
                <div className="form-row">
                  <input
                    type="text"
                    placeholder="Collection name"
                    value={newCollectionName}
                    onChange={(e) => setNewCollectionName(e.target.value)}
                    className="collection-input"
                  />
                  <input
                    type="text"
                    placeholder="slug (url-friendly)"
                    value={newCollectionSlug}
                    onChange={(e) => setNewCollectionSlug(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '-'))}
                    className="collection-input"
                  />
                </div>
                <div className="form-row">
                  <label className="visibility-option">
                    <input
                      type="radio"
                      name="visibility"
                      value="private"
                      checked={newCollectionVisibility === 'private'}
                      onChange={() => setNewCollectionVisibility('private')}
                    />
                    Private (only owner and members)
                  </label>
                  <label className="visibility-option">
                    <input
                      type="radio"
                      name="visibility"
                      value="global"
                      checked={newCollectionVisibility === 'global'}
                      onChange={() => setNewCollectionVisibility('global')}
                    />
                    Global (visible to everyone)
                  </label>
                </div>
                <button type="submit" className="create-button">Create Collection</button>
              </form>
            </section>

            <section className="card">
              <h2>Your Collections</h2>
              {collectionsLoading ? (
                <p>Loading collections...</p>
              ) : collections.length === 0 ? (
                <p className="note">No collections yet. Create one above.</p>
              ) : (
                <table className="collections-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Slug</th>
                      <th>Visibility</th>
                      <th>Owner</th>
                      <th>Members</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {collections.map((collection) => (
                      <tr key={collection.id} className={selectedCollection?.id === collection.id ? 'selected' : ''}>
                        <td>{collection.name}</td>
                        <td><code>{collection.slug}</code></td>
                        <td>
                          <span className={`visibility ${collection.visibility}`}>
                            {collection.visibility}
                          </span>
                        </td>
                        <td>
                          {collection.is_owner ? (
                            <span className="owner-badge">You</span>
                          ) : (
                            collection.owner_github_login
                          )}
                        </td>
                        <td>{collection.member_count}</td>
                        <td>
                          <button
                            onClick={() => handleSelectCollection(collection)}
                            className="manage-button"
                          >
                            {collection.is_owner ? 'Manage' : 'View'}
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>

            {selectedCollection && (
              <section className="card">
                <h2>
                  {selectedCollection.name}
                  <button className="close-button" onClick={() => setSelectedCollection(null)}>×</button>
                </h2>

                {selectedCollection.is_owner && (
                  <div className="collection-actions">
                    <button
                      onClick={() => handleDeleteCollection(selectedCollection)}
                      className="delete-collection-button"
                    >
                      Delete Collection
                    </button>
                  </div>
                )}

                <h3>Members</h3>
                <ul className="members-list">
                  {collectionMembers.map((member) => (
                    <li key={member.user_id} className="member-item">
                      {member.avatar_url && (
                        <img src={member.avatar_url} alt={member.github_login} className="member-avatar" />
                      )}
                      <span className="member-name">
                        {member.name || member.github_login}
                        {member.is_owner && <span className="owner-tag">Owner</span>}
                      </span>
                      {selectedCollection.is_owner && !member.is_owner && (
                        <button
                          onClick={() => handleUnshare(member.user_id)}
                          className="remove-button"
                        >
                          Remove
                        </button>
                      )}
                    </li>
                  ))}
                </ul>

                {selectedCollection.is_owner && collectionInvites.length > 0 && (
                  <>
                    <h3>Pending Invites</h3>
                    <ul className="invites-list">
                      {collectionInvites.map((invite) => (
                        <li key={invite.github_login} className="invite-item">
                          <span className="invite-login">{invite.github_login}</span>
                          <span className="invite-date">
                            Invited {new Date(invite.created_at).toLocaleDateString()}
                          </span>
                          <button
                            onClick={() => handleUnshare(invite.github_login)}
                            className="remove-button"
                          >
                            Cancel
                          </button>
                        </li>
                      ))}
                    </ul>
                  </>
                )}

                {selectedCollection.is_owner && (
                  <>
                    <h3>Share Collection</h3>
                    <form onSubmit={handleShareCollection} className="share-form">
                      <input
                        type="text"
                        placeholder="GitHub username"
                        value={shareGithubLogin}
                        onChange={(e) => setShareGithubLogin(e.target.value)}
                        className="share-input"
                      />
                      <button type="submit" className="share-button">Share</button>
                    </form>
                    {shareError && <p className="share-error">{shareError}</p>}
                    <p className="note">
                      If the user has an account, they'll be added immediately.
                      Otherwise, an invite will be created and they'll be added when they sign up.
                    </p>

                    <h3>Sources</h3>
                    <form onSubmit={handleCreateSource} className="source-form">
                      <div className="form-row">
                        <label className="source-type-option">
                          <input
                            type="radio"
                            name="sourceType"
                            value="github"
                            checked={newSourceType === 'github'}
                            onChange={() => setNewSourceType('github')}
                          />
                          GitHub
                        </label>
                        <label className="source-type-option">
                          <input
                            type="radio"
                            name="sourceType"
                            value="web"
                            checked={newSourceType === 'web'}
                            onChange={() => setNewSourceType('web')}
                          />
                          Web
                        </label>
                        <input
                          type="text"
                          placeholder={newSourceType === 'github' ? 'https://github.com/owner/repo' : 'https://docs.example.com/'}
                          value={newSourceUrl}
                          onChange={(e) => setNewSourceUrl(e.target.value)}
                          className="source-input"
                        />
                        <select
                          value={newSourceInterval}
                          onChange={(e) => setNewSourceInterval(Number(e.target.value))}
                          className="interval-select"
                        >
                          <option value={15}>Every 15 min</option>
                          <option value={30}>Every 30 min</option>
                          <option value={60}>Hourly</option>
                          <option value={120}>Every 2 hours</option>
                          <option value={360}>Every 6 hours</option>
                          <option value={720}>Every 12 hours</option>
                          <option value={1440}>Daily</option>
                        </select>
                        <button type="submit" className="create-button">Add</button>
                      </div>
                    </form>
                    {sourceError && <p className="source-error">{sourceError}</p>}

                    {sources.length === 0 ? (
                      <p className="note">No sources yet. Add a GitHub repo or web URL above.</p>
                    ) : (
                      <table className="sources-table">
                        <thead>
                          <tr>
                            <th>Type</th>
                            <th>URL</th>
                            <th>Docs</th>
                            <th>Key</th>
                            <th>Last Run</th>
                            <th>Actions</th>
                          </tr>
                        </thead>
                        <tbody>
                          {sources.map((source) => (
                            <tr
                              key={source.id}
                              className={`${!source.enabled ? 'disabled' : ''} ${selectedSource?.id === source.id ? 'selected' : ''}`}
                            >
                              <td>
                                <span className={`source-type ${source.type}`}>
                                  {source.type}
                                </span>
                              </td>
                              <td className="source-url">
                                <a href={source.url} target="_blank" rel="noopener noreferrer">
                                  {source.url.length > 40 ? source.url.substring(0, 40) + '...' : source.url}
                                </a>
                              </td>
                              <td className="doc-count">
                                <span className="doc-count-badge">{source.document_count}</span>
                              </td>
                              <td className="deploy-key-cell">
                                {source.type === 'github' ? (
                                  <button
                                    onClick={() => handleSelectSource(source)}
                                    className={`deploy-key-button ${source.deploy_key_fingerprint ? 'has-key' : ''}`}
                                    title={source.deploy_key_fingerprint || 'No deploy key'}
                                  >
                                    {source.deploy_key_fingerprint ? '🔑' : '➕'}
                                  </button>
                                ) : (
                                  <span className="na">-</span>
                                )}
                              </td>
                              <td>
                                {source.last_run_at
                                  ? new Date(source.last_run_at).toLocaleString()
                                  : 'Never'}
                              </td>
                              <td className="source-actions">
                                <button
                                  onClick={() => handleEditSource(source)}
                                  className="edit-button"
                                  disabled={syncingSources.has(source.id)}
                                >
                                  Edit
                                </button>
                                <button
                                  onClick={() => handleSyncNow(source.id)}
                                  className={`sync-button ${syncingSources.has(source.id) ? 'syncing' : ''}`}
                                  disabled={syncingSources.has(source.id)}
                                >
                                  {syncingSources.has(source.id) ? 'Syncing...' : 'Sync'}
                                </button>
                                <button
                                  onClick={() => handleDeleteSource(source.id)}
                                  className="delete-button"
                                  disabled={syncingSources.has(source.id)}
                                >
                                  Delete
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}

                    {editingSource && (
                      <div className="edit-source-section">
                        <h4>
                          Edit Source
                          <button className="close-button" onClick={handleCancelEditSource}>×</button>
                        </h4>
                        <p className="source-url-display">{editingSource.url}</p>
                        <form onSubmit={handleSaveSource} className="edit-source-form">
                          <div className="form-row">
                            <label className="checkbox-label">
                              <input
                                type="checkbox"
                                checked={editSourceEnabled}
                                onChange={(e) => setEditSourceEnabled(e.target.checked)}
                              />
                              Enabled
                            </label>
                          </div>
                          <div className="form-row">
                            <label>Sync Interval</label>
                            <select
                              value={editSourceInterval}
                              onChange={(e) => setEditSourceInterval(Number(e.target.value))}
                              className="interval-select"
                            >
                              <option value={15}>Every 15 min</option>
                              <option value={30}>Every 30 min</option>
                              <option value={60}>Hourly</option>
                              <option value={120}>Every 2 hours</option>
                              <option value={360}>Every 6 hours</option>
                              <option value={720}>Every 12 hours</option>
                              <option value={1440}>Daily</option>
                            </select>
                          </div>
                          {editingSource.type === 'web' && (
                            <div className="form-row">
                              <label>Max Pages</label>
                              <input
                                type="number"
                                min={1}
                                max={1000}
                                value={editSourceMaxPages}
                                onChange={(e) => setEditSourceMaxPages(Number(e.target.value))}
                                className="max-pages-input"
                              />
                            </div>
                          )}
                          <div className="form-actions">
                            <button
                              type="button"
                              onClick={handleCancelEditSource}
                              className="cancel-button"
                            >
                              Cancel
                            </button>
                            <button
                              type="submit"
                              className="save-button"
                              disabled={editSourceLoading}
                            >
                              {editSourceLoading ? 'Saving...' : 'Save'}
                            </button>
                          </div>
                        </form>
                        {editSourceError && <p className="edit-source-error">{editSourceError}</p>}
                      </div>
                    )}

                    {selectedSource && selectedSource.type === 'github' && (
                      <div className="deploy-key-section">
                        <h4>
                          Deploy Key for {selectedSource.url.split('/').slice(-2).join('/')}
                          <button className="close-button" onClick={() => setSelectedSource(null)}>×</button>
                        </h4>

                        {selectedSource.deploy_key_fingerprint ? (
                          <div className="deploy-key-info">
                            <p className="key-status has-key">Deploy key configured</p>
                            <p className="fingerprint">
                              <strong>Fingerprint:</strong> <code>{selectedSource.deploy_key_fingerprint}</code>
                            </p>
                            <button
                              onClick={handleDeleteDeployKey}
                              className="delete-button"
                              disabled={deployKeyLoading}
                            >
                              {deployKeyLoading ? 'Removing...' : 'Remove Key'}
                            </button>
                          </div>
                        ) : (
                          <div className="deploy-key-form-section">
                            <p className="note">
                              Add a deploy key to access private repos. Paste your private key below.
                            </p>
                            <form onSubmit={handleSetDeployKey} className="deploy-key-form">
                              <textarea
                                placeholder="-----BEGIN OPENSSH PRIVATE KEY-----&#10;...&#10;-----END OPENSSH PRIVATE KEY-----"
                                value={deployKeyInput}
                                onChange={(e) => setDeployKeyInput(e.target.value)}
                                className="deploy-key-input"
                                rows={6}
                              />
                              <button
                                type="submit"
                                className="create-button"
                                disabled={deployKeyLoading || !deployKeyInput.trim()}
                              >
                                {deployKeyLoading ? 'Saving...' : 'Save Key'}
                              </button>
                            </form>
                          </div>
                        )}
                        {deployKeyError && <p className="deploy-key-error">{deployKeyError}</p>}
                      </div>
                    )}
                  </>
                )}
              </section>
            )}
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
