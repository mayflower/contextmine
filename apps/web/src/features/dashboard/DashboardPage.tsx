import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useLocation, useNavigate } from 'react-router'

import { api, apiData } from '../../api/client'
import type { ContextResult } from '../../api/types'
import { readSSEStream } from '../../lib/sseReader'
import { DEFAULT_COCKPIT_LAYER, DEFAULT_COCKPIT_VIEW, routeLocation } from '../../routing'

type ContextSource = ContextResult['sources'][number]

export default function DashboardPage() {
  const location = useLocation()
  const navigate = useNavigate()
  const statsQuery = useQuery({
    queryKey: ['stats'],
    queryFn: ({ signal }) => apiData(api.GET('/api/stats', { signal })),
  })
  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: ({ signal }) => apiData(api.GET('/api/health', { signal })),
    refetchInterval: 30_000,
  })
  const collectionsQuery = useQuery({
    queryKey: ['collections'],
    queryFn: ({ signal }) => apiData(api.GET('/api/collections', { signal })),
  })
  const stats = statsQuery.data
  const health = healthQuery.data
  const collections = collectionsQuery.data ?? []

  const [queryText, setQueryText] = useState('')
  const [queryCollectionId, setQueryCollectionId] = useState('')
  const [queryResult, setQueryResult] = useState<ContextResult | null>(null)
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryError, setQueryError] = useState<string | null>(null)
  const [queryMode, setQueryMode] = useState<'quick' | 'deep'>('quick')
  const [researchStep, setResearchStep] = useState<string | null>(null)
  const [researchCitations, setResearchCitations] = useState<string[]>([])
  const [researchRunId, setResearchRunId] = useState<string | null>(null)

  const runQuickQuery = async () => {
    const response = await fetch('/api/context/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({
        query: queryText,
        collection_id: queryCollectionId || null,
        max_chunks: 5,
        max_tokens: 2000,
      }),
    })
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to generate context')
    }

    let markdown = ''
    let metadata: { query: string; chunks_used: number; sources: ContextSource[] } | null = null
    await readSSEStream(response, (eventType, data) => {
      const parsed = JSON.parse(data)
      if (eventType === 'metadata') {
        metadata = {
          query: parsed.query,
          chunks_used: parsed.chunks_used,
          sources: parsed.sources.map((source: { uri: string; title: string; file_path?: string }) => ({
            uri: source.uri,
            title: source.title,
            file_path: source.file_path || null,
          })),
        }
        setQueryResult({ markdown: '', ...metadata })
      } else if (eventType === 'content' && metadata) {
        markdown += parsed.text
        setQueryResult({ markdown, ...metadata })
      } else if (eventType === 'error') {
        throw new Error(parsed.error || 'Stream error')
      }
    })
  }

  const runDeepResearch = async () => {
    const response = await fetch('/api/context/research/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ question: queryText, budget: 10 }),
    })
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || 'Failed to start research')
    }

    await readSSEStream(response, (eventType, data) => {
      const parsed = JSON.parse(data)
      if (eventType === 'step') {
        setResearchStep(parsed.description || `Step ${parsed.step}`)
      } else if (eventType === 'answer') {
        setQueryResult({ markdown: parsed.text, query: queryText, chunks_used: 0, sources: [] })
      } else if (eventType === 'citations') {
        setResearchCitations(parsed.citations || [])
        setResearchRunId(parsed.run_id)
        setQueryResult((result) => result ? { ...result, chunks_used: parsed.steps_used || 0 } : null)
      } else if (eventType === 'error') {
        throw new Error(parsed.error || 'Research error')
      }
    })
  }

  const handleQuery = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!queryText.trim()) return
    setQueryLoading(true)
    setQueryError(null)
    setQueryResult(null)
    setResearchStep(null)
    setResearchCitations([])
    setResearchRunId(null)
    try {
      await (queryMode === 'deep' ? runDeepResearch() : runQuickQuery())
    } catch (error) {
      setQueryError(error instanceof Error ? error.message : 'Request failed')
    } finally {
      setQueryLoading(false)
      setResearchStep(null)
    }
  }

  const openCockpit = () => navigate(routeLocation('cockpit', location.search, {
    collectionId: collections[0]?.id,
    view: DEFAULT_COCKPIT_VIEW,
    layer: DEFAULT_COCKPIT_LAYER,
  }))

  return (
    <>
      <section className="card welcome-card">
        <img src="/logo-512.png" alt="ContextMine" className="welcome-logo" />
        <div className="welcome-content">
          <h2>Your AI's Knowledge Base</h2>
          <p className="welcome-tagline">Give Claude, Cursor, and any MCP-compatible AI assistant instant access to your documentation and codebase.</p>
          <div className="features-grid">
            <div className="feature-item"><span className="feature-icon">🔍</span><div className="feature-text"><strong>Semantic Search</strong><span>Hybrid FTS + vector search with smart ranking finds exactly what you need</span></div></div>
            <div className="feature-item"><span className="feature-icon">🔄</span><div className="feature-text"><strong>Auto-Sync</strong><span>GitHub repos and web docs stay current with scheduled incremental updates</span></div></div>
            <div className="feature-item"><span className="feature-icon">🔌</span><div className="feature-text"><strong>MCP Native</strong><span>Works with Claude Desktop, Cursor, VS Code, Cline, and Claude Code CLI</span></div></div>
            <div className="feature-item"><span className="feature-icon">🧠</span><div className="feature-text"><strong>Deep Research</strong><span>AI agent investigates complex questions across your entire knowledge base</span></div></div>
          </div>
        </div>
      </section>

      <div className="dashboard-grid">
        <div className="dashboard-left">
          <section className="card stats-card">
            <h2>Index Statistics</h2>
            <div className="stats-grid">
              <div className="stat-item"><span className="stat-value">{stats?.collections ?? '-'}</span><span className="stat-label">Collections</span></div>
              <div className="stat-item"><span className="stat-value">{stats?.sources ?? '-'}</span><span className="stat-label">Sources</span></div>
              <div className="stat-item"><span className="stat-value">{stats?.documents ?? '-'}</span><span className="stat-label">Documents</span></div>
              <div className="stat-item"><span className="stat-value">{stats?.chunks ?? '-'}</span><span className="stat-label">Chunks</span></div>
            </div>
            <div className="stats-bar">
              <div className="stats-bar-label"><span>Embeddings</span><span>{stats ? `${stats.embedded_chunks} / ${stats.chunks}` : '-'}</span></div>
              <div className="stats-bar-track"><div className="stats-bar-fill" style={{ width: stats && stats.chunks > 0 ? `${(stats.embedded_chunks / stats.chunks) * 100}%` : '0%' }} /></div>
            </div>
          </section>

          <section className="card">
            <h2>System Status</h2>
            <div className="status-row">
              <span className="label">API</span>
              {healthQuery.isLoading && <span className="status loading">Checking...</span>}
              {healthQuery.isError && <span className="status error">Error</span>}
              {health && <span className={`status ${health.status === 'ok' ? 'ok' : 'error'}`}>{health.status === 'ok' ? 'Healthy' : 'Unhealthy'}</span>}
            </div>
            <div className="status-row"><span className="label">Sync Runs</span><span className="status-counts"><span className="status-count ok">{stats?.runs_by_status?.success ?? 0} completed</span><span className="status-count warning">{stats?.runs_by_status?.running ?? 0} running</span><span className="status-count error">{stats?.runs_by_status?.failed ?? 0} failed</span></span></div>
          </section>

          <section className="card">
            <h2>Query Documentation</h2>
            <div className="query-mode-toggle">
              <button className={`mode-button ${queryMode === 'quick' ? 'active' : ''}`} onClick={() => setQueryMode('quick')} type="button">Quick Search</button>
              <button className={`mode-button ${queryMode === 'deep' ? 'active' : ''}`} onClick={() => setQueryMode('deep')} type="button">Deep Research</button>
            </div>
            <p className="mode-description">{queryMode === 'quick' ? 'Fast semantic search with LLM-synthesized answer from indexed documentation.' : 'Multi-step AI agent that searches, reads code, and investigates complex questions.'}</p>
            <form onSubmit={handleQuery} className="query-form">
              {queryMode === 'quick' && <div className="form-row"><select value={queryCollectionId} onChange={(event) => setQueryCollectionId(event.target.value)} className="collection-select"><option value="">All accessible collections</option>{collections.map((collection) => <option key={collection.id} value={collection.id}>{collection.name}</option>)}</select></div>}
              <div className="form-row"><textarea placeholder={queryMode === 'quick' ? "Enter your query... (e.g., 'How do I use the authentication API?')" : "Enter a complex question... (e.g., 'How does the error handling work in the API layer?')"} value={queryText} onChange={(event) => setQueryText(event.target.value)} className="query-input" rows={3} /></div>
              <button type="submit" className="query-button" disabled={queryLoading || !queryText.trim()}>{queryLoading ? researchStep || (queryMode === 'deep' ? 'Researching...' : 'Generating...') : queryMode === 'deep' ? 'Start Research' : 'Generate Context'}</button>
            </form>
            {queryError && <p className="query-error">{queryError}</p>}
          </section>

          {queryResult && <>
            <section className="card"><h2>Result</h2><div className="query-meta"><span>{queryMode === 'quick' ? `Used ${queryResult.chunks_used} chunks from ${queryResult.sources.length} sources` : `Research completed in ${queryResult.chunks_used} steps with ${researchCitations.length} citations`}</span></div><div className="markdown-content"><pre className="markdown-raw">{queryResult.markdown}</pre></div></section>
            {queryMode === 'quick' && queryResult.sources.length > 0 && <section className="card"><h2>Sources</h2><ul className="sources-list">{queryResult.sources.map((source) => <li key={source.uri} className="source-item"><a href={source.uri} target="_blank" rel="noopener noreferrer">{source.title}</a>{source.file_path && <span className="file-path">{source.file_path}</span>}</li>)}</ul></section>}
            {queryMode === 'deep' && researchCitations.length > 0 && <section className="card"><h2>Evidence Citations</h2><ul className="sources-list citations-list">{researchCitations.map((citation) => <li key={citation} className="source-item citation-item"><code>{citation}</code></li>)}</ul>{researchRunId && <p className="note">Run ID: {researchRunId}</p>}</section>}
          </>}
        </div>

        <div className="dashboard-right">
          <section className="card cockpit-cta-card"><div className="cockpit-cta-copy"><h2>Architecture Cockpit</h2><p>Inspect extracted views across Overview, Topology, C4 Diff, arc42/Ports/Drift, and Exports.</p></div><button type="button" className="cockpit-cta-button" onClick={openCockpit}>Open Cockpit</button></section>
          <section className="card coverage-ingest-card">
            <h2>GitHub Actions Coverage Ingest</h2>
            <p className="note">Coverage reports are pushed from CI. ContextMine validates commit SHA and applies coverage asynchronously to Twin metrics.</p>
            <ol className="coverage-ingest-steps"><li>Get the source ID from Collections → Source details.</li><li>Rotate a source ingest token once (owner session required).</li><li>Store the token as GitHub secret <code>CONTEXTMINE_INGEST_TOKEN</code>.</li><li>Post coverage reports from GitHub Actions after tests.</li></ol>
            <h3>Rotate Token (run in browser console while logged in)</h3>
            <pre className="config-block">{`await fetch("/api/sources/<SOURCE_ID>/metrics/coverage-ingest-token/rotate", {
  method: "POST",
  credentials: "include"
}).then((r) => r.json())`}</pre>
            <h3>GitHub Actions Snippet</h3>
            <pre className="config-block">{`- name: Push coverage to ContextMine
  if: always()
  env:
    CONTEXTMINE_URL: ${globalThis.location.origin}
    CONTEXTMINE_SOURCE_ID: \${{ secrets.CONTEXTMINE_SOURCE_ID }}
    CONTEXTMINE_INGEST_TOKEN: \${{ secrets.CONTEXTMINE_INGEST_TOKEN }}
  run: |
    curl --fail-with-body \
      -X POST "$CONTEXTMINE_URL/api/sources/$CONTEXTMINE_SOURCE_ID/metrics/coverage-ingest" \
      -H "X-ContextMine-Ingest-Token: $CONTEXTMINE_INGEST_TOKEN" \
      -F "commit_sha=\${{ github.sha }}" \
      -F "branch=\${{ github.ref_name }}" \
      -F "workflow_run_id=\${{ github.run_id }}" \
      -F "provider=github_actions" \
      -F "reports=@coverage/lcov.info" \
      -F "reports=@coverage/coverage.xml"`}</pre>
          </section>
          <section className="card">
            <h2>MCP Setup</h2>
            <p className="note">Connect your AI assistant to ContextMine. Authentication is handled via GitHub OAuth.</p>
            <h3>Claude Code (CLI)</h3><code className="usage-example">claude mcp add contextmine {globalThis.location.origin}/mcp</code>
            <h3>Claude Desktop / Cursor / Cline</h3><pre className="config-block">{`{
  "mcpServers": {
    "contextmine": {
      "url": "${globalThis.location.origin}/mcp"
    }
  }
}`}</pre>
            <h3>VS Code</h3><pre className="config-block">{`{
  "mcp": {
    "servers": {
      "contextmine": {
        "url": "${globalThis.location.origin}/mcp"
      }
    }
  }
}`}</pre>
          </section>
        </div>
      </div>
    </>
  )
}
