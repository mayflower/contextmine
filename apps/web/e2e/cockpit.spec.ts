import { expect, test, type Page, type Route } from '@playwright/test'

type Scenario = {
  id: string
  name: string
  version: number
  is_as_is: boolean
}

interface MockOptions {
  collections?: Array<{ id: string; name: string }>
  scenariosByCollection?: Record<string, Scenario[]>
  cityDelayMs?: number
  failCityOnce?: boolean
  cityFailCount?: number
}

const DEFAULT_COLLECTIONS = [{ id: 'col-1', name: 'Alpha Project' }]
const DEFAULT_SCENARIOS: Scenario[] = [
  { id: 'scn-asis', name: 'AS-IS Baseline', version: 1, is_as_is: true },
  { id: 'scn-tobe', name: 'TO-BE Split', version: 2, is_as_is: false },
]

function json(route: Route, payload: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(payload),
  })
}

async function mockApi(page: Page, options: MockOptions = {}) {
  let cityFailuresRemaining = options.cityFailCount ?? (options.failCityOnce ? 1 : 0)
  const collections = options.collections ?? DEFAULT_COLLECTIONS
  const scenariosByCollection = options.scenariosByCollection ?? { 'col-1': DEFAULT_SCENARIOS }

  await page.route('**/api/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname

    if (path === '/api/auth/me') {
      return json(route, {
        id: 'user-1',
        github_login: 'architect',
        name: 'Architecture Owner',
        avatar_url: null,
      })
    }

    if (path === '/api/health') {
      return json(route, { status: 'ok' })
    }

    if (path === '/api/stats') {
      return json(route, {
        collections: collections.length,
        sources: 3,
        documents: 120,
        chunks: 560,
        embedded_chunks: 520,
        runs_by_status: { success: 7, running: 1, failed: 0 },
        recent_runs: [],
      })
    }

    if (path === '/api/collections') {
      return json(
        route,
        collections.map((collection) => ({
          id: collection.id,
          slug: collection.name.toLowerCase().replace(/\s+/g, '-'),
          name: collection.name,
          visibility: 'private',
          owner_id: 'user-1',
          owner_github_login: 'architect',
          created_at: '2026-02-16T00:00:00Z',
          is_owner: true,
          member_count: 1,
        })),
      )
    }

    if (path === '/api/twin/scenarios') {
      const collectionId = url.searchParams.get('collection_id') || ''
      return json(route, {
        scenarios: scenariosByCollection[collectionId] ?? [],
      })
    }

    if (path.includes('/views/city')) {
      if (options.cityDelayMs) {
        await new Promise((resolve) => setTimeout(resolve, options.cityDelayMs))
      }

      if (cityFailuresRemaining > 0) {
        cityFailuresRemaining -= 1
        return json(route, { detail: 'city endpoint failed' }, 500)
      }

      return json(route, {
        collection_id: 'col-1',
        scenario: {
          id: 'scn-asis',
          collection_id: 'col-1',
          name: 'AS-IS Baseline',
          version: 1,
          is_as_is: true,
          base_scenario_id: null,
        },
        summary: {
          metric_nodes: 3,
          coverage_avg: 78.3,
          complexity_avg: 12.5,
          coupling_avg: 3.4,
        },
        hotspots: [
          {
            node_natural_key: 'billing.service.InvoiceService',
            loc: 480,
            symbol_count: 18,
            coverage: 61.5,
            complexity: 24.2,
            coupling: 6.7,
          },
          {
            node_natural_key: 'billing.api.InvoiceController',
            loc: 210,
            symbol_count: 9,
            coverage: 75.9,
            complexity: 9.4,
            coupling: 3.1,
          },
        ],
        cc_json: {
          project: 'alpha',
          generated: true,
        },
      })
    }

    if (path.includes('/views/topology') || path.includes('/views/deep-dive')) {
      return json(route, {
        collection_id: 'col-1',
        scenario: {
          id: 'scn-asis',
          collection_id: 'col-1',
          name: 'AS-IS Baseline',
          version: 1,
          is_as_is: true,
          base_scenario_id: null,
        },
        layer: url.searchParams.get('layer'),
        graph: {
          nodes: [
            {
              id: 'n-1',
              natural_key: 'billing',
              kind: 'service',
              name: 'Billing',
              meta: {},
            },
            {
              id: 'n-2',
              natural_key: 'payments',
              kind: 'service',
              name: 'Payments',
              meta: {},
            },
            {
              id: 'n-3',
              natural_key: 'ledger',
              kind: 'service',
              name: 'Ledger',
              meta: {},
            },
          ],
          edges: [
            { id: 'e-1', source_node_id: 'n-1', target_node_id: 'n-2', kind: 'depends_on', meta: {} },
            { id: 'e-2', source_node_id: 'n-2', target_node_id: 'n-3', kind: 'depends_on', meta: {} },
          ],
          page: 0,
          limit: Number(url.searchParams.get('limit') || 0),
          total_nodes: 3,
        },
      })
    }

    if (path.includes('/views/mermaid')) {
      return json(route, {
        collection_id: 'col-1',
        scenario: {
          id: 'scn-tobe',
          collection_id: 'col-1',
          name: 'TO-BE Split',
          version: 2,
          is_as_is: false,
          base_scenario_id: 'scn-asis',
        },
        mode: 'compare',
        as_is: 'C4Context\nPerson(user, "User")',
        to_be: 'C4Context\nSystem_Boundary(ctx, "Billing")',
        as_is_scenario_id: 'scn-asis',
      })
    }

    if (/\/api\/twin\/scenarios\/[^/]+\/exports$/.test(path) && request.method() === 'POST') {
      return json(route, { id: 'exp-1', status: 'ready' })
    }

    if (/\/api\/twin\/scenarios\/[^/]+\/exports\/[^/]+$/.test(path)) {
      return json(route, {
        id: 'exp-1',
        name: 'cockpit.cc.json',
        content: '{"generated": true, "format": "cc_json"}',
      })
    }

    return json(route, {})
  })
}

test('discoverability: sidebar and dashboard CTA open Architecture Cockpit', async ({ page }) => {
  await mockApi(page)
  await page.goto('/')

  await expect(page.locator('.sidebar li', { hasText: 'Architecture Cockpit' })).toBeVisible()
  await page.getByRole('button', { name: 'Open Cockpit' }).click()

  await expect(page).toHaveURL(/page=cockpit/)
  await expect(page.getByRole('heading', { name: 'Architecture Cockpit' })).toBeVisible()
  await expect(page.getByRole('tab', { name: 'Overview' })).toBeVisible()
})

test('url state preselects collection, scenario, view, and layer', async ({ page }) => {
  await mockApi(page)
  await page.goto('/?page=cockpit&collection=col-1&scenario=scn-tobe&view=deep_dive&layer=code_controlflow')

  await expect(page.getByRole('tab', { name: 'Deep Dive' })).toHaveAttribute('aria-selected', 'true')
  await expect(page.locator('.cockpit2-command-grid label:has-text("Project") select')).toHaveValue('col-1')
  await expect(page.locator('.cockpit2-command-grid label:has-text("Scenario") select')).toHaveValue('scn-tobe')
  await expect(page.locator('.cockpit2-command-grid label:has-text("Layer") select')).toHaveValue('code_controlflow')
})

test('empty state: no scenario onboarding is shown', async ({ page }) => {
  await mockApi(page, {
    scenariosByCollection: { 'col-1': [] },
  })
  await page.goto('/?page=cockpit&collection=col-1')

  await expect(page.getByText('No scenarios found for this project')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Run sync from Runs' })).toBeVisible()
})

test('overview renders skeleton first and then KPI/table content', async ({ page }) => {
  await mockApi(page, { cityDelayMs: 550 })
  await page.goto('/?page=cockpit&collection=col-1&scenario=scn-asis&view=overview')

  await expect(page.locator('.cockpit2-skeleton-card').first()).toBeVisible()
  await expect(page.getByText('System health summary')).toBeVisible()
  await expect(page.getByText('Top hotspots')).toBeVisible()
  await expect(page.getByText('billing.service.InvoiceService')).toBeVisible()
})

test('topology/deep dive show layer selector; overview hides it; graph metadata is visible', async ({ page }) => {
  await mockApi(page)
  await page.goto('/?page=cockpit&collection=col-1&scenario=scn-asis&view=topology')

  await expect(page.locator('.cockpit2-command-grid label:has-text("Layer") select')).toBeVisible()
  await expect(page.getByText('Nodes: 3 / Total: 3 • Edges: 2')).toBeVisible()

  await page.getByRole('tab', { name: 'Overview' }).click()
  await expect(page.locator('.cockpit2-command-grid label:has-text("Layer") select')).toHaveCount(0)

  await page.getByRole('tab', { name: 'Deep Dive' }).click()
  await expect(page.locator('.cockpit2-command-grid label:has-text("Layer") select')).toBeVisible()
  await expect(page.getByText('Nodes: 3 / Total: 3 • Edges: 2')).toBeVisible()
})

test('c4 diff shows AS-IS and TO-BE compare panes', async ({ page }) => {
  await mockApi(page)
  await page.goto('/?page=cockpit&collection=col-1&scenario=scn-tobe&view=c4_diff')

  await expect(page.getByText('AS-IS', { exact: true })).toBeVisible()
  await expect(page.getByText('TO-BE', { exact: true })).toBeVisible()
  await expect(page.locator('.cockpit2-compare-grid .badge.asis')).toBeVisible()
  await expect(page.locator('.cockpit2-compare-grid .badge.tobe')).toBeVisible()
  await expect(page.getByText('System_Boundary(ctx, "Billing")')).toBeVisible()
})

test('exports view can generate output and supports copy/download actions', async ({ page }) => {
  await mockApi(page)
  await page.goto('/?page=cockpit&collection=col-1&scenario=scn-tobe&view=exports')

  await page.getByRole('button', { name: 'Generate export' }).click()
  await expect(page.getByText('"generated": true')).toBeVisible()

  await page.getByRole('button', { name: 'Copy' }).click()

  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Download' }).click(),
  ])
  expect(download.suggestedFilename()).toContain('cockpit-export')
})

test('overview error handling shows inline error and supports retry', async ({ page }) => {
  await mockApi(page, { cityFailCount: 2 })
  await page.goto('/?page=cockpit&collection=col-1&scenario=scn-asis&view=overview')

  await expect(page.getByText('Overview request failed')).toBeVisible()
  await page.getByRole('button', { name: 'Retry' }).first().click()
  await expect(page.getByText('System health summary')).toBeVisible()
})

test('accessibility smoke: tabs are keyboard focusable and aria-selected updates', async ({ page }) => {
  await mockApi(page)
  await page.goto('/?page=cockpit&collection=col-1&scenario=scn-asis&view=overview')

  const overviewTab = page.getByRole('tab', { name: 'Overview' })
  const topologyTab = page.getByRole('tab', { name: 'Topology' })

  await expect(overviewTab).toHaveAttribute('aria-selected', 'true')
  await topologyTab.focus()
  await page.keyboard.press('Enter')
  await expect(topologyTab).toHaveAttribute('aria-selected', 'true')
})
