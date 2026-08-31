import assert from 'node:assert/strict'
import { lookup } from 'node:dns/promises'

import { chromium } from 'playwright'

const webUrl = process.env.CONTEXTMINE_SMOKE_WEB_URL
assert.ok(webUrl, 'CONTEXTMINE_SMOKE_WEB_URL is required')

const browserUrl = process.env.CONTEXTMINE_SMOKE_BROWSER_URL
assert.ok(browserUrl, 'CONTEXTMINE_SMOKE_BROWSER_URL is required')

const resolvedBrowserUrl = new URL(browserUrl)
const browserAddress = await lookup(resolvedBrowserUrl.hostname)
resolvedBrowserUrl.hostname = browserAddress.address

async function connectToBrowser() {
  let lastError

  for (let attempt = 1; attempt <= 60; attempt += 1) {
    try {
      return await chromium.connectOverCDP(resolvedBrowserUrl.href)
    } catch (error) {
      lastError = error
      await new Promise((resolve) => setTimeout(resolve, 1000))
    }
  }

  throw new Error(`browser did not become ready: ${lastError}`)
}

const browser = await connectToBrowser()

try {
  const context = browser.contexts()[0] ?? await browser.newContext()
  const page = await context.newPage()
  const pageErrors = []
  const failedRequests = []

  page.on('pageerror', (error) => pageErrors.push(error.message))
  page.on('requestfailed', (request) => {
    failedRequests.push(`${request.method()} ${request.url()}: ${request.failure()?.errorText}`)
  })

  const response = await page.goto(webUrl, { waitUntil: 'networkidle' })
  assert.equal(response?.status(), 200, 'web root must return HTTP 200')

  await page.getByRole('heading', { name: 'ContextMine', exact: true }).waitFor()
  await page.getByRole('button', { name: 'Sign in with GitHub' }).waitFor()

  const logo = page.getByRole('img', { name: 'ContextMine' })
  await logo.waitFor()
  assert.equal(
    await logo.evaluate((element) => element.complete && element.naturalWidth > 0),
    true,
    'ContextMine logo must load successfully',
  )

  const healthResponse = await page.request.get(`${webUrl}/api/health`)
  assert.equal(healthResponse.status(), 200, 'web API proxy must return HTTP 200')
  assert.deepEqual(await healthResponse.json(), { status: 'ok' })

  assert.deepEqual(pageErrors, [], `browser page errors: ${pageErrors.join('; ')}`)
  assert.deepEqual(
    failedRequests,
    [],
    `browser request failures: ${failedRequests.join('; ')}`,
  )

  console.log(
    JSON.stringify({
      api_proxy: 'pass',
      browser: 'chromium',
      login_surface: 'pass',
      static_assets: 'pass',
      web_url: webUrl,
    }),
  )
} finally {
  await browser.close()
}
