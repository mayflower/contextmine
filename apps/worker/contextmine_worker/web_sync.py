"""Web source sync service using spider_md for crawling and trafilatura for extraction."""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from urllib.parse import urlparse

import trafilatura

# Maximum page content size (500KB per page)
MAX_PAGE_SIZE = 500 * 1024

# Default rate limiting settings
DEFAULT_DELAY_MS = 500  # 500ms between requests
DEFAULT_MAX_PAGES = 100


@dataclass
class WebPage:
    """Represents a crawled web page."""

    url: str
    title: str
    markdown: str
    content_hash: str
    etag: str | None = None
    last_modified: str | None = None


@dataclass
class WebSyncStats:
    """Statistics from a web sync operation."""

    pages_crawled: int = 0
    pages_skipped: int = 0
    docs_created: int = 0
    docs_updated: int = 0
    docs_deleted: int = 0
    errors: list[str] = field(default_factory=list)


def is_url_in_scope(url: str, base_url: str) -> bool:
    """Check if a URL is within the allowed scope.

    Rules:
    - Same hostname (no subdomains)
    - Same scheme
    - Path must start with base path prefix
    """
    try:
        parsed_url = urlparse(url)
        parsed_base = urlparse(base_url)
    except Exception:
        return False

    # Must be same host (no subdomains)
    if parsed_url.netloc != parsed_base.netloc:
        return False

    # Must be same scheme
    if parsed_url.scheme != parsed_base.scheme:
        return False

    # Must have same or deeper path prefix
    base_path = parsed_base.path
    url_path = parsed_url.path

    # Normalize paths
    if base_path == "/" or base_path == "":
        return True  # Root path allows all paths on same host

    # URL path must start with base path
    return url_path.startswith(base_path)


def get_page_title(page: WebPage) -> str:
    """Get a display title for a page."""
    if page.title:
        return page.title
    # Fall back to URL path
    parsed = urlparse(page.url)
    return parsed.path or page.url


def extract_markdown_with_trafilatura(html: str) -> str | None:
    """Extract clean markdown from HTML using trafilatura.

    Args:
        html: Raw HTML content

    Returns:
        Extracted markdown text, or None if extraction fails
    """
    if not html:
        return None

    try:
        text = trafilatura.extract(
            html,
            include_links=True,
            include_formatting=True,
            include_tables=True,
            output_format="markdown",
        )
        return text
    except Exception:
        return None


def run_spider_md(
    base_url: str,
    max_pages: int = DEFAULT_MAX_PAGES,
    user_agent: str = "ContextMine-Spider/0.1",
    delay_ms: int = DEFAULT_DELAY_MS,
    start_url: str | None = None,
) -> list[WebPage]:
    """Run the spider_md binary and collect results.

    Args:
        base_url: URL path prefix for scoping (determines which pages to include)
        max_pages: Maximum pages to crawl (default 100)
        user_agent: User agent string
        delay_ms: Delay between requests in milliseconds (default 500ms)
        start_url: URL to start crawling from (defaults to base_url if not provided)

    Returns:
        List of WebPage objects from crawl results
    """
    # Use start_url if provided, otherwise fall back to base_url
    crawl_start = start_url or base_url

    cmd = [
        "spider_md",
        "--base-url",
        base_url,  # Used for scoping (path prefix filter)
        "--start-url",
        crawl_start,  # Used as crawling entry point
        "--max-pages",
        str(max_pages),
        "--user-agent",
        user_agent,
        "--delay-ms",
        str(delay_ms),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Spider crawl timed out after 10 minutes") from e
    except FileNotFoundError as e:
        raise RuntimeError("spider_md binary not found - is it installed?") from e

    if result.returncode != 0:
        stderr = result.stderr[:500] if result.stderr else "No error output"
        raise RuntimeError(f"Spider crawl failed: {stderr}")

    # Parse JSON lines output
    pages = []
    skipped_too_large = 0

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
            html = data["html"]

            # Skip pages that are too large
            if len(html.encode("utf-8")) > MAX_PAGE_SIZE:
                skipped_too_large += 1
                continue

            # Extract markdown using trafilatura
            markdown = extract_markdown_with_trafilatura(html)

            # Skip if extraction failed or produced empty content
            if not markdown or len(markdown.strip()) < 50:
                continue

            # Compute hash of extracted markdown (not raw HTML)
            content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

            pages.append(
                WebPage(
                    url=data["url"],
                    title=data.get("title", ""),
                    markdown=markdown,
                    content_hash=content_hash,
                    etag=data.get("etag"),
                    last_modified=data.get("last_modified"),
                )
            )
        except (json.JSONDecodeError, KeyError):
            # Skip malformed lines
            continue

    return pages
