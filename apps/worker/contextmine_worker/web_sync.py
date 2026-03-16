"""Web source sync service using spider_md for crawling and trafilatura for extraction."""

import hashlib
import json
import logging
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
import trafilatura
from lxml import html as lxml_html

logger = logging.getLogger(__name__)

# Maximum page content size (5MB per page - high because modern SPAs have large HTML)
MAX_PAGE_SIZE = 5 * 1024 * 1024

# Default rate limiting settings
DEFAULT_DELAY_MS = 500  # 500ms between requests
DEFAULT_MAX_PAGES = 100

# Default user agent string sent with crawl requests
DEFAULT_USER_AGENT = "ContextMine-Spider/0.1"


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


def _crawl_python(
    base_url: str,
    max_pages: int = DEFAULT_MAX_PAGES,
    user_agent: str = DEFAULT_USER_AGENT,
    delay_ms: int = DEFAULT_DELAY_MS,
    start_url: str | None = None,
) -> list[WebPage]:
    """Pure Python crawler using httpx + lxml for link extraction.

    BFS crawl starting from start_url, staying within base_url scope.
    Each page's HTML is extracted to markdown via trafilatura.
    """
    crawl_start = start_url or base_url
    visited: set[str] = set()
    queue: deque[str] = deque([crawl_start])
    pages: list[WebPage] = []

    with httpx.Client(
        follow_redirects=True,
        timeout=30.0,
        headers={"User-Agent": user_agent},
    ) as client:
        while queue and len(pages) < max_pages:
            url = queue.popleft()

            # Normalize: strip fragment
            parsed = urlparse(url)
            url = parsed._replace(fragment="").geturl()

            if url in visited:
                continue
            visited.add(url)

            if not is_url_in_scope(url, base_url):
                continue

            try:
                response = client.get(url)
                response.raise_for_status()
            except Exception as e:
                logger.debug("Failed to fetch %s: %s", url, e)
                continue

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                continue

            raw_html = response.text
            if not raw_html or len(raw_html.encode("utf-8")) > MAX_PAGE_SIZE:
                continue

            # Extract links from HTML for BFS crawl
            try:
                tree = lxml_html.fromstring(raw_html)
                tree.make_links_absolute(url)
                for element, _attr, link, _pos in tree.iterlinks():
                    if element.tag == "a":
                        link_parsed = urlparse(link)
                        clean_link = link_parsed._replace(fragment="").geturl()
                        if clean_link not in visited and is_url_in_scope(clean_link, base_url):
                            queue.append(clean_link)
            except Exception:
                pass

            # Extract markdown using trafilatura
            markdown = extract_markdown_with_trafilatura(raw_html)
            if not markdown or len(markdown.strip()) < 50:
                continue

            content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

            # Extract title
            title = ""
            try:
                title_els = tree.xpath("//title/text()")
                if title_els:
                    title = str(title_els[0]).strip()
            except Exception:
                pass

            pages.append(
                WebPage(
                    url=url,
                    title=title,
                    markdown=markdown,
                    content_hash=content_hash,
                    etag=response.headers.get("etag"),
                    last_modified=response.headers.get("last-modified"),
                )
            )

            logger.info("Crawled %s (%d/%d)", url, len(pages), max_pages)

            # Rate limiting
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

    return pages


def _crawl_spider_md(
    base_url: str,
    max_pages: int = DEFAULT_MAX_PAGES,
    user_agent: str = "ContextMine-Spider/0.1",
    delay_ms: int = DEFAULT_DELAY_MS,
    start_url: str | None = None,
) -> list[WebPage]:
    """Run the spider_md Rust binary and collect results."""
    crawl_start = start_url or base_url

    cmd = [
        "spider_md",
        "--base-url",
        base_url,
        "--start-url",
        crawl_start,
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
            timeout=600,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    if result.returncode != 0:
        return []

    pages = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
            html = data["html"]

            if len(html.encode("utf-8")) > MAX_PAGE_SIZE:
                continue

            markdown = extract_markdown_with_trafilatura(html)
            if not markdown or len(markdown.strip()) < 50:
                continue

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
            continue

    return pages


def run_spider_md(
    base_url: str,
    max_pages: int = DEFAULT_MAX_PAGES,
    user_agent: str = "ContextMine-Spider/0.1",
    delay_ms: int = DEFAULT_DELAY_MS,
    start_url: str | None = None,
) -> list[WebPage]:
    """Crawl a website and return extracted pages.

    Tries the spider_md Rust binary first. If it returns no pages
    (known issue with spider-rs crate), falls back to a pure Python
    crawler using httpx + lxml.
    """
    # Try Rust binary first
    pages = _crawl_spider_md(
        base_url=base_url,
        max_pages=max_pages,
        user_agent=user_agent,
        delay_ms=delay_ms,
        start_url=start_url,
    )

    if pages:
        logger.info("spider_md returned %d pages", len(pages))
        return pages

    # Fallback to Python crawler
    logger.info("spider_md returned 0 pages, falling back to Python crawler")
    return _crawl_python(
        base_url=base_url,
        max_pages=max_pages,
        user_agent=user_agent,
        delay_ms=delay_ms,
        start_url=start_url,
    )
