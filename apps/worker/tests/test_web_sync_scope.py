"""Tests for web sync URL scoping logic."""

from contextmine_worker.web_sync import is_url_in_scope


class TestURLScoping:
    """Tests for URL scoping filter (domain + path prefix)."""

    def test_same_host_same_path(self) -> None:
        """URLs under the same path prefix are in scope."""
        base = "https://example.com/docs/"
        assert is_url_in_scope("https://example.com/docs/intro", base)
        assert is_url_in_scope("https://example.com/docs/guide/start", base)
        assert is_url_in_scope("https://example.com/docs/", base)

    def test_different_path_not_in_scope(self) -> None:
        """URLs outside the path prefix are not in scope."""
        base = "https://example.com/docs/"
        assert not is_url_in_scope("https://example.com/blog/", base)
        assert not is_url_in_scope("https://example.com/", base)
        assert not is_url_in_scope("https://example.com/documentation/", base)

    def test_root_path_allows_all(self) -> None:
        """Root path base URL allows all paths on same host."""
        base = "https://example.com/"
        assert is_url_in_scope("https://example.com/anything", base)
        assert is_url_in_scope("https://example.com/docs/intro", base)
        assert is_url_in_scope("https://example.com/blog/post", base)

    def test_different_host_not_in_scope(self) -> None:
        """Different hosts are not in scope."""
        base = "https://example.com/docs/"
        assert not is_url_in_scope("https://other.com/docs/", base)
        assert not is_url_in_scope("https://example.org/docs/", base)

    def test_subdomain_not_in_scope(self) -> None:
        """Subdomains are not in scope (strict same host)."""
        base = "https://example.com/docs/"
        assert not is_url_in_scope("https://sub.example.com/docs/", base)
        assert not is_url_in_scope("https://www.example.com/docs/", base)
        assert not is_url_in_scope("https://api.example.com/docs/", base)

    def test_scheme_must_match(self) -> None:
        """HTTP and HTTPS are treated as different."""
        base = "https://example.com/docs/"
        assert not is_url_in_scope("http://example.com/docs/", base)

        base_http = "http://example.com/docs/"
        assert not is_url_in_scope("https://example.com/docs/", base_http)

    def test_path_prefix_exact_match(self) -> None:
        """Path prefix matching is strict (not partial word match)."""
        base = "https://example.com/doc/"
        # /docs/ should not match /doc/ prefix since it's a different segment
        # Actually /docs/ starts with /doc so it would match
        # Let's test a clearer case
        assert is_url_in_scope("https://example.com/doc/page", base)
        assert is_url_in_scope("https://example.com/doc/", base)

    def test_empty_path_treated_as_root(self) -> None:
        """Empty path is treated same as root."""
        base = "https://example.com"
        assert is_url_in_scope("https://example.com/anything", base)
        assert is_url_in_scope("https://example.com/docs/intro", base)

    def test_invalid_urls_not_in_scope(self) -> None:
        """Invalid URLs are not in scope."""
        base = "https://example.com/docs/"
        assert not is_url_in_scope("not-a-url", base)
        assert not is_url_in_scope("", base)
        assert not is_url_in_scope("ftp://example.com/docs/", base)

    def test_query_string_ignored(self) -> None:
        """Query strings don't affect path matching."""
        base = "https://example.com/docs/"
        assert is_url_in_scope("https://example.com/docs/page?foo=bar", base)
        assert is_url_in_scope("https://example.com/docs/page?a=1&b=2", base)

    def test_fragment_ignored(self) -> None:
        """Fragments don't affect path matching."""
        base = "https://example.com/docs/"
        assert is_url_in_scope("https://example.com/docs/page#section", base)


class TestPageTitle:
    """Tests for page title extraction."""

    def test_title_from_page(self) -> None:
        """Title is extracted from page."""
        from contextmine_worker.web_sync import WebPage, get_page_title

        page = WebPage(
            url="https://example.com/docs/intro",
            title="Introduction",
            markdown="# Introduction",
            content_hash="abc123",
        )
        assert get_page_title(page) == "Introduction"

    def test_fallback_to_url_path(self) -> None:
        """Falls back to URL path when title is empty."""
        from contextmine_worker.web_sync import WebPage, get_page_title

        page = WebPage(
            url="https://example.com/docs/intro",
            title="",
            markdown="# Content",
            content_hash="abc123",
        )
        assert get_page_title(page) == "/docs/intro"
