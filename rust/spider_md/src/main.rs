//! spider_md - Web crawler that converts HTML to Markdown with strict URL scoping
//!
//! This CLI tool crawls websites starting from a base URL, staying strictly within
//! the same hostname and path prefix. It converts HTML to Markdown and outputs
//! JSON lines format.

use clap::Parser;
use hex;
use html2md::rewrite_html;
use serde::Serialize;
use sha2::{Digest, Sha256};
use spider::configuration::Configuration;
use spider::website::Website;
use std::io::{self, Write};
use url::Url;

#[derive(Parser, Debug)]
#[command(name = "spider_md")]
#[command(about = "Web crawler that converts HTML to Markdown with strict URL scoping")]
struct Args {
    /// Base URL for scoping (only pages within this path prefix are included)
    #[arg(long)]
    base_url: String,

    /// URL to start crawling from (defaults to base_url if not provided)
    #[arg(long)]
    start_url: Option<String>,

    /// Maximum number of pages to crawl
    #[arg(long, default_value = "100")]
    max_pages: usize,

    /// User agent string
    #[arg(long, default_value = "Mozilla/5.0 (compatible; ContextMine/1.0; +https://github.com/mayflower/contextmine)")]
    user_agent: String,

    /// Request delay in milliseconds
    #[arg(long, default_value = "100")]
    delay_ms: u64,

    /// Disable HTTP caching (by default caching respects Cache-Control, ETag, Last-Modified)
    #[arg(long)]
    no_cache: bool,
}

#[derive(Serialize)]
struct PageOutput {
    url: String,
    title: String,
    markdown: String,
    content_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    etag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_modified: Option<String>,
}

/// Check if a URL is within the allowed scope (same host + path prefix)
fn is_url_in_scope(url: &str, base_url: &Url) -> bool {
    let parsed = match Url::parse(url) {
        Ok(u) => u,
        Err(_) => return false,
    };

    // Must be same host (no subdomains allowed)
    if parsed.host_str() != base_url.host_str() {
        return false;
    }

    // Must be same scheme
    if parsed.scheme() != base_url.scheme() {
        return false;
    }

    // Must have same or deeper path prefix
    let base_path = base_url.path();
    let url_path = parsed.path();

    // Normalize paths - ensure base path ends considerations
    if base_path == "/" {
        return true; // Root path allows all paths on same host
    }

    // URL path must start with base path
    url_path.starts_with(base_path)
}

/// Extract title from HTML
fn extract_title(html: &str) -> String {
    // Simple title extraction - look for <title> tag
    if let Some(start) = html.find("<title>") {
        if let Some(end) = html[start..].find("</title>") {
            let title_start = start + 7;
            let title_end = start + end;
            if title_end > title_start {
                return html[title_start..title_end].trim().to_string();
            }
        }
    }
    // Fallback: try case-insensitive
    let lower = html.to_lowercase();
    if let Some(start) = lower.find("<title>") {
        if let Some(end) = lower[start..].find("</title>") {
            let title_start = start + 7;
            let title_end = start + end;
            if title_end > title_start {
                return html[title_start..title_end].trim().to_string();
            }
        }
    }
    String::new()
}

/// Compute SHA-256 hash of content
fn compute_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Parse and validate base URL (used for scoping)
    let base_url = match Url::parse(&args.base_url) {
        Ok(u) => u,
        Err(e) => {
            eprintln!("Invalid base URL: {}", e);
            std::process::exit(1);
        }
    };

    // Determine start URL (defaults to base_url if not provided)
    let start_url_str = args.start_url.as_deref().unwrap_or(&args.base_url);

    // Validate start URL
    let start_url = match Url::parse(start_url_str) {
        Ok(u) => u,
        Err(e) => {
            eprintln!("Invalid start URL: {}", e);
            std::process::exit(1);
        }
    };

    // Ensure HTTPS or HTTP
    if base_url.scheme() != "https" && base_url.scheme() != "http" {
        eprintln!("URL must use http or https scheme");
        std::process::exit(1);
    }

    // Start URL must be within base URL scope
    if !is_url_in_scope(start_url.as_str(), &base_url) {
        eprintln!(
            "Start URL {} is not within base URL scope {}",
            start_url, base_url
        );
        std::process::exit(1);
    }

    // Configure spider
    let mut config = Configuration::new();
    config.with_user_agent(Some(&args.user_agent));
    config.with_respect_robots_txt(true);
    config.with_delay(args.delay_ms);
    config.with_request_timeout(Some(std::time::Duration::from_secs(30)));
    // Limit depth, concurrency, and budget for controlled crawling
    config.with_depth(5);
    // Budget limits total pages crawled
    let mut budget = spider::hashbrown::HashMap::new();
    budget.insert("*", args.max_pages as u32);
    config.with_budget(Some(budget));
    // Enable HTTP caching (respects Cache-Control, ETag, Last-Modified headers)
    // Caching is enabled by default, use --no-cache to disable
    config.with_caching(!args.no_cache);

    // Create website crawler starting from start_url
    let mut website = Website::new(start_url.as_str());
    website.with_config(config);

    // Scrape the website (async) - scrape() collects page content, crawl() only collects links
    website.scrape().await;

    let stdout = io::stdout();
    let mut handle = stdout.lock();

    // Get pages - handle Option properly
    let pages = match website.get_pages() {
        Some(p) => p,
        None => return,
    };

    let mut page_count = 0;

    // Process each page (with max_pages limit)
    for page in pages.iter() {
        if page_count >= args.max_pages {
            break;
        }
        let page_url = page.get_url();

        // Filter by scope
        if !is_url_in_scope(page_url, &base_url) {
            continue;
        }

        let html = page.get_html();

        // Skip empty pages
        if html.is_empty() {
            continue;
        }

        // Extract title
        let title = extract_title(&html);

        // Convert to Markdown (using CommonMark format)
        let markdown = rewrite_html(&html, true);

        // Skip pages with no meaningful content
        if markdown.trim().is_empty() {
            continue;
        }

        // Compute hash
        let content_hash = compute_hash(&markdown);

        // Extract HTTP cache headers if available
        let (etag, last_modified) = if let Some(ref headers) = page.headers {
            let etag: Option<String> = headers
                .get("etag")
                .or_else(|| headers.get("ETag"))
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            let last_modified: Option<String> = headers
                .get("last-modified")
                .or_else(|| headers.get("Last-Modified"))
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            (etag, last_modified)
        } else {
            (None, None)
        };

        // Create output
        let output = PageOutput {
            url: page_url.to_string(),
            title,
            markdown,
            content_hash,
            etag,
            last_modified,
        };

        // Write JSON line
        if let Ok(json) = serde_json::to_string(&output) {
            let _ = writeln!(handle, "{}", json);
            page_count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_in_scope_same_host() {
        let base = Url::parse("https://example.com/docs/").unwrap();
        assert!(is_url_in_scope("https://example.com/docs/intro", &base));
        assert!(is_url_in_scope("https://example.com/docs/guide/start", &base));
        assert!(!is_url_in_scope("https://example.com/blog/", &base));
        assert!(!is_url_in_scope("https://other.com/docs/", &base));
    }

    #[test]
    fn test_url_in_scope_root_path() {
        let base = Url::parse("https://example.com/").unwrap();
        assert!(is_url_in_scope("https://example.com/anything", &base));
        assert!(is_url_in_scope("https://example.com/docs/intro", &base));
        assert!(!is_url_in_scope("https://other.com/", &base));
    }

    #[test]
    fn test_url_in_scope_no_subdomain() {
        let base = Url::parse("https://example.com/docs/").unwrap();
        assert!(!is_url_in_scope("https://sub.example.com/docs/", &base));
        assert!(!is_url_in_scope("https://www.example.com/docs/", &base));
    }

    #[test]
    fn test_url_in_scope_scheme_match() {
        let base = Url::parse("https://example.com/docs/").unwrap();
        assert!(!is_url_in_scope("http://example.com/docs/", &base));
    }

    #[test]
    fn test_extract_title() {
        let html = "<html><head><title>My Page</title></head><body></body></html>";
        assert_eq!(extract_title(html), "My Page");
    }

    #[test]
    fn test_extract_title_case_insensitive() {
        let html = "<html><head><TITLE>My Page</TITLE></head><body></body></html>";
        assert_eq!(extract_title(html), "My Page");
    }

    #[test]
    fn test_compute_hash() {
        let hash1 = compute_hash("Hello, world!");
        let hash2 = compute_hash("Hello, world!");
        let hash3 = compute_hash("Different content");
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 64); // SHA-256 = 64 hex chars
    }
}
