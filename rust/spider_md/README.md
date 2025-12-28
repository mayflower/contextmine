# spider_md

A web crawler that converts HTML to Markdown with strict URL scoping.

## Features

- Respects robots.txt
- Strict URL scoping: same hostname + path prefix only (no subdomains)
- Converts HTML to Markdown using html2md
- Outputs JSON lines format for easy processing
- Content hashing for incremental updates

## Building

```bash
# From the rust/spider_md directory
cargo build --release

# The binary will be at target/release/spider_md
```

## Usage

```bash
spider_md --base-url https://example.com/docs/ --max-pages 100
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--base-url` | Base URL to start crawling from | Required |
| `--max-pages` | Maximum number of pages to crawl | 100 |
| `--user-agent` | User agent string | ContextMine-Spider/0.1 |
| `--delay-ms` | Request delay in milliseconds | 100 |

### Output Format

Each line is a JSON object:

```json
{"url": "https://example.com/docs/intro", "title": "Introduction", "markdown": "# Introduction\n...", "content_hash": "abc123..."}
```

## URL Scoping Rules

The crawler enforces strict URL scoping:

1. **Same hostname only** - No subdomains allowed
   - `https://example.com/docs/` will NOT crawl `https://sub.example.com/docs/`

2. **Path prefix matching** - Only URLs under the base path are crawled
   - `https://example.com/docs/` will crawl `https://example.com/docs/intro`
   - `https://example.com/docs/` will NOT crawl `https://example.com/blog/`

3. **Scheme matching** - HTTP and HTTPS are treated as different
   - `https://example.com/` will NOT crawl `http://example.com/`

## Running Tests

```bash
cargo test
```

## Docker Build

The binary is built as part of the worker Docker image. See the main docker-compose.yml for details.
