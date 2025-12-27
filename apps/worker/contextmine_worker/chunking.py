"""Chunking service for documents.

This module provides chunking functionality that:
- Preserves fenced code blocks intact
- Uses LangChain splitters for markdown and code
- Supports incremental chunk maintenance
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Default chunk settings
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200

# File extension to Language mapping
EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".jsx": Language.JS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".c": Language.C,
    ".cpp": Language.CPP,
    ".h": Language.C,
    ".hpp": Language.CPP,
    ".cs": Language.CSHARP,
    ".swift": Language.SWIFT,
    ".kt": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".html": Language.HTML,
    ".md": Language.MARKDOWN,
    ".mdx": Language.MARKDOWN,
    ".rst": Language.RST,
    ".sol": Language.SOL,
    ".proto": Language.PROTO,
}


@dataclass
class ChunkResult:
    """Result of chunking a document."""

    content: str
    chunk_index: int
    chunk_hash: str
    meta: dict


def compute_chunk_hash(content: str) -> str:
    """Compute SHA-256 hash of chunk content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def extract_code_blocks(text: str) -> list[tuple[int, int, str]]:
    """Extract fenced code blocks from text.

    Returns list of (start_pos, end_pos, block_content) tuples.
    """
    blocks = []
    # Match fenced code blocks (``` or ~~~)
    pattern = r"(```|~~~)(\w*)\n(.*?)\1"
    for match in re.finditer(pattern, text, re.DOTALL):
        blocks.append((match.start(), match.end(), match.group(0)))
    return blocks


def split_markdown_preserving_code_fences(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split markdown text while preserving fenced code blocks intact.

    This ensures no code fence is split across chunks.
    """
    # Find all code blocks
    code_blocks = extract_code_blocks(text)

    if not code_blocks:
        # No code blocks, use standard splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)

    # Build segments: alternating text and code blocks
    segments: list[tuple[str, bool]] = []  # (content, is_code_block)
    last_end = 0

    for start, end, block in code_blocks:
        # Add text before this code block
        if start > last_end:
            segments.append((text[last_end:start], False))
        # Add the code block
        segments.append((block, True))
        last_end = end

    # Add remaining text after last code block
    if last_end < len(text):
        segments.append((text[last_end:], False))

    # Now split non-code segments and combine
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: list[str] = []
    current_chunk = ""

    for content, is_code_block in segments:
        if is_code_block:
            # Code blocks are never split
            # Check if adding it to current chunk exceeds limit
            if current_chunk and len(current_chunk) + len(content) > chunk_size:
                # Flush current chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = content
            else:
                current_chunk += content
        else:
            # Split text segment
            if not content.strip():
                continue

            sub_chunks = text_splitter.split_text(content)
            for i, sub_chunk in enumerate(sub_chunks):
                if i == 0 and current_chunk:
                    # Try to append to current chunk
                    if len(current_chunk) + len(sub_chunk) <= chunk_size:
                        current_chunk += sub_chunk
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sub_chunk
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sub_chunk

    # Flush remaining
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def split_code_file(
    text: str,
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split a code file using language-aware splitter if available."""
    ext = Path(file_path).suffix.lower()
    language = EXTENSION_TO_LANGUAGE.get(ext)

    if language:
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            return splitter.split_text(text)
        except Exception:
            # Fall back to generic splitter
            pass

    # Generic fallback
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def is_markdown_file(file_path: str) -> bool:
    """Check if a file is a markdown file."""
    ext = Path(file_path).suffix.lower()
    return ext in {".md", ".mdx", ".rst", ".txt", ".adoc"}


def chunk_document(
    content: str,
    file_path: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkResult]:
    """Chunk a document into smaller pieces.

    Args:
        content: Document content
        file_path: Optional file path for language detection
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks

    Returns:
        List of ChunkResult objects
    """
    if not content or not content.strip():
        return []

    # Determine chunking strategy
    if file_path and not is_markdown_file(file_path):
        chunks = split_code_file(content, file_path, chunk_size, chunk_overlap)
    else:
        chunks = split_markdown_preserving_code_fences(content, chunk_size, chunk_overlap)

    # Build results
    results = []
    for i, chunk_content in enumerate(chunks):
        if not chunk_content.strip():
            continue

        chunk_hash = compute_chunk_hash(chunk_content)
        results.append(
            ChunkResult(
                content=chunk_content,
                chunk_index=i,
                chunk_hash=chunk_hash,
                meta={
                    "file_path": file_path,
                    "chunk_size": len(chunk_content),
                },
            )
        )

    return results


def chunk_with_headers(
    content: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkResult]:
    """Chunk markdown content using header-based splitting first.

    This uses MarkdownHeaderTextSplitter to split by headers,
    then RecursiveCharacterTextSplitter for large sections.
    """
    # Define headers to split on
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]

    # First split by headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    try:
        header_splits = header_splitter.split_text(content)
    except Exception:
        # Fall back to simple chunking
        return chunk_document(content, None, chunk_size, chunk_overlap)

    results = []
    chunk_index = 0

    for doc in header_splits:
        section_content = doc.page_content
        section_meta = doc.metadata

        if len(section_content) <= chunk_size:
            # Section fits in one chunk
            if section_content.strip():
                chunk_hash = compute_chunk_hash(section_content)
                results.append(
                    ChunkResult(
                        content=section_content,
                        chunk_index=chunk_index,
                        chunk_hash=chunk_hash,
                        meta=section_meta,
                    )
                )
                chunk_index += 1
        else:
            # Split large section, preserving code blocks
            sub_chunks = split_markdown_preserving_code_fences(
                section_content, chunk_size, chunk_overlap
            )
            for sub_chunk in sub_chunks:
                if sub_chunk.strip():
                    chunk_hash = compute_chunk_hash(sub_chunk)
                    results.append(
                        ChunkResult(
                            content=sub_chunk,
                            chunk_index=chunk_index,
                            chunk_hash=chunk_hash,
                            meta=section_meta,
                        )
                    )
                    chunk_index += 1

    return results
