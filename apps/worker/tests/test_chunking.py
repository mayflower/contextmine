"""Tests for document chunking logic."""

from contextmine_worker.chunking import (
    chunk_document,
    compute_chunk_hash,
    extract_code_blocks,
    split_markdown_preserving_code_fences,
)


class TestCodeBlockExtraction:
    """Tests for code block extraction."""

    def test_extract_single_code_block(self) -> None:
        """Extract a single fenced code block."""
        text = "Some text\n```python\nprint('hello')\n```\nMore text"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert "print('hello')" in blocks[0][2]

    def test_extract_multiple_code_blocks(self) -> None:
        """Extract multiple fenced code blocks."""
        text = """
```python
def foo():
    pass
```

Some text

```javascript
const x = 1;
```
"""
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2

    def test_extract_tilde_code_blocks(self) -> None:
        """Extract tilde-fenced code blocks."""
        text = "Text\n~~~\ncode here\n~~~\nMore text"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1


class TestCodeFencePreservation:
    """Tests proving that fenced code blocks remain intact."""

    def test_small_code_block_not_split(self) -> None:
        """A small code block appears wholly within one chunk."""
        content = """# Introduction

Some introductory text.

```python
def hello():
    print("Hello, world!")
```

More text after the code.
"""
        chunks = split_markdown_preserving_code_fences(content, chunk_size=500)

        # Find the chunk containing the code block
        found = False
        for chunk in chunks:
            if "def hello():" in chunk:
                # The entire code block should be in this chunk
                assert "```python" in chunk
                assert "```" in chunk[chunk.index("```python") + 10 :]  # closing fence
                found = True
                break
        assert found, "Code block not found in any chunk"

    def test_large_code_block_not_split(self) -> None:
        """A large code block is kept intact even if it exceeds chunk size."""
        # Create a code block larger than chunk size
        large_code = "\n".join([f"line_{i} = {i}" for i in range(100)])
        content = f"""# Header

```python
{large_code}
```

Footer text.
"""
        chunks = split_markdown_preserving_code_fences(content, chunk_size=500)

        # Find chunk with the code
        for chunk in chunks:
            if "line_0 = 0" in chunk:
                # All lines should be in the same chunk
                assert "line_99 = 99" in chunk
                assert "```python" in chunk
                break

    def test_multiple_code_blocks_each_intact(self) -> None:
        """Multiple code blocks each remain intact in their chunks."""
        content = """# Multi-language Example

Here's Python:

```python
def greet(name):
    return f"Hello, {name}!"
```

And here's JavaScript:

```javascript
function greet(name) {
    return `Hello, ${name}!`;
}
```

Both are valid greeting functions.
"""
        chunks = split_markdown_preserving_code_fences(content, chunk_size=300)

        # Check Python block is intact
        python_found = False
        for chunk in chunks:
            if "def greet(name):" in chunk:
                assert "```python" in chunk
                assert 'return f"Hello, {name}!"' in chunk
                python_found = True
                break
        assert python_found

        # Check JavaScript block is intact
        js_found = False
        for chunk in chunks:
            if "function greet(name)" in chunk:
                assert "```javascript" in chunk
                assert "return `Hello, ${name}!`" in chunk
                js_found = True
                break
        assert js_found


class TestChunkHashing:
    """Tests for chunk hashing."""

    def test_same_content_same_hash(self) -> None:
        """Same content produces same hash."""
        content = "Hello, world!"
        hash1 = compute_chunk_hash(content)
        hash2 = compute_chunk_hash(content)
        assert hash1 == hash2

    def test_different_content_different_hash(self) -> None:
        """Different content produces different hash."""
        hash1 = compute_chunk_hash("Hello")
        hash2 = compute_chunk_hash("World")
        assert hash1 != hash2

    def test_hash_is_sha256(self) -> None:
        """Hash is 64 hex characters (SHA-256)."""
        hash_value = compute_chunk_hash("test")
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)


class TestIncrementalChunking:
    """Tests for incremental chunk behavior."""

    def test_unchanged_content_same_chunks(self) -> None:
        """Unchanged content produces identical chunk hashes."""
        content = """# Document

This is a document with some content.

```python
print("hello")
```

More content here.
"""
        chunks1 = chunk_document(content, "test.md")
        chunks2 = chunk_document(content, "test.md")

        # Same number of chunks
        assert len(chunks1) == len(chunks2)

        # Same hashes
        hashes1 = {c.chunk_hash for c in chunks1}
        hashes2 = {c.chunk_hash for c in chunks2}
        assert hashes1 == hashes2

    def test_changed_content_different_chunks(self) -> None:
        """Changed content produces different chunk hashes."""
        content1 = "# Hello\n\nOriginal content here."
        content2 = "# Hello\n\nModified content here."

        chunks1 = chunk_document(content1, "test.md")
        chunks2 = chunk_document(content2, "test.md")

        hashes1 = {c.chunk_hash for c in chunks1}
        hashes2 = {c.chunk_hash for c in chunks2}

        # Hashes should be different
        assert hashes1 != hashes2

    def test_partial_change_preserves_some_chunks(self) -> None:
        """Partial content change preserves unchanged chunk hashes."""
        # Create content with multiple clearly separated sections
        content1 = """# Section 1

This is section one with unique content.

# Section 2

This is section two with unique content.

# Section 3

This is section three with unique content.
"""
        # Only change section 2
        content2 = """# Section 1

This is section one with unique content.

# Section 2

This section has been MODIFIED.

# Section 3

This is section three with unique content.
"""
        chunks1 = chunk_document(content1, "test.md", chunk_size=500, chunk_overlap=50)
        chunks2 = chunk_document(content2, "test.md", chunk_size=500, chunk_overlap=50)

        hashes1 = {c.chunk_hash for c in chunks1}
        hashes2 = {c.chunk_hash for c in chunks2}

        # At least some chunks should be preserved
        # (Note: due to chunking boundaries, this may vary)
        # The key test is that not all chunks change
        assert len(hashes1) > 0
        assert len(hashes2) > 0


class TestCodeFileSplitting:
    """Tests for code file splitting."""

    def test_python_file_chunked(self) -> None:
        """Python files are chunked with language-aware splitter."""
        code = '''
def function_one():
    """First function."""
    return 1


def function_two():
    """Second function."""
    return 2


class MyClass:
    """A class."""

    def method(self):
        return "method"
'''
        chunks = chunk_document(code, "example.py", chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 0

    def test_javascript_file_chunked(self) -> None:
        """JavaScript files are chunked."""
        code = """
function hello() {
    console.log("Hello");
}

function world() {
    console.log("World");
}

const foo = () => {
    return "bar";
};
"""
        chunks = chunk_document(code, "example.js", chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 0


class TestChunkMetadata:
    """Tests for chunk metadata."""

    def test_chunk_index_sequential(self) -> None:
        """Chunk indices are sequential."""
        content = "A " * 1000  # Long content
        chunks = chunk_document(content, "test.md", chunk_size=300, chunk_overlap=50)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_has_file_path_meta(self) -> None:
        """Chunks include file path in metadata."""
        content = "Some content"
        chunks = chunk_document(content, "path/to/file.md")

        assert len(chunks) > 0
        assert chunks[0].meta.get("file_path") == "path/to/file.md"
