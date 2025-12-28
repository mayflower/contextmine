"""Tests for GitHub sync incremental logic."""

import tempfile
from pathlib import Path

from contextmine_worker.github_sync import (
    compute_content_hash,
    get_changed_files,
    is_eligible_file,
)
from git import Repo


class TestFileEligibility:
    """Tests for file eligibility checking."""

    def test_markdown_files_eligible(self) -> None:
        """Test that markdown files are eligible."""
        assert is_eligible_file(Path("README.md"), Path("/tmp"))
        assert is_eligible_file(Path("docs/guide.md"), Path("/tmp"))
        assert is_eligible_file(Path("src/index.mdx"), Path("/tmp"))

    def test_code_files_eligible(self) -> None:
        """Test that code files are eligible."""
        assert is_eligible_file(Path("main.py"), Path("/tmp"))
        assert is_eligible_file(Path("src/app.ts"), Path("/tmp"))
        assert is_eligible_file(Path("lib/utils.js"), Path("/tmp"))
        assert is_eligible_file(Path("cmd/server.go"), Path("/tmp"))

    def test_config_files_eligible(self) -> None:
        """Test that config files are eligible."""
        assert is_eligible_file(Path("config.yaml"), Path("/tmp"))
        assert is_eligible_file(Path("settings.json"), Path("/tmp"))
        assert is_eligible_file(Path("pyproject.toml"), Path("/tmp"))

    def test_special_files_eligible(self) -> None:
        """Test that special files without extensions are eligible."""
        assert is_eligible_file(Path("Dockerfile"), Path("/tmp"))
        assert is_eligible_file(Path("Makefile"), Path("/tmp"))

    def test_binary_files_not_eligible(self) -> None:
        """Test that binary files are not eligible."""
        assert not is_eligible_file(Path("image.png"), Path("/tmp"))
        assert not is_eligible_file(Path("app.exe"), Path("/tmp"))
        assert not is_eligible_file(Path("data.bin"), Path("/tmp"))

    def test_hidden_files_not_eligible(self) -> None:
        """Test that hidden files are not eligible."""
        assert not is_eligible_file(Path(".gitignore"), Path("/tmp"))
        assert not is_eligible_file(Path(".env"), Path("/tmp"))

    def test_node_modules_not_eligible(self) -> None:
        """Test that node_modules files are not eligible."""
        assert not is_eligible_file(Path("node_modules/package/index.js"), Path("/tmp"))

    def test_venv_not_eligible(self) -> None:
        """Test that venv files are not eligible."""
        assert not is_eligible_file(Path("venv/lib/python/site.py"), Path("/tmp"))
        assert not is_eligible_file(Path(".venv/lib/python/site.py"), Path("/tmp"))

    def test_github_folder_eligible(self) -> None:
        """Test that .github folder files are eligible."""
        assert is_eligible_file(Path(".github/workflows/ci.yaml"), Path("/tmp"))


class TestContentHash:
    """Tests for content hashing."""

    def test_hash_consistency(self) -> None:
        """Test that same content produces same hash."""
        content = "Hello, world!"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2

    def test_hash_different_for_different_content(self) -> None:
        """Test that different content produces different hash."""
        hash1 = compute_content_hash("Hello")
        hash2 = compute_content_hash("World")
        assert hash1 != hash2

    def test_hash_length(self) -> None:
        """Test that hash is SHA-256 (64 hex chars)."""
        hash_value = compute_content_hash("test")
        assert len(hash_value) == 64


class TestIncrementalSync:
    """Tests for incremental sync using local git repos."""

    def test_full_index_returns_all_files(self) -> None:
        """Test that full index (no cursor) returns all files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create repo with some files
            repo = Repo.init(tmpdir)

            # Create files
            (Path(tmpdir) / "file1.py").write_text("print('hello')")
            (Path(tmpdir) / "file2.md").write_text("# README")
            (Path(tmpdir) / "subdir").mkdir()
            (Path(tmpdir) / "subdir" / "file3.ts").write_text("const x = 1")

            # Commit
            repo.index.add(["file1.py", "file2.md", "subdir/file3.ts"])
            repo.index.commit("Initial commit")

            sha = repo.head.commit.hexsha

            # Get changed files (full index)
            changed, deleted = get_changed_files(repo, None, sha)

            assert len(changed) == 3
            assert "file1.py" in changed
            assert "file2.md" in changed
            assert "subdir/file3.ts" in changed
            assert len(deleted) == 0

    def test_incremental_detects_new_files(self) -> None:
        """Test that incremental sync detects new files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repo.init(tmpdir)

            # Create initial file
            (Path(tmpdir) / "file1.py").write_text("print('hello')")
            repo.index.add(["file1.py"])
            commit_a = repo.index.commit("Commit A")

            # Add new file
            (Path(tmpdir) / "file2.py").write_text("print('world')")
            repo.index.add(["file2.py"])
            commit_b = repo.index.commit("Commit B")

            # Get changed files
            changed, deleted = get_changed_files(repo, commit_a.hexsha, commit_b.hexsha)

            assert len(changed) == 1
            assert "file2.py" in changed
            assert len(deleted) == 0

    def test_incremental_detects_modified_files(self) -> None:
        """Test that incremental sync detects modified files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repo.init(tmpdir)

            # Create initial file
            (Path(tmpdir) / "file1.py").write_text("print('hello')")
            repo.index.add(["file1.py"])
            commit_a = repo.index.commit("Commit A")

            # Modify file
            (Path(tmpdir) / "file1.py").write_text("print('world')")
            repo.index.add(["file1.py"])
            commit_b = repo.index.commit("Commit B")

            # Get changed files
            changed, deleted = get_changed_files(repo, commit_a.hexsha, commit_b.hexsha)

            assert len(changed) == 1
            assert "file1.py" in changed
            assert len(deleted) == 0

    def test_incremental_detects_deleted_files(self) -> None:
        """Test that incremental sync detects deleted files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repo.init(tmpdir)

            # Create initial files
            (Path(tmpdir) / "file1.py").write_text("print('hello')")
            (Path(tmpdir) / "file2.py").write_text("print('world')")
            repo.index.add(["file1.py", "file2.py"])
            commit_a = repo.index.commit("Commit A")

            # Delete file
            (Path(tmpdir) / "file2.py").unlink()
            repo.index.remove(["file2.py"])
            commit_b = repo.index.commit("Commit B")

            # Get changed files
            changed, deleted = get_changed_files(repo, commit_a.hexsha, commit_b.hexsha)

            assert len(changed) == 0
            assert len(deleted) == 1
            assert "file2.py" in deleted

    def test_incremental_detects_renamed_files(self) -> None:
        """Test that incremental sync handles renamed files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repo.init(tmpdir)

            # Create initial file
            (Path(tmpdir) / "old_name.py").write_text("print('hello')")
            repo.index.add(["old_name.py"])
            commit_a = repo.index.commit("Commit A")

            # Rename file
            (Path(tmpdir) / "old_name.py").rename(Path(tmpdir) / "new_name.py")
            repo.index.remove(["old_name.py"])
            repo.index.add(["new_name.py"])
            commit_b = repo.index.commit("Commit B")

            # Get changed files
            changed, deleted = get_changed_files(repo, commit_a.hexsha, commit_b.hexsha)

            # Renamed file should appear in changed and old name in deleted
            assert "new_name.py" in changed
            assert "old_name.py" in deleted

    def test_multiple_changes_in_one_commit(self) -> None:
        """Test incremental sync with multiple changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Repo.init(tmpdir)

            # Create initial files
            (Path(tmpdir) / "keep.py").write_text("keep")
            (Path(tmpdir) / "modify.py").write_text("old content")
            (Path(tmpdir) / "delete.py").write_text("delete me")
            repo.index.add(["keep.py", "modify.py", "delete.py"])
            commit_a = repo.index.commit("Commit A")

            # Make multiple changes
            (Path(tmpdir) / "modify.py").write_text("new content")
            (Path(tmpdir) / "delete.py").unlink()
            (Path(tmpdir) / "add.py").write_text("new file")
            repo.index.add(["modify.py", "add.py"])
            repo.index.remove(["delete.py"])
            commit_b = repo.index.commit("Commit B")

            # Get changed files
            changed, deleted = get_changed_files(repo, commit_a.hexsha, commit_b.hexsha)

            assert len(changed) == 2
            assert "modify.py" in changed
            assert "add.py" in changed
            assert len(deleted) == 1
            assert "delete.py" in deleted
            # keep.py should not be in changed
            assert "keep.py" not in changed
