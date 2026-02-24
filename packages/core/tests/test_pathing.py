from contextmine_core.pathing import canonicalize_repo_relative_path


def test_canonicalize_repo_relative_path_equivalence() -> None:
    expected = "a/b.php"
    assert canonicalize_repo_relative_path("./a/b.php") == expected
    assert canonicalize_repo_relative_path("a/b.php") == expected
    assert canonicalize_repo_relative_path(r"a\b.php") == expected
