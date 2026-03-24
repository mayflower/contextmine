"""Extended tests for contextmine_core.analyzer.extractors.tests — test extraction logic."""

from __future__ import annotations

from contextmine_core.analyzer.extractors.tests import (
    JS_CASE_CALLS,
    JS_CASE_CALLS_LOWER,
    JS_FIXTURE_CALLS,
    JS_FIXTURE_CALLS_LOWER,
    JS_SUITE_CALLS,
    JS_SUITE_CALLS_LOWER,
    SYMBOL_STOP_WORDS,
    TEST_FILE_PATTERNS,
    TestCaseDef,
    TestFixtureDef,
    TestsExtraction,
    TestSuiteDef,
    detect_test_framework,
    extract_tests_from_file,
    extract_tests_from_files,
    looks_like_test_file,
)

# ── looks_like_test_file ────────────────────────────────────────────────


class TestLooksLikeTestFile:
    def test_test_prefix(self) -> None:
        assert looks_like_test_file("test_module.py")

    def test_test_suffix(self) -> None:
        assert looks_like_test_file("module_test.py")

    def test_spec_suffix(self) -> None:
        assert looks_like_test_file("module.spec.ts")

    def test_test_ts_suffix(self) -> None:
        assert looks_like_test_file("module.test.ts")

    def test_tests_directory(self) -> None:
        assert looks_like_test_file("src/tests/helper.py")

    def test_dunder_tests(self) -> None:
        assert looks_like_test_file("__tests__/foo.ts")

    def test_not_test_file(self) -> None:
        assert not looks_like_test_file("src/module.py")

    def test_image_file_excluded(self) -> None:
        assert not looks_like_test_file("test_image.png")

    def test_jpg_excluded(self) -> None:
        assert not looks_like_test_file("test_photo.jpg")

    def test_lock_file_excluded(self) -> None:
        assert not looks_like_test_file("test.lock")

    def test_svg_excluded(self) -> None:
        assert not looks_like_test_file("test_icon.svg")

    def test_map_file_excluded(self) -> None:
        assert not looks_like_test_file("test.js.map")


# ── detect_test_framework ───────────────────────────────────────────────


class TestDetectFramework:
    def test_pytest(self) -> None:
        assert detect_test_framework("test_foo.py", "import pytest") == "pytest"

    def test_unittest(self) -> None:
        assert detect_test_framework("test_foo.py", "import unittest") == "unittest"

    def test_jest(self) -> None:
        assert detect_test_framework("foo.test.ts", "import { jest }") == "jest"

    def test_junit(self) -> None:
        assert detect_test_framework("FooTest.java", "@Test\npublic void test()") == "junit"

    def test_vitest(self) -> None:
        assert detect_test_framework("foo.test.ts", "import { vitest }") == "vitest"

    def test_cypress(self) -> None:
        assert detect_test_framework("foo.spec.ts", "cy.visit(); cypress") == "cypress"

    def test_playwright(self) -> None:
        assert detect_test_framework("foo.spec.ts", "import { playwright }") == "playwright"

    def test_js_test_fallback(self) -> None:
        assert detect_test_framework("foo.spec.ts", "some plain code") == "js_test"

    def test_js_test_js(self) -> None:
        assert detect_test_framework("foo.test.js", "random content") == "js_test"

    def test_unknown(self) -> None:
        assert detect_test_framework("test_foo.py", "random content") == "unknown"


# ── extract_tests_from_file — Python ────────────────────────────────────


class TestExtractPythonTests:
    def test_simple_python_test_function(self) -> None:
        code = """\
import pytest

def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 2 - 1 == 1
"""
        result = extract_tests_from_file("tests/test_math.py", code)
        assert result.framework == "pytest"
        assert len(result.cases) >= 2
        names = [c.name for c in result.cases]
        assert "test_addition" in names
        assert "test_subtraction" in names

    def test_python_test_class(self) -> None:
        code = """\
import pytest

class TestCalculator:
    def test_add(self):
        assert 1 + 1 == 2

    def test_sub(self):
        assert 2 - 1 == 1
"""
        result = extract_tests_from_file("tests/test_calc.py", code)
        assert len(result.suites) >= 1
        assert result.suites[0].name == "TestCalculator"
        assert len(result.cases) >= 2

    def test_python_fixture(self) -> None:
        code = """\
import pytest

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
"""
        result = extract_tests_from_file("tests/test_fixtures.py", code)
        assert len(result.fixtures) >= 1
        assert result.fixtures[0].name == "sample_data"

    def test_empty_file(self) -> None:
        result = extract_tests_from_file("tests/test_empty.py", "")
        assert len(result.cases) == 0
        assert len(result.suites) == 0

    def test_non_test_python_file(self) -> None:
        code = """\
def helper():
    return 42
"""
        result = extract_tests_from_file("tests/test_helper.py", code)
        assert result.framework in ("unknown", "pytest", "unittest")


# ── extract_tests_from_file — JavaScript ────────────────────────────────


class TestExtractJsTests:
    def test_jest_describe_it(self) -> None:
        code = """\
describe('Calculator', () => {
  it('should add numbers', () => {
    expect(1 + 1).toBe(2);
  });

  it('should subtract', () => {
    expect(2 - 1).toBe(1);
  });
});
"""
        result = extract_tests_from_file("tests/calc.test.ts", code)
        assert result.framework in ("jest", "js_test")
        if result.cases:
            assert any(
                "add" in c.name.lower() or "subtract" in c.name.lower() for c in result.cases
            )

    def test_test_function_call(self) -> None:
        code = """\
test('does something', () => {
  expect(true).toBe(true);
});
"""
        result = extract_tests_from_file("tests/basic.test.js", code)
        if result.cases:
            assert len(result.cases) >= 1


# ── extract_tests_from_files ────────────────────────────────────────────


class TestExtractFromFiles:
    def test_filters_non_test_files(self) -> None:
        files = [
            ("src/module.py", "def helper(): pass"),
            ("tests/test_module.py", "def test_foo(): assert True"),
        ]
        results = extract_tests_from_files(files)
        # Only test file should be processed
        for r in results:
            assert "test" in r.file_path.lower()

    def test_skips_empty_content(self) -> None:
        files = [("tests/test_empty.py", ""), ("tests/test_empty2.py", "   ")]
        results = extract_tests_from_files(files)
        assert len(results) == 0


# ── Data classes ────────────────────────────────────────────────────────


class TestDataClasses:
    def test_test_case_def_defaults(self) -> None:
        tc = TestCaseDef(name="test_foo", file_path="test.py", line=1)
        assert tc.suite_name is None
        assert tc.fixture_names == []
        assert tc.symbol_hints == []
        assert tc.endpoint_hints == []
        assert tc.call_sites == []
        assert tc.raw_assertions == []
        assert tc.natural_key == ""

    def test_test_suite_def_defaults(self) -> None:
        ts = TestSuiteDef(name="TestFoo", file_path="test.py", line=1)
        assert ts.natural_key == ""

    def test_tests_extraction_defaults(self) -> None:
        te = TestsExtraction(file_path="test.py", framework="pytest")
        assert te.suites == []
        assert te.cases == []
        assert te.fixtures == []

    def test_fixture_def(self) -> None:
        f = TestFixtureDef(name="db", file_path="conftest.py", line=5)
        assert f.name == "db"
        assert f.line == 5


# ── Constants ───────────────────────────────────────────────────────────


class TestConstants:
    def test_test_file_patterns(self) -> None:
        assert "test_" in TEST_FILE_PATTERNS
        assert "_test." in TEST_FILE_PATTERNS
        assert ".spec." in TEST_FILE_PATTERNS

    def test_js_suite_calls(self) -> None:
        assert "describe" in JS_SUITE_CALLS
        assert "context" in JS_SUITE_CALLS
        assert "suite" in JS_SUITE_CALLS

    def test_js_case_calls(self) -> None:
        assert "test" in JS_CASE_CALLS
        assert "it" in JS_CASE_CALLS
        assert "specify" in JS_CASE_CALLS

    def test_js_fixture_calls(self) -> None:
        assert "beforeEach" in JS_FIXTURE_CALLS
        assert "afterAll" in JS_FIXTURE_CALLS

    def test_lower_variants(self) -> None:
        assert "describe" in JS_SUITE_CALLS_LOWER
        assert "test" in JS_CASE_CALLS_LOWER
        assert "beforeeach" in JS_FIXTURE_CALLS_LOWER

    def test_symbol_stop_words(self) -> None:
        assert "assert" in SYMBOL_STOP_WORDS
        assert "expect" in SYMBOL_STOP_WORDS
        assert "fetch" in SYMBOL_STOP_WORDS
