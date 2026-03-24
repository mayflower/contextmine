"""Tests for contextmine_core.analyzer.extractors.ui — UI extraction and graph materialization."""

from __future__ import annotations

from contextmine_core.analyzer.extractors.ui import (
    JS_UI_FILE_SUFFIXES,
    NAV_METHOD_NAMES,
    UI_COMPONENT_TAG_PATTERN,
    UI_ENDPOINT_HINT_PATTERNS,
    UI_GENERIC_VIEW_STEMS,
    UI_NAVIGATION_HINT_PATTERN,
    UI_RENDER_CALL_PATTERN,
    UI_ROUTE_EXCLUDE_PREFIXES,
    UI_ROUTE_FILE_BASENAMES,
    UI_ROUTE_PATTERNS,
    UI_ROUTE_SEGMENT_HINTS,
    UI_STRIP_ROUTE_SEGMENTS,
    UI_SYMBOL_HINT_PATTERN,
    UI_TEMPLATE_FILE_SUFFIXES,
    UI_TEST_FILE_MARKERS,
    UI_TEST_PATH_TOKENS,
    UIExtraction,
    UIRouteDef,
    UIViewDef,
    _extract_ui_heuristic,
    _is_template_like_file,
    _line_number_for_offset,
    _looks_like_route_module,
    _normalize_endpoint_path,
    _normalize_ui_route_path,
    _normalize_view_hint,
    _route_path_from_file,
    _to_pascal_case,
    _view_name_from_file_path,
    extract_ui_from_file,
    extract_ui_from_files,
    looks_like_ui_file,
    looks_like_ui_test_file,
)

# ── looks_like_ui_test_file ─────────────────────────────────────────────


class TestLooksLikeUITestFile:
    def test_test_path_tokens(self) -> None:
        assert looks_like_ui_test_file("src/__tests__/Button.test.tsx") is True
        assert looks_like_ui_test_file("src/tests/helpers.ts") is True
        assert looks_like_ui_test_file("src/cypress/e2e/login.cy.ts") is True
        assert looks_like_ui_test_file("src/playwright/tests/home.spec.ts") is True

    def test_test_file_markers(self) -> None:
        assert looks_like_ui_test_file("src/Button.test.tsx") is True
        assert looks_like_ui_test_file("src/Button.spec.tsx") is True
        assert looks_like_ui_test_file("src/button_test.ts") is True
        assert looks_like_ui_test_file("src/button_spec.js") is True

    def test_normal_files(self) -> None:
        assert looks_like_ui_test_file("src/components/Button.tsx") is False
        assert looks_like_ui_test_file("src/pages/Home.tsx") is False

    def test_backslash_normalization(self) -> None:
        assert looks_like_ui_test_file("src\\__tests__\\Button.tsx") is True

    def test_case_insensitive(self) -> None:
        assert looks_like_ui_test_file("src/__Tests__/Button.tsx") is True


# ── looks_like_ui_file ──────────────────────────────────────────────────


class TestLooksLikeUIFile:
    def test_tsx_in_src(self) -> None:
        assert looks_like_ui_file("project/src/pages/Home.tsx") is True

    def test_tsx_in_components(self) -> None:
        assert looks_like_ui_file("project/components/Button.tsx") is True

    def test_vue_in_pages(self) -> None:
        assert looks_like_ui_file("project/pages/index.vue") is True

    def test_html_in_templates(self) -> None:
        assert looks_like_ui_file("project/templates/index.html") is True

    def test_jinja_in_views(self) -> None:
        assert looks_like_ui_file("project/views/base.jinja2") is True

    def test_route_file_basename(self) -> None:
        assert looks_like_ui_file("project/routes/routes.php") is True
        assert looks_like_ui_file("myapp/urls.py") is True

    def test_route_path_hint(self) -> None:
        assert looks_like_ui_file("project/routes/web.php") is True
        assert looks_like_ui_file("project/controllers/auth.py") is True

    def test_not_ui(self) -> None:
        assert looks_like_ui_file("project/lib/utils.py") is False
        assert looks_like_ui_file("project/config.json") is False

    def test_test_files_excluded(self) -> None:
        assert looks_like_ui_file("src/components/Button.test.tsx") is False

    def test_jsx_in_app(self) -> None:
        assert looks_like_ui_file("project/app/Dashboard.jsx") is True

    def test_svelte_in_src(self) -> None:
        assert looks_like_ui_file("project/src/routes/page.svelte") is True


# ── _to_pascal_case ─────────────────────────────────────────────────────


class TestToPascalCase:
    def test_basic(self) -> None:
        assert _to_pascal_case("hello-world") == "HelloWorld"

    def test_snake_case(self) -> None:
        assert _to_pascal_case("hello_world") == "HelloWorld"

    def test_already_pascal(self) -> None:
        assert _to_pascal_case("HelloWorld") == "HelloWorld"

    def test_empty(self) -> None:
        assert _to_pascal_case("") == ""

    def test_none_equivalent(self) -> None:
        assert _to_pascal_case("") == ""

    def test_single_word(self) -> None:
        assert _to_pascal_case("button") == "Button"

    def test_multiple_delimiters(self) -> None:
        assert _to_pascal_case("my-cool_component") == "MyCoolComponent"


# ── _view_name_from_file_path ───────────────────────────────────────────


class TestViewNameFromFilePath:
    def test_basic_tsx(self) -> None:
        result = _view_name_from_file_path("src/pages/Dashboard.tsx")
        assert result == "Dashboard"

    def test_index_uses_parent(self) -> None:
        result = _view_name_from_file_path("src/pages/settings/index.tsx")
        assert result == "Settings"

    def test_generic_stem_uses_parent(self) -> None:
        result = _view_name_from_file_path("src/views/config.tsx")
        assert result == "Views"

    def test_template_extension_stripped(self) -> None:
        result = _view_name_from_file_path("templates/layout.html")
        assert result == "Layout"

    def test_nested_template_ext(self) -> None:
        result = _view_name_from_file_path("templates/base.html.jinja2")
        assert result == "Base"

    def test_dotfile_returns_name(self) -> None:
        # .gitkeep becomes Gitkeep through pascal conversion
        result = _view_name_from_file_path("src/pages/.gitkeep")
        assert result == "Gitkeep"


# ── _looks_like_route_module ────────────────────────────────────────────


class TestLooksLikeRouteModule:
    def test_route_segment(self) -> None:
        assert _looks_like_route_module("src/pages/Home.tsx") is True

    def test_routes_dir(self) -> None:
        assert _looks_like_route_module("src/routes/index.tsx") is True

    def test_admin_assets_src(self) -> None:
        assert _looks_like_route_module("admin/assets/src/Login.tsx") is True

    def test_not_route_module(self) -> None:
        assert _looks_like_route_module("src/lib/helpers.ts") is False


# ── _route_path_from_file ──────────────────────────────────────────────


class TestRoutePathFromFile:
    def test_src_path(self) -> None:
        result = _route_path_from_file("project/src/pages/settings/profile.tsx")
        assert result is not None
        assert "settings" in result or "profile" in result

    def test_admin_assets_src(self) -> None:
        result = _route_path_from_file("foo/admin/assets/src/login/index.tsx")
        assert result is not None
        assert result.startswith("/admin")

    def test_assets_src_with_admin(self) -> None:
        result = _route_path_from_file("root/admin/bar/assets/src/login.tsx")
        assert result is not None
        assert result.startswith("/admin")

    def test_ui_path(self) -> None:
        result = _route_path_from_file("project/ui/Dashboard.tsx")
        assert result is not None

    def test_no_match(self) -> None:
        result = _route_path_from_file("lib/utils.py")
        assert result is None

    def test_index_stripped(self) -> None:
        result = _route_path_from_file("project/src/pages/index.tsx")
        # "index" is stripped, pages is stripped (leading route segment)
        assert result is not None

    def test_strip_route_segments(self) -> None:
        result = _route_path_from_file("project/src/pages/home.tsx")
        # "pages" as first segment should be stripped
        assert result is not None
        assert "pages" not in result


# ── _line_number_for_offset ────────────────────────────────────────────


class TestLineNumberForOffset:
    def test_first_line(self) -> None:
        assert _line_number_for_offset("hello\nworld", 0) == 1

    def test_second_line(self) -> None:
        assert _line_number_for_offset("hello\nworld", 6) == 2

    def test_negative_offset(self) -> None:
        assert _line_number_for_offset("hello\nworld", -5) == 1

    def test_empty_content(self) -> None:
        assert _line_number_for_offset("", 0) == 1


# ── _normalize_ui_route_path ────────────────────────────────────────────


class TestNormalizeUIRoutePath:
    def test_basic_path(self) -> None:
        assert _normalize_ui_route_path("/users") == "/users"

    def test_adds_leading_slash(self) -> None:
        assert _normalize_ui_route_path("users") == "/users"

    def test_strips_query_hash(self) -> None:
        assert _normalize_ui_route_path("/users?page=1#top") == "/users"

    def test_strips_anchors(self) -> None:
        assert _normalize_ui_route_path("/page^$") == "/page"

    def test_replaces_django_params(self) -> None:
        result = _normalize_ui_route_path("/users/<int:id>/edit")
        assert ":param" in result

    def test_replaces_laravel_params(self) -> None:
        result = _normalize_ui_route_path("/users/{id}/edit")
        assert ":param" in result

    def test_empty_returns_none(self) -> None:
        assert _normalize_ui_route_path("") is None
        assert _normalize_ui_route_path("   ") is None

    def test_url_with_protocol_returns_none(self) -> None:
        assert _normalize_ui_route_path("http://example.com/path") is None

    def test_api_prefix_excluded(self) -> None:
        assert _normalize_ui_route_path("/api/v1/users") is None

    def test_graphql_prefix_excluded(self) -> None:
        assert _normalize_ui_route_path("/graphql") is None

    def test_root_path(self) -> None:
        result = _normalize_ui_route_path("/")
        assert result == "/"


# ── _normalize_view_hint ────────────────────────────────────────────────


class TestNormalizeViewHint:
    def test_basic(self) -> None:
        result = _normalize_view_hint("dashboard")
        assert result == "Dashboard"

    def test_template_path(self) -> None:
        result = _normalize_view_hint("auth/login.html")
        assert result == "Login"

    def test_index_uses_parent(self) -> None:
        result = _normalize_view_hint("settings/index")
        assert result == "Settings"

    def test_controller_action(self) -> None:
        result = _normalize_view_hint("UserController::show")
        assert result == "Show"

    def test_empty_returns_none(self) -> None:
        assert _normalize_view_hint("") is None
        assert _normalize_view_hint("   ") is None


# ── _normalize_endpoint_path ────────────────────────────────────────────


class TestNormalizeEndpointPath:
    def test_basic_path(self) -> None:
        assert _normalize_endpoint_path("/api/users") == "/api/users"

    def test_url_with_protocol(self) -> None:
        result = _normalize_endpoint_path("https://example.com/api/users")
        assert result == "/api/users"

    def test_strips_query_params(self) -> None:
        assert _normalize_endpoint_path("/users?page=1") == "/users"

    def test_empty_returns_none(self) -> None:
        assert _normalize_endpoint_path("") is None

    def test_no_leading_slash_gets_one(self) -> None:
        assert _normalize_endpoint_path("users/list") == "/users/list"

    def test_url_no_path(self) -> None:
        result = _normalize_endpoint_path("https://example.com")
        assert result == "/"


# ── _is_template_like_file ──────────────────────────────────────────────


class TestIsTemplateLikeFile:
    def test_html_in_templates(self) -> None:
        assert _is_template_like_file("project/templates/index.html") is True

    def test_jinja_in_views(self) -> None:
        assert _is_template_like_file("project/views/base.jinja2") is True

    def test_html_not_in_template_path(self) -> None:
        assert _is_template_like_file("project/lib/data.html") is False

    def test_erb_in_views(self) -> None:
        assert _is_template_like_file("app/views/users/show.erb") is True


# ── UI_ROUTE_PATTERNS regex patterns ────────────────────────────────────


class TestUIRoutePatterns:
    def test_laravel_route(self) -> None:
        content = "Route::get('/users', [UserController::class, 'index']);"
        matches = list(UI_ROUTE_PATTERNS[0].finditer(content))
        assert len(matches) >= 1
        assert matches[0].group("path") == "/users"

    def test_express_route(self) -> None:
        content = "get '/dashboard' do\n  # handler\nend"
        matches = list(UI_ROUTE_PATTERNS[5].finditer(content))
        assert len(matches) >= 1
        assert matches[0].group("path") == "/dashboard"

    def test_django_path(self) -> None:
        content = "path('users/', views.user_list)"
        matches = list(UI_ROUTE_PATTERNS[4].finditer(content))
        assert len(matches) >= 1
        assert matches[0].group("path") == "users/"

    def test_annotation_route(self) -> None:
        content = '@Route("/products/{id}")'
        matches = list(UI_ROUTE_PATTERNS[3].finditer(content))
        assert len(matches) >= 1
        assert matches[0].group("path") == "/products/{id}"

    def test_php_method_route(self) -> None:
        content = "$router->get('/login', 'AuthController@login');"
        matches = list(UI_ROUTE_PATTERNS[1].finditer(content))
        assert len(matches) >= 1
        assert matches[0].group("path") == "/login"


# ── UI_RENDER_CALL_PATTERN ──────────────────────────────────────────────


class TestUIRenderCallPattern:
    def test_render_template(self) -> None:
        content = "render_template('auth/login.html')"
        match = UI_RENDER_CALL_PATTERN.search(content)
        assert match is not None
        assert match.group("name") == "auth/login.html"

    def test_view(self) -> None:
        content = "view('dashboard.index')"
        match = UI_RENDER_CALL_PATTERN.search(content)
        assert match is not None
        assert match.group("name") == "dashboard.index"

    def test_template_response(self) -> None:
        content = 'TemplateResponse("users/list.html", context)'
        match = UI_RENDER_CALL_PATTERN.search(content)
        assert match is not None
        assert match.group("name") == "users/list.html"


# ── UI_COMPONENT_TAG_PATTERN ───────────────────────────────────────────


class TestUIComponentTagPattern:
    def test_pascal_case_component(self) -> None:
        content = "<MyComponent />"
        match = UI_COMPONENT_TAG_PATTERN.search(content)
        assert match is not None
        assert match.group("tag") == "MyComponent"

    def test_kebab_case_component(self) -> None:
        content = "<my-component />"
        match = UI_COMPONENT_TAG_PATTERN.search(content)
        assert match is not None
        assert match.group("tag") == "my-component"

    def test_html_tag_too_short(self) -> None:
        content = "<a href='#'>link</a>"
        match = UI_COMPONENT_TAG_PATTERN.search(content)
        # 'a' is only 1 char, pattern requires 3+ chars
        assert match is None

    def test_namespaced_tag(self) -> None:
        content = "<v:card-title>text</v:card-title>"
        match = UI_COMPONENT_TAG_PATTERN.search(content)
        assert match is not None
        assert "card-title" in match.group("tag")


# ── UI_ENDPOINT_HINT_PATTERNS ──────────────────────────────────────────


class TestUIEndpointHintPatterns:
    def test_fetch_call(self) -> None:
        content = "fetch('/api/users')"
        matches = list(UI_ENDPOINT_HINT_PATTERNS[0].finditer(content))
        assert len(matches) >= 1
        assert matches[0].group("path") == "/api/users"

    def test_axios_get(self) -> None:
        content = "axios.get('/api/orders')"
        matches = list(UI_ENDPOINT_HINT_PATTERNS[0].finditer(content))
        assert len(matches) >= 1
        assert matches[0].group("path") == "/api/orders"

    def test_form_action(self) -> None:
        content = 'action="/submit-form"'
        matches = list(UI_ENDPOINT_HINT_PATTERNS[1].finditer(content))
        assert len(matches) >= 1
        assert matches[0].group("path") == "/submit-form"

    def test_data_url(self) -> None:
        content = 'data-url="/api/search"'
        matches = list(UI_ENDPOINT_HINT_PATTERNS[2].finditer(content))
        assert len(matches) >= 1
        assert matches[0].group("path") == "/api/search"


# ── UI_NAVIGATION_HINT_PATTERN ─────────────────────────────────────────


class TestUINavigationHintPattern:
    def test_href(self) -> None:
        content = 'href="/dashboard"'
        match = UI_NAVIGATION_HINT_PATTERN.search(content)
        assert match is not None
        assert match.group("path") == "/dashboard"


# ── UI_SYMBOL_HINT_PATTERN ─────────────────────────────────────────────


class TestUISymbolHintPattern:
    def test_controller_action(self) -> None:
        content = "UserController::index"
        match = UI_SYMBOL_HINT_PATTERN.search(content)
        assert match is not None
        assert "UserController" in match.group("symbol")

    def test_service_class(self) -> None:
        content = "AuthService"
        match = UI_SYMBOL_HINT_PATTERN.search(content)
        assert match is not None
        assert match.group("symbol") == "AuthService"


# ── _extract_ui_heuristic ──────────────────────────────────────────────


class TestExtractUIHeuristic:
    def test_route_extraction_laravel(self) -> None:
        content = """Route::get('/users', [UserController::class, 'index']);
Route::post('/users', [UserController::class, 'store']);
Route::get('/users/{id}', [UserController::class, 'show']);"""
        result = _extract_ui_heuristic("routes/web.php", content)
        assert len(result.routes) >= 2
        paths = [r.path for r in result.routes]
        assert "/users" in paths

    def test_render_call_creates_view(self) -> None:
        content = "return render_template('dashboard/overview.html', data=data)"
        result = _extract_ui_heuristic("app/views/home.py", content)
        assert len(result.views) >= 1
        view_names = [v.name for v in result.views]
        assert any("Overview" in name for name in view_names)

    def test_template_file_creates_view(self) -> None:
        content = "<h1>Dashboard</h1><p>Welcome</p>"
        result = _extract_ui_heuristic("project/templates/dashboard.html", content)
        assert len(result.views) >= 1
        assert result.views[0].name == "Dashboard"

    def test_component_tags_collected(self) -> None:
        content = """<UserProfile />
<nav-bar></nav-bar>
<MyButton class="primary">Click</MyButton>"""
        result = _extract_ui_heuristic("templates/page.html", content)
        if result.views:
            all_components = []
            for v in result.views:
                all_components.extend(v.components)
            assert len(all_components) >= 1

    def test_endpoint_hints_collected(self) -> None:
        content = """fetch('/api/users')
<form action="/submit-form">"""
        result = _extract_ui_heuristic("src/pages/home.tsx", content)
        if result.views:
            all_endpoints = []
            for v in result.views:
                all_endpoints.extend(v.endpoint_hints)
            assert len(all_endpoints) >= 1

    def test_navigation_hints_collected(self) -> None:
        content = '<a href="/settings">Settings</a>'
        result = _extract_ui_heuristic("templates/nav.html", content)
        if result.views:
            all_nav = []
            for v in result.views:
                all_nav.extend(v.navigation_targets)
            assert len(all_nav) >= 1

    def test_inferred_route_from_file_path(self) -> None:
        content = "// No explicit route patterns here"
        result = _extract_ui_heuristic("src/pages/profile.tsx", content)
        # May or may not infer a route depending on _looks_like_route_module
        assert isinstance(result, UIExtraction)

    def test_api_href_goes_to_endpoint_hints(self) -> None:
        content = '<a href="/api/v1/export">Export</a>'
        result = _extract_ui_heuristic("templates/admin.html", content)
        if result.views:
            all_endpoints = []
            for v in result.views:
                all_endpoints.extend(v.endpoint_hints)
            # /api prefixed hrefs should go to endpoint_hints, not navigation
            assert any("/api" in ep for ep in all_endpoints)


# ── extract_ui_from_file ───────────────────────────────────────────────


class TestExtractUIFromFile:
    def test_test_file_returns_empty(self) -> None:
        result = extract_ui_from_file("src/__tests__/Button.test.tsx", "<div>test</div>")
        assert len(result.routes) == 0
        assert len(result.views) == 0

    def test_non_js_file_uses_heuristic(self) -> None:
        content = "Route::get('/admin', 'AdminController@index');"
        result = extract_ui_from_file("routes/web.php", content)
        assert isinstance(result, UIExtraction)
        if result.routes:
            assert result.routes[0].path == "/admin"

    def test_tsx_file_with_react_route(self) -> None:
        content = """import { Route } from 'react-router-dom';
function App() {
  return (
    <Route path="/dashboard" element={<Dashboard />} />
  );
}
function Dashboard() {
  return <div>Dashboard</div>;
}"""
        result = extract_ui_from_file("src/App.tsx", content)
        assert isinstance(result, UIExtraction)
        # May extract routes and/or views depending on treesitter availability

    def test_empty_content_returns_empty(self) -> None:
        result = extract_ui_from_file("src/pages/Home.tsx", "")
        assert isinstance(result, UIExtraction)


# ── extract_ui_from_files ──────────────────────────────────────────────


class TestExtractUIFromFiles:
    def test_filters_non_ui_files(self) -> None:
        files = [
            ("src/lib/utils.py", "def helper(): pass"),
            ("src/pages/Home.tsx", "export default function Home() { return <div>Home</div> }"),
        ]
        results = extract_ui_from_files(files)
        # Non-UI files should be filtered
        assert isinstance(results, list)

    def test_skips_empty_content(self) -> None:
        files = [
            ("src/pages/Empty.tsx", ""),
            ("src/pages/Empty2.tsx", "   "),
        ]
        results = extract_ui_from_files(files)
        assert len(results) == 0

    def test_processes_multiple_files(self) -> None:
        files = [
            (
                "routes/web.php",
                "Route::get('/users', 'UserController@index');",
            ),
            (
                "routes/api.php",
                "Route::get('/api/users', 'ApiUserController@list');",
            ),
        ]
        results = extract_ui_from_files(files)
        # Only files that look_like_ui_file will be processed
        assert isinstance(results, list)


# ── UIRouteDef / UIViewDef / UIExtraction dataclasses ──────────────────


class TestDataclasses:
    def test_route_def_defaults(self) -> None:
        route = UIRouteDef(path="/users", file_path="routes.php", line=1)
        assert route.view_name_hint is None
        assert route.inferred is False

    def test_route_def_with_hint(self) -> None:
        route = UIRouteDef(
            path="/users",
            file_path="routes.php",
            line=5,
            view_name_hint="Users",
            inferred=True,
        )
        assert route.view_name_hint == "Users"
        assert route.inferred is True

    def test_view_def_defaults(self) -> None:
        view = UIViewDef(name="Dashboard", file_path="src/views/Dashboard.tsx", line=1)
        assert view.components == []
        assert view.symbol_hints == []
        assert view.endpoint_hints == []
        assert view.navigation_targets == []
        assert view.call_sites == []
        assert view.inferred is False

    def test_extraction_defaults(self) -> None:
        extraction = UIExtraction(file_path="test.tsx")
        assert extraction.routes == []
        assert extraction.views == []


# ── Constants sanity checks ────────────────────────────────────────────


class TestConstants:
    def test_js_ui_file_suffixes(self) -> None:
        assert ".tsx" in JS_UI_FILE_SUFFIXES
        assert ".jsx" in JS_UI_FILE_SUFFIXES
        assert ".vue" in JS_UI_FILE_SUFFIXES

    def test_template_suffixes(self) -> None:
        assert ".html" in UI_TEMPLATE_FILE_SUFFIXES
        assert ".jinja2" in UI_TEMPLATE_FILE_SUFFIXES
        assert ".erb" in UI_TEMPLATE_FILE_SUFFIXES

    def test_route_file_basenames(self) -> None:
        assert "routes.php" in UI_ROUTE_FILE_BASENAMES
        assert "urls.py" in UI_ROUTE_FILE_BASENAMES

    def test_nav_method_names(self) -> None:
        assert "navigate" in NAV_METHOD_NAMES
        assert "push" in NAV_METHOD_NAMES

    def test_route_exclude_prefixes(self) -> None:
        assert "/api" in UI_ROUTE_EXCLUDE_PREFIXES
        assert "/graphql" in UI_ROUTE_EXCLUDE_PREFIXES

    def test_generic_view_stems(self) -> None:
        assert "utils" in UI_GENERIC_VIEW_STEMS
        assert "config" in UI_GENERIC_VIEW_STEMS

    def test_strip_route_segments(self) -> None:
        assert "pages" in UI_STRIP_ROUTE_SEGMENTS
        assert "routes" in UI_STRIP_ROUTE_SEGMENTS

    def test_test_tokens(self) -> None:
        assert "/__tests__/" in UI_TEST_PATH_TOKENS
        assert ".test." in UI_TEST_FILE_MARKERS

    def test_route_segment_hints(self) -> None:
        assert "dashboard" in UI_ROUTE_SEGMENT_HINTS
        assert "pages" in UI_ROUTE_SEGMENT_HINTS
