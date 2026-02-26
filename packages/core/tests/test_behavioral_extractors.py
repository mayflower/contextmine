from contextmine_core.analyzer.extractors.flows import synthesize_user_flows
from contextmine_core.analyzer.extractors.tests import extract_tests_from_files
from contextmine_core.analyzer.extractors.ui import extract_ui_from_files
from contextmine_core.twin.projections import (
    build_test_matrix_projection,
    build_ui_map_projection,
    build_user_flows_projection,
    compute_rebuild_readiness,
)


def test_extract_tests_from_files_detects_cases_and_fixtures() -> None:
    files = [
        (
            "tests/test_checkout.py",
            """
import pytest

@pytest.fixture
def api_client():
    return object()

def test_checkout_creates_order(api_client):
    result = create_order()
    assert result.id
""",
        )
    ]
    extracted = extract_tests_from_files(files)
    assert len(extracted) == 1
    assert extracted[0].cases
    assert extracted[0].fixtures
    assert extracted[0].cases[0].name == "test_checkout_creates_order"


def test_extract_ui_from_files_detects_routes_and_views() -> None:
    files = [
        (
            "apps/web/src/routes.tsx",
            """
import { Route } from "react-router-dom"
export default function CheckoutPage() {
  return <Route path="/checkout" />
}
""",
        )
    ]
    extracted = extract_ui_from_files(files)
    assert len(extracted) == 1
    assert extracted[0].routes
    assert extracted[0].views
    assert extracted[0].routes[0].path == "/checkout"


def test_extract_ui_from_files_skips_ui_test_files() -> None:
    files = [
        (
            "phpmyfaq/admin/assets/src/api/media-browser.test.ts",
            """
describe("media browser", () => {
  it("loads entries", async () => {
    await api.get("/api/media")
  })
})
""",
        )
    ]

    extracted = extract_ui_from_files(files)
    assert extracted == []


def test_extract_ui_from_files_infers_view_and_route_for_non_jsx_module() -> None:
    files = [
        (
            "phpmyfaq/admin/assets/src/content/categories.ts",
            """
export async function loadCategories() {
  return api.get("/api/category")
}

export function openCreateCategory(router) {
  router.push("/admin/content/categories/new")
}
""",
        )
    ]

    extracted = extract_ui_from_files(files)
    assert len(extracted) == 1
    assert len(extracted[0].views) == 1
    assert extracted[0].views[0].name == "Categories"
    assert "/api/category" in extracted[0].views[0].endpoint_hints
    assert extracted[0].routes
    assert extracted[0].routes[0].path == "/admin/content/categories"
    assert extracted[0].routes[0].view_name_hint == "Categories"


def test_flow_synthesis_creates_flow_nodes() -> None:
    ui = extract_ui_from_files(
        [
            (
                "apps/web/src/checkout.tsx",
                """
import { Route } from "react-router-dom"
function CheckoutPage() {
  fetch("/api/orders")
  return <Route path="/checkout" element={<CheckoutPage />} />
}
""",
            )
        ]
    )
    tests = extract_tests_from_files(
        [
            (
                "tests/test_checkout.py",
                """
def test_checkout():
    response = client.post("/api/orders")
    assert response.status_code == 201
""",
            )
        ]
    )
    synthesis = synthesize_user_flows(ui, tests)
    assert synthesis.flows
    assert synthesis.flows[0].steps


def test_behavioral_projections_and_readiness() -> None:
    nodes = [
        {
            "id": "1",
            "kind": "ui_route",
            "name": "/checkout",
            "natural_key": "ui_route:/checkout",
            "meta": {},
        },
        {
            "id": "2",
            "kind": "ui_view",
            "name": "CheckoutPage",
            "natural_key": "ui_view:checkout",
            "meta": {},
        },
        {
            "id": "3",
            "kind": "ui_component",
            "name": "OrderForm",
            "natural_key": "ui_component:order_form",
            "meta": {},
        },
        {
            "id": "4",
            "kind": "test_case",
            "name": "test_checkout",
            "natural_key": "test_case:checkout",
            "meta": {},
        },
        {
            "id": "5",
            "kind": "symbol",
            "name": "create_order",
            "natural_key": "symbol:create_order",
            "meta": {},
        },
        {
            "id": "6",
            "kind": "user_flow",
            "name": "Flow /checkout",
            "natural_key": "user_flow:/checkout",
            "meta": {},
        },
        {
            "id": "7",
            "kind": "flow_step",
            "name": "Invoke /api/orders",
            "natural_key": "flow_step:1",
            "meta": {},
        },
        {
            "id": "8",
            "kind": "api_endpoint",
            "name": "POST /api/orders",
            "natural_key": "api:post:/api/orders",
            "meta": {},
        },
    ]
    edges = [
        {
            "id": "e1",
            "source_node_id": "1",
            "target_node_id": "2",
            "kind": "ui_route_renders_view",
            "meta": {},
        },
        {
            "id": "e2",
            "source_node_id": "2",
            "target_node_id": "3",
            "kind": "ui_view_composes_component",
            "meta": {},
        },
        {
            "id": "e3",
            "source_node_id": "4",
            "target_node_id": "5",
            "kind": "test_case_covers_symbol",
            "meta": {},
        },
        {
            "id": "e4",
            "source_node_id": "6",
            "target_node_id": "7",
            "kind": "user_flow_has_step",
            "meta": {},
        },
        {
            "id": "e5",
            "source_node_id": "7",
            "target_node_id": "8",
            "kind": "flow_step_calls_endpoint",
            "meta": {},
        },
        {
            "id": "e6",
            "source_node_id": "4",
            "target_node_id": "6",
            "kind": "test_case_verifies_flow",
            "meta": {},
        },
    ]

    ui_map = build_ui_map_projection(nodes, edges)
    test_matrix = build_test_matrix_projection(nodes, edges)
    user_flows = build_user_flows_projection(nodes, edges)
    readiness = compute_rebuild_readiness(nodes, edges)

    assert ui_map["summary"]["routes"] == 1
    assert test_matrix["summary"]["test_cases"] == 1
    assert user_flows["summary"]["user_flows"] == 1
    assert 0 <= readiness["score"] <= 100
