"""Coverage tests for extractors: schema, openapi, protobuf, jobs.

Targets uncovered pure-function code paths to push overall coverage toward 80%.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schema DDL extraction (schema.py lines 189-340)
# ---------------------------------------------------------------------------


class TestSchemaExtractors:
    def test_strip_sql_identifier_basic(self) -> None:
        from contextmine_core.analyzer.extractors.schema import _strip_sql_identifier

        assert _strip_sql_identifier('"users"') == "users"
        assert _strip_sql_identifier("`orders`") == "orders"
        assert _strip_sql_identifier("[items]") == "items"
        assert _strip_sql_identifier("  schema.table  ") == "schema.table"
        assert _strip_sql_identifier("") == ""

    def test_split_sql_items(self) -> None:
        from contextmine_core.analyzer.extractors.schema import _split_sql_items

        result = _split_sql_items("id INT, name VARCHAR(50), email TEXT")
        assert len(result) == 3
        assert "id INT" in result[0]

    def test_split_sql_items_with_nested_parens(self) -> None:
        from contextmine_core.analyzer.extractors.schema import _split_sql_items

        result = _split_sql_items("id INT, name VARCHAR(50), CHECK(length(name) > 0)")
        assert len(result) == 3

    def test_split_sql_items_with_quotes(self) -> None:
        from contextmine_core.analyzer.extractors.schema import _split_sql_items

        result = _split_sql_items("id INT, name VARCHAR DEFAULT 'hello, world'")
        assert len(result) == 2

    def test_normalize_sql_type(self) -> None:
        from contextmine_core.analyzer.extractors.schema import _normalize_sql_type

        assert _normalize_sql_type("VARCHAR(255)") == "String"
        assert _normalize_sql_type("TEXT") == "Text"
        assert _normalize_sql_type("BIGINT") == "BigInteger"
        assert _normalize_sql_type("SMALLINT") == "SmallInteger"
        assert _normalize_sql_type("INTEGER") == "Integer"
        assert _normalize_sql_type("SERIAL") == "Integer"
        assert _normalize_sql_type("BOOLEAN") == "Boolean"
        assert _normalize_sql_type("TIMESTAMP") == "DateTime"
        assert _normalize_sql_type("DATE") == "Date"
        assert _normalize_sql_type("TIME") == "Time"
        assert _normalize_sql_type("UUID") == "UUID"
        assert _normalize_sql_type("JSONB") == "JSON"
        assert _normalize_sql_type("NUMERIC(10,2)") == "Numeric"
        assert _normalize_sql_type("FLOAT") == "Float"
        assert _normalize_sql_type("DOUBLE") == "Float"
        assert _normalize_sql_type("BYTEA") == "LargeBinary"
        assert _normalize_sql_type("BLOB") == "LargeBinary"
        assert _normalize_sql_type("") == "unknown"
        assert _normalize_sql_type("CUSTOM_TYPE") == "CUSTOM_TYPE"
        assert _normalize_sql_type("BIGSERIAL") == "BigInteger"
        assert _normalize_sql_type("SMALLSERIAL") == "SmallInteger"
        assert _normalize_sql_type("REAL") == "Float"
        assert _normalize_sql_type("MONEY") == "Numeric"
        assert _normalize_sql_type("DECIMAL(10,2)") == "Numeric"
        assert _normalize_sql_type("character varying(100)") == "String"
        assert _normalize_sql_type("CHAR(10)") == "String"
        assert _normalize_sql_type("VARBINARY(MAX)") == "LargeBinary"

    def test_extract_schema_from_sql_ddl(self) -> None:
        from contextmine_core.analyzer.extractors.schema import _extract_schema_from_sql_ddl

        sql = """
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email TEXT,
    role_id INTEGER REFERENCES roles(id),
    FOREIGN KEY (role_id) REFERENCES roles(id)
);

CREATE TABLE IF NOT EXISTS orders (
    id BIGSERIAL,
    user_id INTEGER,
    amount NUMERIC(10,2),
    created_at TIMESTAMP,
    PRIMARY KEY (id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""
        extraction = _extract_schema_from_sql_ddl("migrations/001.sql", sql)
        assert extraction.framework == "sql"
        assert len(extraction.tables) == 2
        users_table = next(t for t in extraction.tables if t.name == "users")
        assert len(users_table.columns) >= 3
        orders_table = next(t for t in extraction.tables if t.name == "orders")
        assert len(orders_table.primary_keys) >= 1
        assert len(extraction.foreign_keys) >= 1

    def test_extract_schema_ddl_no_tables(self) -> None:
        from contextmine_core.analyzer.extractors.schema import _extract_schema_from_sql_ddl

        extraction = _extract_schema_from_sql_ddl("schema.sql", "SELECT 1;")
        assert len(extraction.tables) == 0

    def test_extract_schema_ddl_constraints(self) -> None:
        from contextmine_core.analyzer.extractors.schema import _extract_schema_from_sql_ddl

        sql = """
CREATE TABLE items (
    id INT,
    name TEXT NOT NULL,
    CONSTRAINT pk_items PRIMARY KEY (id),
    UNIQUE (name),
    INDEX idx_name (name),
    CHECK (id > 0)
);
"""
        extraction = _extract_schema_from_sql_ddl("schema.sql", sql)
        assert len(extraction.tables) == 1
        # Constraint lines should be skipped as columns
        table = extraction.tables[0]
        # Only id and name should be columns, not the constraint lines
        col_names = [c.name for c in table.columns]
        assert "id" in col_names
        assert "name" in col_names


# ---------------------------------------------------------------------------
# OpenAPI extraction (openapi.py lines 65-175)
# ---------------------------------------------------------------------------


class TestOpenAPIExtractor:
    def test_extract_openapi_3(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi

        content = """
openapi: "3.0.0"
info:
  title: Test API
  version: "1.0.0"
servers:
  - url: /api/v1
paths:
  /users:
    get:
      operationId: listUsers
      summary: List all users
      tags:
        - users
      responses:
        "200":
          description: OK
    post:
      operationId: createUser
      summary: Create a user
      requestBody:
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/User"
components:
  schemas:
    User:
      type: object
      properties:
        name:
          type: string
"""
        result = extract_from_openapi("openapi.yaml", content)
        assert result.title == "Test API"
        assert result.version == "1.0.0"
        assert result.base_path == "/api/v1"
        assert len(result.endpoints) >= 2
        assert "User" in result.schemas

    def test_extract_swagger_2(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi

        content = """
swagger: "2.0"
info:
  title: Legacy API
  version: "0.1"
basePath: /api
paths:
  /items:
    get:
      operationId: listItems
      summary: List items
      parameters:
        - in: body
          name: body
          schema:
            $ref: "#/definitions/Item"
      responses:
        200:
          description: OK
definitions:
  Item:
    type: object
    properties:
      name:
        type: string
"""
        result = extract_from_openapi("swagger.yaml", content)
        assert result.title == "Legacy API"
        assert result.base_path == "/api"
        assert len(result.endpoints) >= 1
        assert "Item" in result.schemas

    def test_extract_openapi_invalid(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi

        result = extract_from_openapi("spec.yaml", "{{invalid")
        assert len(result.endpoints) == 0

    def test_extract_openapi_not_dict(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi

        result = extract_from_openapi("spec.yaml", "- item1\n- item2")
        assert len(result.endpoints) == 0

    def test_from_document_non_dict(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi_document

        result = extract_from_openapi_document("spec.yaml", "not a dict")
        assert len(result.endpoints) == 0


# ---------------------------------------------------------------------------
# Protobuf extraction (protobuf.py lines 85-165)
# ---------------------------------------------------------------------------


class TestProtobufExtractor:
    def test_extract_basic_proto(self) -> None:
        from contextmine_core.analyzer.extractors.protobuf import extract_from_protobuf

        content = """
syntax = "proto3";
package example.v1;

import "google/protobuf/timestamp.proto";

enum Status {
    STATUS_UNSPECIFIED = 0;
    STATUS_ACTIVE = 1;
    STATUS_INACTIVE = 2;
}

message User {
    string id = 1;
    string name = 2;
    Status status = 3;
    google.protobuf.Timestamp created_at = 4;

    message Address {
        string street = 1;
    }

    enum Role {
        ROLE_UNSPECIFIED = 0;
        ROLE_ADMIN = 1;
    }
}

service UserService {
    rpc GetUser (GetUserRequest) returns (User);
    rpc ListUsers (ListUsersRequest) returns (ListUsersResponse);
}

message GetUserRequest {
    string user_id = 1;
}

message ListUsersRequest {
    int32 page_size = 1;
}

message ListUsersResponse {
    repeated User users = 1;
}
"""
        result = extract_from_protobuf("user.proto", content)
        assert result.syntax == "proto3"
        assert result.package == "example.v1"
        assert len(result.imports) >= 1
        assert len(result.enums) >= 1
        assert len(result.messages) >= 3
        # User message has nested message and enum
        user_msg = next(m for m in result.messages if m.name == "User")
        assert len(user_msg.fields) >= 3
        assert len(user_msg.nested_messages) >= 1
        assert len(user_msg.nested_enums) >= 1
        # Services
        assert len(result.services) >= 1
        svc = result.services[0]
        assert svc.name == "UserService"
        assert len(svc.rpcs) >= 2


# ---------------------------------------------------------------------------
# Jobs extraction (jobs.py lines 167-220)
# ---------------------------------------------------------------------------


class TestJobsHelpers:
    def test_is_config_file_skip_binaries(self) -> None:
        from contextmine_core.analyzer.extractors.jobs import _is_config_file

        assert _is_config_file("image.png") is False
        assert _is_config_file("archive.zip") is False
        assert _is_config_file("lib.dll") is False
        assert _is_config_file("cache.pyc") is False
        assert _is_config_file("lockfile.lock") is False

    def test_is_config_file_accepts_config(self) -> None:
        from contextmine_core.analyzer.extractors.jobs import _is_config_file

        assert _is_config_file("config.yaml") is True
        assert _is_config_file("settings.json") is True
        assert _is_config_file("deploy.toml") is True
        assert _is_config_file("tasks.py") is True
        assert _is_config_file("pipeline.ts") is True
        assert _is_config_file("infra.tf") is True
        assert _is_config_file("build.xml") is True
        assert _is_config_file("Jenkinsfile") is True

    def test_is_config_file_rejects_unknown(self) -> None:
        from contextmine_core.analyzer.extractors.jobs import _is_config_file

        assert _is_config_file("readme.md") is False
        assert _is_config_file("style.css") is False


# ---------------------------------------------------------------------------
# Test extractor: deeper Python and JS extraction
# ---------------------------------------------------------------------------


class TestTestExtractorDeep:
    def test_python_test_with_fixtures_and_signals(self) -> None:
        from contextmine_core.analyzer.extractors.tests import extract_tests_from_file

        code = """
import pytest

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(scope="session")
def db():
    return create_db()

class TestUserAPI:
    def test_create_user(self, client, db):
        response = client.post("/api/users", json={"name": "alice"})
        assert response.status_code == 201
        user = response.json()
        assert user["name"] == "alice"

    def test_get_user(self, client):
        response = client.get("/api/users/1")
        assert response.status_code == 200

def test_standalone_function():
    result = compute_hash("data")
    assert len(result) == 64
"""
        extraction = extract_tests_from_file("tests/test_api.py", code)
        assert extraction.framework == "pytest"
        assert len(extraction.fixtures) >= 1
        assert len(extraction.cases) >= 3
        assert len(extraction.suites) >= 1
        # Check signals were extracted
        for case in extraction.cases:
            if case.name == "test_create_user":
                assert len(case.call_sites) > 0
                break

    def test_js_test_with_describe_and_beforeeach(self) -> None:
        from contextmine_core.analyzer.extractors.tests import extract_tests_from_file

        code = """
import { describe, it, expect, beforeEach, afterEach } from 'jest';

describe('AuthService', () => {
  beforeEach(() => {
    setupMocks();
  });

  afterEach(() => {
    clearMocks();
  });

  it('should login user', () => {
    const result = authService.login('user', 'pass');
    expect(result.token).toBeDefined();
  });

  it('should reject bad password', () => {
    expect(() => authService.login('user', 'wrong')).toThrow();
  });

  describe('admin operations', () => {
    it('should list users', () => {
      const users = authService.listUsers();
      expect(users).toHaveLength(3);
    });
  });
});
"""
        extraction = extract_tests_from_file("tests/auth.test.ts", code)
        assert extraction.framework == "jest"
        assert len(extraction.suites) >= 2  # 'AuthService' + 'admin operations'
        assert len(extraction.cases) >= 3
        assert len(extraction.fixtures) >= 2  # beforeEach + afterEach

    def test_python_test_with_endpoint_hints(self) -> None:
        from contextmine_core.analyzer.extractors.tests import extract_tests_from_file

        code = """
import pytest
import requests

def test_health_check():
    resp = requests.get("http://localhost:8000/health")
    assert resp.status_code == 200

def test_api_users():
    resp = requests.post("http://localhost:8000/api/users")
    assert resp.status_code == 201
"""
        extraction = extract_tests_from_file("tests/test_endpoints.py", code)
        assert len(extraction.cases) >= 2
        for case in extraction.cases:
            if case.name == "test_health_check":
                assert len(case.endpoint_hints) > 0 or len(case.call_sites) > 0

    def test_js_test_with_endpoint_calls(self) -> None:
        from contextmine_core.analyzer.extractors.tests import extract_tests_from_file

        code = """
import { describe, it, expect } from 'jest';

describe('API tests', () => {
  it('should fetch users', async () => {
    const resp = await fetch('/api/users');
    const data = await resp.json();
    expect(data.length).toBeGreaterThan(0);
  });
});
"""
        extraction = extract_tests_from_file("tests/api.test.ts", code)
        assert len(extraction.cases) >= 1
        assert len(extraction.suites) >= 1


# ---------------------------------------------------------------------------
# OpenAPI: extract_from_openapi_document more branches
# ---------------------------------------------------------------------------


class TestOpenAPIDocumentBranches:
    def test_no_servers_no_basepath(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi_document

        doc = {
            "openapi": "3.0.0",
            "paths": {
                "/test": {"get": {"summary": "Test"}},
            },
        }
        result = extract_from_openapi_document("spec.yaml", doc)
        assert len(result.endpoints) == 1
        assert result.base_path is None

    def test_basepath_from_swagger2(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi_document

        doc = {
            "swagger": "2.0",
            "basePath": "/api/v2",
            "paths": {
                "/users": {"get": {"operationId": "listUsers"}},
            },
        }
        result = extract_from_openapi_document("swagger.yaml", doc)
        assert result.base_path == "/api/v2"
        assert len(result.endpoints) == 1

    def test_non_dict_path_items_skipped(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi_document

        doc = {
            "openapi": "3.0.0",
            "paths": {
                "/good": {"get": {"summary": "Works"}},
                "/bad": "not a dict",
            },
        }
        result = extract_from_openapi_document("spec.yaml", doc)
        assert len(result.endpoints) == 1

    def test_swagger2_body_param_ref(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi_document

        doc = {
            "swagger": "2.0",
            "paths": {
                "/create": {
                    "post": {
                        "operationId": "create",
                        "parameters": [
                            {
                                "in": "body",
                                "name": "body",
                                "schema": {"$ref": "#/definitions/CreateRequest"},
                            },
                        ],
                    },
                },
            },
        }
        result = extract_from_openapi_document("swagger.yaml", doc)
        assert len(result.endpoints) == 1
        endpoint = result.endpoints[0]
        assert endpoint.request_body_ref == "CreateRequest"

    def test_definitions_schemas(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi_document

        doc = {
            "swagger": "2.0",
            "paths": {},
            "definitions": {
                "User": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        }
        result = extract_from_openapi_document("swagger.yaml", doc)
        assert "User" in result.schemas

    def test_components_schemas(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi_document

        doc = {
            "openapi": "3.0.0",
            "paths": {},
            "components": {
                "schemas": {
                    "Order": {"type": "object", "properties": {"id": {"type": "integer"}}},
                },
            },
        }
        result = extract_from_openapi_document("spec.yaml", doc)
        assert "Order" in result.schemas
