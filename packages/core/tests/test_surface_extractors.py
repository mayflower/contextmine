"""Tests for System Surface Catalog extractors."""

from contextmine_core.analyzer.extractors.graphql import extract_from_graphql
from contextmine_core.analyzer.extractors.jobs import JobKind, extract_jobs
from contextmine_core.analyzer.extractors.openapi import extract_from_openapi
from contextmine_core.analyzer.extractors.protobuf import extract_from_protobuf
from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor


class TestOpenAPIExtractor:
    """Tests for OpenAPI specification parsing."""

    def test_extract_basic_endpoints(self) -> None:
        """Test extracting endpoints from OpenAPI spec."""
        content = """
openapi: "3.0.0"
info:
  title: Test API
  version: "1.0.0"
paths:
  /users:
    get:
      operationId: listUsers
      summary: List all users
      tags:
        - users
      responses:
        "200":
          description: Success
  /users/{id}:
    get:
      operationId: getUser
      summary: Get user by ID
      responses:
        "200":
          description: Success
    delete:
      operationId: deleteUser
      summary: Delete a user
      responses:
        "204":
          description: No Content
"""
        result = extract_from_openapi("api.yaml", content)

        assert result.title == "Test API"
        assert result.version == "1.0.0"
        assert len(result.endpoints) == 3

        endpoints_by_op = {e.operation_id: e for e in result.endpoints}
        assert "listUsers" in endpoints_by_op
        assert endpoints_by_op["listUsers"].method == "GET"
        assert endpoints_by_op["listUsers"].path == "/users"
        assert "users" in endpoints_by_op["listUsers"].tags

        assert "deleteUser" in endpoints_by_op
        assert endpoints_by_op["deleteUser"].method == "DELETE"

    def test_extract_request_response_schemas(self) -> None:
        """Test extracting schema references."""
        content = """
openapi: "3.0.0"
info:
  title: Test API
  version: "1.0.0"
paths:
  /users:
    post:
      operationId: createUser
      requestBody:
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateUserRequest"
      responses:
        "201":
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/User"
components:
  schemas:
    CreateUserRequest:
      type: object
      properties:
        name:
          type: string
        email:
          type: string
    User:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
"""
        result = extract_from_openapi("api.yaml", content)

        assert len(result.endpoints) == 1
        endpoint = result.endpoints[0]
        assert endpoint.request_body_ref == "CreateUserRequest"
        assert endpoint.response_refs.get("201") == "User"

        assert "CreateUserRequest" in result.schemas
        assert "User" in result.schemas


class TestGraphQLExtractor:
    """Tests for GraphQL schema parsing."""

    def test_extract_types(self) -> None:
        """Test extracting GraphQL types."""
        content = """
type User {
  id: ID!
  name: String!
  email: String
  posts: [Post!]!
}

type Post {
  id: ID!
  title: String!
  author: User!
}

enum Status {
  DRAFT
  PUBLISHED
  ARCHIVED
}
"""
        result = extract_from_graphql("schema.graphql", content)

        assert len(result.types) == 3

        types_by_name = {t.name: t for t in result.types}
        assert "User" in types_by_name
        assert "Post" in types_by_name
        assert "Status" in types_by_name

        user = types_by_name["User"]
        assert user.kind == "type"
        assert len(user.fields) == 4
        field_names = {f.name for f in user.fields}
        assert "id" in field_names
        assert "name" in field_names
        assert "posts" in field_names

        status = types_by_name["Status"]
        assert status.kind == "enum"
        assert "DRAFT" in status.enum_values
        assert "PUBLISHED" in status.enum_values

    def test_extract_operations(self) -> None:
        """Test extracting Query/Mutation operations."""
        content = """
type Query {
  users: [User!]!
  user(id: ID!): User
}

type Mutation {
  createUser(name: String!, email: String!): User!
  deleteUser(id: ID!): Boolean!
}
"""
        result = extract_from_graphql("schema.graphql", content)

        assert len(result.operations) == 2

        ops_by_name = {o.name: o for o in result.operations}
        assert "Query" in ops_by_name
        assert "Mutation" in ops_by_name

        query = ops_by_name["Query"]
        assert len(query.fields) == 2

        mutation = ops_by_name["Mutation"]
        assert len(mutation.fields) == 2


class TestProtobufExtractor:
    """Tests for Protobuf file parsing."""

    def test_extract_messages(self) -> None:
        """Test extracting Protobuf messages."""
        content = """
syntax = "proto3";

package myapp.users;

message User {
  string id = 1;
  string name = 2;
  string email = 3;
  repeated string tags = 4;
}

message CreateUserRequest {
  string name = 1;
  string email = 2;
}
"""
        result = extract_from_protobuf("users.proto", content)

        assert result.syntax == "proto3"
        assert result.package == "myapp.users"
        assert len(result.messages) == 2

        msgs_by_name = {m.name: m for m in result.messages}
        assert "User" in msgs_by_name
        assert "CreateUserRequest" in msgs_by_name

        user = msgs_by_name["User"]
        assert len(user.fields) == 4

        tags_field = next(f for f in user.fields if f.name == "tags")
        assert tags_field.repeated is True

    def test_extract_services(self) -> None:
        """Test extracting Protobuf services."""
        content = """
syntax = "proto3";

package myapp.users;

service UserService {
  rpc GetUser(GetUserRequest) returns (User);
  rpc ListUsers(ListUsersRequest) returns (stream User);
  rpc CreateUser(CreateUserRequest) returns (User);
}

message GetUserRequest {
  string id = 1;
}

message ListUsersRequest {
  int32 limit = 1;
}

message CreateUserRequest {
  string name = 1;
}

message User {
  string id = 1;
  string name = 2;
}
"""
        result = extract_from_protobuf("users.proto", content)

        assert len(result.services) == 1
        service = result.services[0]
        assert service.name == "UserService"
        assert len(service.rpcs) == 3

        rpcs_by_name = {r.name: r for r in service.rpcs}
        assert "GetUser" in rpcs_by_name
        assert "ListUsers" in rpcs_by_name
        assert "CreateUser" in rpcs_by_name

        list_rpc = rpcs_by_name["ListUsers"]
        assert list_rpc.response_stream is True
        assert list_rpc.request_stream is False


class TestJobsExtractor:
    """Tests for job definition parsing."""

    def test_extract_github_workflow(self) -> None:
        """Test extracting GitHub Actions workflow."""
        content = """
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - name: Build
        run: docker build .
"""
        result = extract_jobs(".github/workflows/ci.yml", content)

        assert len(result.jobs) == 2

        jobs_by_name = {j.name: j for j in result.jobs}
        assert "test" in jobs_by_name
        assert "build" in jobs_by_name

        test_job = jobs_by_name["test"]
        assert test_job.kind == JobKind.GITHUB_JOB
        assert test_job.runs_on == "ubuntu-latest"
        assert len(test_job.steps) == 2

    def test_extract_github_scheduled_workflow(self) -> None:
        """Test extracting scheduled GitHub workflow."""
        content = """
name: Nightly Build

on:
  schedule:
    - cron: "0 0 * * *"

jobs:
  nightly:
    runs-on: ubuntu-latest
    steps:
      - name: Nightly task
        run: echo "Nightly"
"""
        result = extract_jobs(".github/workflows/nightly.yml", content)

        assert len(result.jobs) == 1
        job = result.jobs[0]
        assert any(t.cron == "0 0 * * *" for t in job.triggers)

    def test_extract_k8s_cronjob(self) -> None:
        """Test extracting Kubernetes CronJob."""
        content = """
apiVersion: batch/v1
kind: CronJob
metadata:
  name: backup-job
  namespace: production
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: backup:latest
              command: ["/bin/sh", "-c", "backup.sh"]
          restartPolicy: OnFailure
"""
        result = extract_jobs("backup-cronjob.yaml", content)

        assert len(result.jobs) == 1
        job = result.jobs[0]
        assert job.kind == JobKind.K8S_CRONJOB
        assert job.name == "backup-job"
        assert job.schedule == "0 2 * * *"
        assert job.container_image == "backup:latest"

    def test_extract_prefect_deployment(self) -> None:
        """Test extracting Prefect deployment."""
        content = """
deployments:
  - name: daily-sync
    flow_name: sync_data
    entrypoint: flows/sync.py:sync_data
    schedules:
      - cron: "0 6 * * *"
    work_pool:
      name: kubernetes-pool
"""
        result = extract_jobs("prefect.yaml", content)

        assert len(result.jobs) == 1
        job = result.jobs[0]
        assert job.kind == JobKind.PREFECT_DEPLOYMENT
        assert job.name == "daily-sync"
        assert job.schedule == "0 6 * * *"


class TestSurfaceCatalogExtractor:
    """Tests for the unified surface catalog extractor."""

    def test_auto_detect_openapi(self) -> None:
        """Test auto-detection of OpenAPI specs."""
        extractor = SurfaceCatalogExtractor()

        content = """
openapi: "3.0.0"
info:
  title: Test
  version: "1.0"
paths:
  /test:
    get:
      responses:
        "200":
          description: OK
"""
        result = extractor.add_file("api/openapi.yaml", content)
        assert result is True
        assert len(extractor.catalog.openapi_specs) == 1

    def test_auto_detect_graphql(self) -> None:
        """Test auto-detection of GraphQL schemas."""
        extractor = SurfaceCatalogExtractor()

        content = """
type Query {
  hello: String!
}
"""
        result = extractor.add_file("schema.graphql", content)
        assert result is True
        assert len(extractor.catalog.graphql_schemas) == 1

    def test_auto_detect_proto(self) -> None:
        """Test auto-detection of Protobuf files."""
        extractor = SurfaceCatalogExtractor()

        content = """
syntax = "proto3";
message Test {
  string id = 1;
}
"""
        result = extractor.add_file("test.proto", content)
        assert result is True
        assert len(extractor.catalog.protobuf_files) == 1

    def test_auto_detect_workflow(self) -> None:
        """Test auto-detection of GitHub Actions workflows."""
        extractor = SurfaceCatalogExtractor()

        content = """
name: Test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo test
"""
        result = extractor.add_file(".github/workflows/test.yml", content)
        assert result is True
        assert len(extractor.catalog.job_definitions) == 1
