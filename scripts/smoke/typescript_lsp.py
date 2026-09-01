"""Exercise the shipped TypeScript language server over JSON-RPC stdio."""

from __future__ import annotations

import json
import os
import select
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


class LspProcess:
    """Small synchronous LSP client used only by the system smoke gate."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._buffer = bytearray()
        self._process = subprocess.Popen(  # noqa: S603
            ["typescript-language-server", "--stdio", "--log-level", "1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def send(self, message: dict[str, Any]) -> None:
        """Send one JSON-RPC message using LSP content-length framing."""
        if self._process.stdin is None:
            raise RuntimeError("language-server stdin is unavailable")

        body = json.dumps(message, separators=(",", ":")).encode()
        header = f"Content-Length: {len(body)}\r\n\r\n".encode()
        self._process.stdin.write(header + body)
        self._process.stdin.flush()

    def request(
        self,
        request_id: int,
        method: str,
        params: dict[str, Any] | None,
        *,
        timeout_seconds: float = 30,
    ) -> Any:
        """Send a request and wait for its response."""
        self.send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        deadline = time.monotonic() + timeout_seconds

        while True:
            message = self._read_message(deadline)
            if "method" in message and "id" in message:
                self._answer_server_request(message)
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(f"LSP {method} failed: {message['error']}")
            return message.get("result")

    def notify(self, method: str, params: dict[str, Any] | None) -> None:
        """Send a JSON-RPC notification."""
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def close(self) -> None:
        """Stop the process even when a protocol assertion failed."""
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

    def stderr(self) -> str:
        """Return bounded server diagnostics after the process stopped."""
        if self._process.stderr is None or self._process.poll() is None:
            return ""
        return self._process.stderr.read().decode(errors="replace")[-4000:]

    def _read_message(self, deadline: float) -> dict[str, Any]:
        if self._process.stdout is None:
            raise RuntimeError("language-server stdout is unavailable")

        while True:
            separator = self._buffer.find(b"\r\n\r\n")
            if separator >= 0:
                header = self._buffer[:separator].decode("ascii")
                content_length = self._content_length(header)
                body_start = separator + 4
                body_end = body_start + content_length
                if len(self._buffer) >= body_end:
                    body = bytes(self._buffer[body_start:body_end])
                    del self._buffer[:body_end]
                    parsed = json.loads(body)
                    if not isinstance(parsed, dict):
                        raise RuntimeError("LSP message is not a JSON object")
                    return parsed

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for an LSP response")

            ready, _, _ = select.select([self._process.stdout.fileno()], [], [], remaining)
            if not ready:
                raise TimeoutError("timed out waiting for LSP output")

            chunk = os.read(self._process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(
                    "language server stopped before responding"
                    f" (exit={self._process.poll()}, stderr={self.stderr()!r})"
                )
            self._buffer.extend(chunk)

    @staticmethod
    def _content_length(header: str) -> int:
        for line in header.split("\r\n"):
            name, separator, value = line.partition(":")
            if separator and name.lower() == "content-length":
                return int(value.strip())
        raise RuntimeError("LSP message is missing Content-Length")

    def _answer_server_request(self, message: dict[str, Any]) -> None:
        method = message["method"]
        params = message.get("params") or {}

        if method == "workspace/configuration":
            result: Any = [None for _ in params.get("items", [])]
        elif method == "workspace/workspaceFolders":
            result = [{"uri": self._workspace.as_uri(), "name": self._workspace.name}]
        else:
            result = None

        self.send({"jsonrpc": "2.0", "id": message["id"], "result": result})


def command_output(*command: str) -> str:
    """Run a bounded version probe."""
    return subprocess.run(  # noqa: S603
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def main() -> None:
    """Initialize TypeScript LSP and prove semantic requests against a fixture."""
    source = """export function greet(name: string): string {
  return `Hello ${name}`
}

const message = greet("ContextMine")
console.log(message)
"""

    with tempfile.TemporaryDirectory(prefix="contextmine-typescript-lsp-") as directory:
        workspace = Path(directory)
        source_path = workspace / "index.ts"
        tsserver_path = (
            Path(command_output("npm", "root", "--global")) / "typescript" / "lib" / "tsserver.js"
        )
        if not tsserver_path.is_file():
            raise RuntimeError(f"global TypeScript server not found at {tsserver_path}")

        source_path.write_text(source)
        (workspace / "tsconfig.json").write_text(
            json.dumps(
                {
                    "compilerOptions": {
                        "strict": True,
                        "target": "ES2022",
                        "module": "ESNext",
                        "noEmit": True,
                    },
                    "files": ["index.ts"],
                }
            )
        )

        client = LspProcess(workspace)
        try:
            capabilities = client.request(
                1,
                "initialize",
                {
                    "processId": os.getpid(),
                    "clientInfo": {"name": "contextmine-smoke", "version": "1"},
                    "rootUri": workspace.as_uri(),
                    "workspaceFolders": [{"uri": workspace.as_uri(), "name": workspace.name}],
                    "capabilities": {
                        "workspace": {"configuration": True, "workspaceFolders": True},
                        "textDocument": {
                            "definition": {"linkSupport": True},
                            "hover": {"contentFormat": ["plaintext"]},
                        },
                    },
                    "initializationOptions": {
                        "tsserver": {"path": str(tsserver_path)},
                    },
                    "trace": "off",
                },
            )
            if not isinstance(capabilities, dict) or "capabilities" not in capabilities:
                raise RuntimeError("LSP initialize response did not contain capabilities")

            client.notify("initialized", {})
            client.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": source_path.as_uri(),
                        "languageId": "typescript",
                        "version": 1,
                        "text": source,
                    }
                },
            )

            position = {"line": 4, "character": 18}
            text_document = {"uri": source_path.as_uri()}
            hover = client.request(
                2,
                "textDocument/hover",
                {"textDocument": text_document, "position": position},
            )
            if not hover:
                raise RuntimeError("TypeScript LSP returned no hover information")

            definition = client.request(
                3,
                "textDocument/definition",
                {"textDocument": text_document, "position": position},
            )
            definitions = definition if isinstance(definition, list) else [definition]
            if not any(
                isinstance(item, dict)
                and (item.get("uri") or item.get("targetUri")) == source_path.as_uri()
                for item in definitions
            ):
                raise RuntimeError("TypeScript LSP did not resolve the local definition")

            client.request(4, "shutdown", None)
            client.notify("exit", None)
            if client._process.wait(timeout=5) != 0:
                raise RuntimeError(
                    f"TypeScript LSP exited unsuccessfully (stderr={client.stderr()!r})"
                )
        finally:
            client.close()

    print(
        json.dumps(
            {
                "definition": "pass",
                "hover": "pass",
                "initialize": "pass",
                "node": command_output("node", "--version"),
                "typescript": command_output("tsc", "--version"),
                "typescript_language_server": command_output(
                    "typescript-language-server", "--version"
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
