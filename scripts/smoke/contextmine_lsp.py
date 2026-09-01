"""Exercise a real language server through ContextMine's LSP adapter."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

from contextmine_core.lsp.manager import LspManager


async def exercise_lsp() -> dict[str, str]:
    """Resolve a cross-file TypeScript definition through LspManager."""
    server_binary = shutil.which("typescript-language-server")
    if server_binary is None:
        raise RuntimeError("typescript-language-server is not available on PATH")

    with tempfile.TemporaryDirectory(prefix="contextmine-lsp-manager-") as directory:
        workspace = Path(directory)
        definition_path = workspace / "greeting.ts"
        usage_path = workspace / "index.ts"

        definition_path.write_text(
            "export function greet(name: string): string {\n  return `Hello ${name}`\n}\n"
        )
        usage_path.write_text(
            'import { greet } from "./greeting"\n\n'
            'const message = greet("ContextMine")\n'
            "console.log(message)\n"
        )
        (workspace / "tsconfig.json").write_text(
            json.dumps(
                {
                    "compilerOptions": {
                        "strict": True,
                        "target": "ES2022",
                        "module": "ESNext",
                        "moduleResolution": "Bundler",
                        "noEmit": True,
                    },
                    "files": ["greeting.ts", "index.ts"],
                }
            )
        )

        manager = LspManager(request_timeout_seconds=30)
        try:
            client = await manager.get_client(usage_path, project_root=workspace)
            cached_client = await manager.get_client(usage_path, project_root=workspace)
            if cached_client is not client:
                raise RuntimeError("LspManager did not reuse its cached client")

            hover = await client.get_hover(str(usage_path), line=3, column=18)
            if hover is None or not hover.signature:
                raise RuntimeError("ContextMine LSP adapter returned no hover information")

            definitions = await client.get_definition(str(usage_path), line=3, column=18)
            if not any(
                Path(location.file_path).resolve() == definition_path for location in definitions
            ):
                raise RuntimeError(
                    "ContextMine LSP adapter did not resolve the cross-file definition"
                )
        finally:
            await manager.shutdown()

    return {
        "cache": "pass",
        "cross_file_definition": "pass",
        "hover": "pass",
        "multilspy": importlib.metadata.version("multilspy"),
        "server_binary": server_binary,
    }


def main() -> None:
    """Run the async adapter smoke and emit a compact machine-readable result."""
    print(json.dumps(asyncio.run(exercise_lsp()), sort_keys=True))


if __name__ == "__main__":
    main()
