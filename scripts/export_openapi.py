"""Export the FastAPI schema used by the generated web client."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    # Schema generation must not depend on production authentication or model access.
    os.environ.setdefault("DEBUG", "true")
    os.environ.setdefault("MODEL_CALLS_ENABLED", "false")

    from app.main import create_app

    output = Path(__file__).parents[1] / "apps" / "web" / "openapi.json"
    output.write_text(
        json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
