"""JSON I/O helpers."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4), encoding="utf-8")
