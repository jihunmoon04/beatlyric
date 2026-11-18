"""Minimal zero-dependency .env loader for the beatlyric pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from common import ROOT_DIR

_LOADED = False


def load_dotenv(env_path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Load key=value pairs from .env into os.environ and return them."""
    global _LOADED
    path = env_path or (ROOT_DIR / ".env")
    parsed: dict[str, str] = {}
    if not path.exists():
        _LOADED = True
        return parsed

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]
        if not key:
            continue
        parsed[key] = val
        if override or key not in os.environ or not os.environ[key]:
            os.environ[key] = val

    _LOADED = True
    return parsed


def get_env(key: str, default: str = "") -> str:
    """Return an environment variable, loading .env on first access."""
    if not _LOADED:
        load_dotenv()
    return os.environ.get(key, default).strip()
