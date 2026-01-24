from __future__ import annotations

from pathlib import Path
from typing import Optional

_DOTENV_CACHE: dict[str, str] | None = None


def load_dotenv(project_root: Path) -> dict[str, str]:
    global _DOTENV_CACHE
    if _DOTENV_CACHE is not None:
        return _DOTENV_CACHE
    env_path = project_root / ".env"
    data: dict[str, str] = {}
    if not env_path.exists():
        _DOTENV_CACHE = data
        return data
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and ((value[0] == value[-1]) and value[0] in {"'", '"'}):
            value = value[1:-1].strip()
        if not key:
            continue
        data[key] = value
    _DOTENV_CACHE = data
    return data


def get_dotenv(key: str) -> Optional[str]:
    if _DOTENV_CACHE is None:
        return None
    return _DOTENV_CACHE.get(key)
