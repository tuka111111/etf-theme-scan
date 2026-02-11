from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def get_rules_source() -> str:
    return "config/rules.yaml"


def get_rules_hash(rules_path: Path) -> str:
    data: Any = {}
    try:
        if rules_path.exists():
            raw = rules_path.read_text(encoding="utf-8")
            loaded = yaml.safe_load(raw)
            if loaded is not None:
                data = loaded
    except Exception:
        data = {}
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
