# pipeline/validate.py
from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from jsonschema import Draft202012Validator


def load_schema(schema_path: str | Path) -> Dict[str, Any]:
    p = Path(schema_path)
    return json.loads(p.read_text(encoding="utf-8"))


def validate_records(schema: Dict[str, Any], records: List[Dict[str, Any]]) -> List[str]:
    """Validate a list of records.

    If `schema` describes a *single record object*, each record is validated.
    If `schema` describes an *envelope* with a `rows` array, callers should use
    `must_validate()` which will wrap records and validate the whole payload.

    Returns list of error strings (empty => OK).
    """
    v = Draft202012Validator(schema)
    errs: List[str] = []
    for i, rec in enumerate(records):
        for e in v.iter_errors(rec):
            path = ".".join([str(x) for x in e.absolute_path]) if e.absolute_path else "(root)"
            errs.append(f"row={i} path={path} msg={e.message}")
    return errs


def _is_envelope_schema(schema: Dict[str, Any]) -> bool:
    """Heuristic: schema is an envelope if it has a top-level `rows` property."""
    props = schema.get("properties")
    if not isinstance(props, dict):
        return False
    return "rows" in props


def must_validate(schema_path: str | Path, records_or_payload: Any) -> Any:
    """Validate data against a JSON Schema.

    Supported inputs:
      - list[dict]: treated as "rows" and validated against either a per-record schema
        (each dict validated) or an envelope schema (auto-wrapped).
      - dict: validated as-is (useful when caller already built an envelope).

    Returns the validated instance (possibly wrapped in an envelope).
    """
    schema = load_schema(schema_path)

    # Case 1: caller already passed a payload object
    if isinstance(records_or_payload, dict):
        v = Draft202012Validator(schema)
        errs: List[str] = []
        for e in v.iter_errors(records_or_payload):
            path = ".".join([str(x) for x in e.absolute_path]) if e.absolute_path else "(root)"
            errs.append(f"path={path} msg={e.message}")
        if errs:
            head = "\n".join(errs[:30])
            more = "" if len(errs) <= 30 else f"\n... ({len(errs)-30} more)"
            raise ValueError(f"Schema validation failed for {schema_path}:\n{head}{more}")
        return records_or_payload

    # Case 2: caller passed rows (list of dicts)
    if not isinstance(records_or_payload, list):
        raise TypeError(f"must_validate expects list[dict] or dict, got={type(records_or_payload)!r}")

    records: List[Dict[str, Any]] = records_or_payload

    # If schema is an envelope, wrap rows and validate the whole object once.
    if _is_envelope_schema(schema):
        payload = {
            "schema_version": "1.0",
            "generated_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "rows": records,
        }
        return must_validate(schema_path, payload)

    # Otherwise, schema is assumed to be per-record.
    errs = validate_records(schema, records)
    if errs:
        head = "\n".join(errs[:30])
        more = "" if len(errs) <= 30 else f"\n... ({len(errs)-30} more)"
        raise ValueError(f"Schema validation failed for {schema_path}:\n{head}{more}")
    return records
