"""
Snapshot utilities.

Used by theme_scan_core.py.

Public API:
- snapshot_outputs(out_dir: str, files: list[str]) -> pathlib.Path | None

Creates a timestamped subdirectory under `out_dir` (UTC, YYYYMMDDHHMM) and copies
specified files into it.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, Optional


def _utc_stamp_yyyymmddhhmm() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M")

def _jst_stamp_yyyymmddhh() -> str:
    utc_now = datetime.now(timezone.utc)
    jst_now = utc_now.astimezone(timezone(timedelta(hours=9)))
    jst_formatted = jst_now.strftime("%Y%m%d%H%M")
    return jst_formatted

def snapshot_outputs(out_dir: str | Path, files: Iterable[str]) -> Optional[Path]:
    """
    Copy `files` into out_dir/<YYYYMMDDHHMM>/.

    Only copies regular files that exist.
    Returns the snapshot directory path, or None if there was nothing to copy.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_list = [str(p) for p in files if p and os.path.isfile(p)]
    if not file_list:
        return None

    snap_dir = out_dir / _jst_stamp_yyyymmddhh()
    snap_dir.mkdir(parents=True, exist_ok=True)

    for src in file_list:
        src_p = Path(src)
        dst_p = snap_dir / src_p.name
        shutil.copy2(src_p, dst_p)

    return snap_dir


__all__ = ["snapshot_outputs"]