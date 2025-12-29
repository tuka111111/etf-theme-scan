## AI SAFETY BOUNDARY (Pipeline)

This directory is **the only area** where Codex is allowed to add or modify code.

Rules:
- Codex MAY create new files under `pipeline/`
- Codex MAY import from `theme_scan_core.py`
- Codex MUST NOT modify `theme_scan_core.py`
- Codex MUST treat all outputs as append-only artifacts

If a change outside this directory seems necessary:
→ STOP and report the reason instead of editing.