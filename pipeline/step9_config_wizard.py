from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline.common import ensure_dir


def _parse_scoring_yaml(path: Path) -> Dict[str, int]:
    thresholds: Dict[str, int] = {}
    if not path.exists():
        return thresholds
    lines = path.read_text(encoding="utf-8").splitlines()
    in_thresholds = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("thresholds:"):
            in_thresholds = True
            continue
        if in_thresholds:
            if ":" not in line:
                continue
            key, val = [p.strip() for p in line.split(":", 1)]
            key = key.strip('"')
            try:
                thresholds[key] = int(float(val))
            except Exception:
                continue
    return thresholds


def _parse_rules_yaml(path: Path) -> Dict[str, object]:
    rules = {
        "flags": {"ignore": [], "weight": {}},
        "env_bias": {},
    }
    if not path.exists():
        return rules
    lines = path.read_text(encoding="utf-8").splitlines()
    section = None
    sub = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("flags:"):
            section = "flags"
            sub = None
            continue
        if line.startswith("env_bias:"):
            section = "env_bias"
            sub = None
            continue
        if section == "flags" and line.startswith("ignore:"):
            sub = "ignore"
            continue
        if section == "flags" and line.startswith("weight:"):
            sub = "weight"
            continue
        if section == "flags" and sub == "ignore" and line.startswith("-"):
            rules["flags"]["ignore"].append(line.lstrip("-").strip())
            continue
        if section == "flags" and sub == "weight" and ":" in line:
            k, v = [p.strip() for p in line.split(":", 1)]
            try:
                rules["flags"]["weight"][k] = float(v)
            except Exception:
                continue
            continue
        if section == "env_bias" and ":" in line and not line.startswith("-"):
            key = line.split(":", 1)[0].strip()
            rules["env_bias"].setdefault(key, {"action": "allow", "mult": 1.0, "score_adjust": 0})
            sub = key
            continue
        if section == "env_bias" and sub in rules["env_bias"] and ":" in line:
            k, v = [p.strip() for p in line.split(":", 1)]
            if k == "score_adjust":
                try:
                    rules["env_bias"][sub][k] = int(float(v))
                except Exception:
                    pass
            elif k == "mult":
                try:
                    rules["env_bias"][sub][k] = float(v)
                except Exception:
                    pass
            elif k == "action":
                rules["env_bias"][sub][k] = v
    return rules


def _write_scoring_yaml(path: Path, thresholds: Dict[str, int]) -> None:
    lines = ["thresholds:"]
    for key, val in thresholds.items():
        lines.append(f"  \"{key}\": {val}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rules_yaml(path: Path, rules: Dict[str, object]) -> None:
    lines = ["flags:", "  ignore:"]
    for item in rules.get("flags", {}).get("ignore", []):
        lines.append(f"    - {item}")
    lines.append("  weight:")
    for k, v in rules.get("flags", {}).get("weight", {}).items():
        lines.append(f"    {k}: {v}")
    lines.append("env_bias:")
    for k, v in rules.get("env_bias", {}).items():
        lines.append(f"  {k}:")
        lines.append(f"    action: {v.get('action', 'allow')}")
        lines.append(f"    mult: {v.get('mult', 1.0)}")
        lines.append(f"    score_adjust: {v.get('score_adjust', 0)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prompt_int(label: str, current: int) -> int:
    val = input(f"{label} [{current}]: ").strip()
    if not val:
        return current
    try:
        return int(float(val))
    except Exception:
        print("Invalid number, keeping current.")
        return current


def _prompt_float(label: str, current: float) -> float:
    val = input(f"{label} [{current}]: ").strip()
    if not val:
        return current
    try:
        return float(val)
    except Exception:
        print("Invalid number, keeping current.")
        return current


def _prompt_list(label: str, current: List[str]) -> List[str]:
    val = input(f"{label} (comma separated) [{', '.join(current)}]: ").strip()
    if not val:
        return current
    return [p.strip() for p in val.split(",") if p.strip()]


def _prompt_weights(current: Dict[str, float]) -> Dict[str, float]:
    print("Enter weights as key=value, comma separated (blank to keep):")
    print(f"current: {current}")
    raw = input("weights: ").strip()
    if not raw:
        return current
    out: Dict[str, float] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        k, v = [p.strip() for p in part.split("=", 1)]
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out if out else current


def _load_thresholds_json(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        print(f"Warning: failed to read thresholds json: {path}")
        return {}
    horizons = payload.get("horizons", {})
    out: Dict[str, int] = {}
    for key, val in horizons.items():
        try:
            threshold = int(float(val.get("threshold")))
        except Exception:
            continue
        out[f"{key}D"] = threshold
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Step9 config wizard for scoring/rules. Example:\n"
            "  Step9-A: python pipeline/step9a_finalize_thresholds.py --input out/step8/score_validation.csv "
            "--out out/step9/thresholds_final.json\n"
            "  Wizard:  python pipeline/step9_config_wizard.py --thresholds-json out/step9/thresholds_final.json\n"
            "  Dashboard: python -m streamlit run pipeline/step9_dashboard.py -- --input out/step8/score_validation.csv"
        )
    )
    ap.add_argument("--config-dir", default="config", help="Config directory relative to repo root")
    ap.add_argument(
        "--thresholds-json",
        default="out/step9/thresholds_final.json",
        help="Optional thresholds_final.json to seed scoring.yaml",
    )
    args = ap.parse_args()

    config_dir = ensure_dir(ROOT / args.config_dir)
    scoring_path = config_dir / "scoring.yaml"
    rules_path = config_dir / "rules.yaml"

    thresholds = _parse_scoring_yaml(scoring_path)
    thresholds_json = _load_thresholds_json(ROOT / args.thresholds_json)
    if thresholds_json:
        for k, v in thresholds_json.items():
            if k not in thresholds:
                thresholds[k] = v
    if not thresholds:
        thresholds = {"1D": 70, "5D": 75, "20D": 80}
    print("\nThresholds")
    for key in sorted(thresholds.keys()):
        thresholds[key] = _prompt_int(f"{key} min_score", thresholds[key])

    rules = _parse_rules_yaml(rules_path)
    print("\nFlags ignore list")
    rules["flags"]["ignore"] = _prompt_list("ignore", rules["flags"].get("ignore", []))
    print("\nFlags weight map")
    rules["flags"]["weight"] = _prompt_weights(rules["flags"].get("weight", {}))

    print("\nEnv bias rules")
    for env_key in ["bull", "neutral", "bear"]:
        rules["env_bias"].setdefault(env_key, {"action": "allow", "mult": 1.0, "score_adjust": 0})
        action = input(f"{env_key} action [" + rules["env_bias"][env_key]["action"] + "]: ").strip()
        if action:
            rules["env_bias"][env_key]["action"] = action
        if rules["env_bias"][env_key]["action"] == "weight":
            rules["env_bias"][env_key]["mult"] = _prompt_float(
                f"{env_key} mult", rules["env_bias"][env_key].get("mult", 1.0)
            )
        else:
            rules["env_bias"][env_key]["mult"] = rules["env_bias"][env_key].get("mult", 1.0)
        rules["env_bias"][env_key]["score_adjust"] = _prompt_int(
            f"{env_key} score_adjust", rules["env_bias"][env_key].get("score_adjust", 0)
        )

    _write_scoring_yaml(scoring_path, thresholds)
    _write_rules_yaml(rules_path, rules)
    print(f"\nSaved: {scoring_path}")
    print(f"Saved: {rules_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
