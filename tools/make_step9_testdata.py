#!/usr/bin/env python3
"""
Create test data for Step9 dashboard.

Outputs:
- out/step8/score_validation.csv
- out/step9/thresholds_final.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT_CSV = Path("out/step8/score_validation.csv")
OUT_THR = Path("out/step9/thresholds_final.json")


def main() -> int:
    rng = np.random.default_rng(42)

    horizons = [1, 5, 20]
    buckets = [
        ("score 60-69", 60),
        ("score 70-79", 70),
        ("score>=80", 80),
    ]

    rows: list[dict] = []

    # score_bucket rows
    for h in horizons:
        # Create a plausible progression: higher score -> better avg_return/win_rate
        base_ret = {1: 0.0002, 5: 0.0006, 20: 0.0015}[h]
        base_win = {1: 0.48, 5: 0.50, 20: 0.52}[h]

        for label, bmin in buckets:
            # sample sizes
            sample = int(rng.integers(80, 420))

            # improvements by bucket
            if bmin == 60:
                ret_mu = base_ret - abs(base_ret) * 0.6
                win_mu = base_win - 0.06
            elif bmin == 70:
                ret_mu = base_ret + abs(base_ret) * 0.3
                win_mu = base_win + 0.01
            else:  # 80
                ret_mu = base_ret + abs(base_ret) * 1.2
                win_mu = base_win + 0.05

            avg_return = float(rng.normal(ret_mu, max(1e-6, abs(base_ret) * 0.25)))
            win_rate = float(np.clip(rng.normal(win_mu, 0.03), 0.0, 1.0))
            median_return = float(rng.normal(ret_mu * 0.8, max(1e-6, abs(base_ret) * 0.20)))

            rows.append(
                {
                    "horizon": h,
                    "group_type": "score_bucket",
                    "group_key": label,
                    "avg_return": avg_return,
                    "win_rate": win_rate,
                    "median_return": median_return,
                    "sample": sample,
                }
            )

    # (optional) direction rows (Dashboard側が無視してもOK)
    directions = ["up", "flat", "down"]
    for h in horizons:
        base_ret = {1: 0.00015, 5: 0.0005, 20: 0.0012}[h]
        for d in directions:
            sample = int(rng.integers(120, 520))
            drift = {"up": +1.0, "flat": 0.0, "down": -1.0}[d]
            avg_return = float(rng.normal(base_ret * drift, max(1e-6, abs(base_ret) * 0.35)))
            win_rate = float(np.clip(rng.normal(0.50 + 0.06 * drift, 0.04), 0.0, 1.0))
            median_return = float(rng.normal(avg_return * 0.8, max(1e-6, abs(base_ret) * 0.25)))

            rows.append(
                {
                    "horizon": h,
                    "group_type": "direction",
                    "group_key": d,
                    "avg_return": avg_return,
                    "win_rate": win_rate,
                    "median_return": median_return,
                    "sample": sample,
                }
            )

    df = pd.DataFrame(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    thresholds = {
        "horizons": {
            # 例: 1H/5Hは70採用、20Hは80採用
            "1": {"threshold": 70},
            "5": {"threshold": 70},
            "20": {"threshold": 80},
        }
    }
    OUT_THR.parent.mkdir(parents=True, exist_ok=True)
    OUT_THR.write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {OUT_CSV}  rows={len(df)}")
    print(f"[ok] wrote: {OUT_THR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
