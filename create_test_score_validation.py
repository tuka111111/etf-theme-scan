# create_test_score_validation.py
# Usage:
#   python create_test_score_validation.py out/step8/score_validation.csv

from __future__ import annotations
import csv
import os
import sys
from pathlib import Path

ROWS = [
    # score_bucket (horizon 1/5/20)
    {"horizon": 1,  "group_type": "score_bucket", "group_key": "score>=80",  "avg_return": 0.0012, "win_rate": 0.56, "sample": 120, "median_return": 0.0006},
    {"horizon": 1,  "group_type": "score_bucket", "group_key": "score70-79", "avg_return": 0.0006, "win_rate": 0.53, "sample": 220, "median_return": 0.0002},
    {"horizon": 1,  "group_type": "score_bucket", "group_key": "score60-69", "avg_return": 0.0002, "win_rate": 0.51, "sample": 260, "median_return": 0.0000},
    {"horizon": 1,  "group_type": "score_bucket", "group_key": "score<60",   "avg_return": -0.0003,"win_rate": 0.48, "sample": 300, "median_return": -0.0001},

    {"horizon": 5,  "group_type": "score_bucket", "group_key": "score>=80",  "avg_return": 0.0035, "win_rate": 0.58, "sample": 120, "median_return": 0.0019},
    {"horizon": 5,  "group_type": "score_bucket", "group_key": "score70-79", "avg_return": 0.0018, "win_rate": 0.55, "sample": 220, "median_return": 0.0009},
    {"horizon": 5,  "group_type": "score_bucket", "group_key": "score60-69", "avg_return": 0.0007, "win_rate": 0.52, "sample": 260, "median_return": 0.0002},
    {"horizon": 5,  "group_type": "score_bucket", "group_key": "score<60",   "avg_return": -0.0008,"win_rate": 0.47, "sample": 300, "median_return": -0.0004},

    {"horizon": 20, "group_type": "score_bucket", "group_key": "score>=80",  "avg_return": 0.0100, "win_rate": 0.62, "sample": 120, "median_return": 0.0052},
    {"horizon": 20, "group_type": "score_bucket", "group_key": "score70-79", "avg_return": 0.0060, "win_rate": 0.58, "sample": 220, "median_return": 0.0027},
    {"horizon": 20, "group_type": "score_bucket", "group_key": "score60-69", "avg_return": 0.0022, "win_rate": 0.53, "sample": 260, "median_return": 0.0010},
    {"horizon": 20, "group_type": "score_bucket", "group_key": "score<60",   "avg_return": -0.0015,"win_rate": 0.46, "sample": 300, "median_return": -0.0006},

    # env (horizon 1/5/20)
    {"horizon": 1,  "group_type": "env", "group_key": "trend", "avg_return": 0.0009, "win_rate": 0.54, "sample": 420, "median_return": 0.0003},
    {"horizon": 1,  "group_type": "env", "group_key": "range", "avg_return": -0.0002,"win_rate": 0.49, "sample": 480, "median_return": -0.0001},

    {"horizon": 5,  "group_type": "env", "group_key": "trend", "avg_return": 0.0026, "win_rate": 0.56, "sample": 420, "median_return": 0.0011},
    {"horizon": 5,  "group_type": "env", "group_key": "range", "avg_return": -0.0005,"win_rate": 0.48, "sample": 480, "median_return": -0.0002},

    {"horizon": 20, "group_type": "env", "group_key": "trend", "avg_return": 0.0075, "win_rate": 0.60, "sample": 420, "median_return": 0.0034},
    {"horizon": 20, "group_type": "env", "group_key": "range", "avg_return": -0.0010,"win_rate": 0.45, "sample": 480, "median_return": -0.0005},

    # flag (horizon 20中心、1/5も少し入れる)
    {"horizon": 20, "group_type": "flag", "group_key": "dir_up",         "avg_return": 0.0085, "win_rate": 0.61, "sample": 260, "median_return": 0.0039},
    {"horizon": 20, "group_type": "flag", "group_key": "dir_flat",       "avg_return": 0.0010, "win_rate": 0.50, "sample": 180, "median_return": 0.0004},
    {"horizon": 20, "group_type": "flag", "group_key": "dir_down",       "avg_return": -0.0040,"win_rate": 0.40, "sample": 160, "median_return": -0.0018},
    {"horizon": 20, "group_type": "flag", "group_key": "atr_high",       "avg_return": 0.0060, "win_rate": 0.57, "sample": 140, "median_return": 0.0021},
    {"horizon": 20, "group_type": "flag", "group_key": "news_avoid=True","avg_return": 0.0042, "win_rate": 0.55, "sample": 90,  "median_return": 0.0017},

    {"horizon": 5,  "group_type": "flag", "group_key": "dir_up",   "avg_return": 0.0029, "win_rate": 0.57, "sample": 260, "median_return": 0.0012},
    {"horizon": 1,  "group_type": "flag", "group_key": "dir_up",   "avg_return": 0.0010, "win_rate": 0.54, "sample": 260, "median_return": 0.0004},
]

FIELDS = ["horizon", "group_type", "group_key", "avg_return", "win_rate", "sample", "median_return"]

def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/step8/score_validation.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in ROWS:
            w.writerow(r)

    print(f"Wrote {len(ROWS)} rows -> {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())