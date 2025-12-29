## ⚠️ IMPORTANT: Codex / AI Code Generation Guard

This repository uses AI-assisted code generation (Codex).
To prevent destructive or unintended changes, **the following rules are mandatory**:

- ❌ Do NOT modify existing files unless explicitly instructed.
- ❌ Do NOT refactor, rename, or reformat existing logic.
- ❌ Do NOT optimize or “clean up” code unless asked.
- ✅ Add new functionality ONLY by creating new files or modules.
- ✅ Assume backward compatibility is critical.
- ✅ When unsure, STOP and ask for clarification.

Breaking these rules is considered a critical error.


出力CSVは contracts/step1_universe.schema.json に必ず準拠
schema にない列は追加しない
型・列名を変えない


# etf-theme-scan

ETFの **保有銘柄（Holdings）CSV / XLSX** を入力として、  
テーマ別に銘柄をスキャン・集計・レポート化するための CLI ツールです。

主に以下を目的としています。

- ETFの中身を **テーマ（Theme）視点で可視化**
- CSV / XLSX の **誤投入耐性**
- ローカルCSV固定運用（再現性重視）
- 機械処理しやすい **CSV / Markdown 出力**

---

## Features

- ✅ ETFテーマ別スキャン（例: SMH）
- ✅ Holdings CSV / XLSX 両対応（VanEck / SSGA 想定）
- ✅ XLSX誤投入の自動検知
- ✅ CLIベースで再現性のある実行
- ✅ レポート自動生成
  - サマリーCSV
  - Markdownレポート
  - 実行スナップショット

---

## Directory Structure

.
├── theme_scan_cli.py        # CLI エントリーポイント
├── readers/
│   ├── vaneck.py            # VanEck CSV / XLSX reader
│   └── ssga.py              # SSGA reader
├── data/
│   └── holdings/            # ローカル固定のETF Holdings
├── out/
│   ├── summary_themes.csv
│   ├── report_ALL.md
│   └── YYYYMMDDHHMM/        # snapshot
├── requirements.txt
└── README.md

---

## Requirements

- Python 3.9+
- venv 推奨

```bash
pip install -r requirements.txt


⸻

Usage

基本実行例

python theme_scan_cli.py \
  --themes SMH \
  --holdings ssga \
  --out ./out

オプション

option	description
--themes	テーマ名（例: SMH）
--holdings	holdings種別（ssga / vaneck）
--out	出力ディレクトリ


⸻

Holdings Data Policy
	•	SMH はローカルCSV固定
	•	外部ダウンロードは行わない
	•	CSV / XLSX の誤投入は自動検知して安全に失敗

⸻

Output

1. サマリーCSV

out/summary_themes.csv

2. Markdownレポート

out/report_ALL.md

3. 実行スナップショット

out/YYYYMMDDHHMM/


⸻

Design Philosophy
	•	指標より 構造
	•	可視化より 再現性
	•	自動化より 壊れにくさ
	•	「人が判断するための下処理」に特化

⸻

Disclaimer

本ツールは 投資助言を目的としません。
出力結果は教育・分析目的のみで使用してください。
