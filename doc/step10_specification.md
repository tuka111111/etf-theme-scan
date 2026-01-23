
⸻

1) Runner の役割（責務）
	•	Step6 の最新意思決定（decision_latest）を読み取り
	•	Step7 の行動ログ（trade_actions_*.csv）を読み取り
	•	今日の要約（UI/通知のソース）を生成
	•	逸脱（decision に反する行動）が増えたら警告を生成
	•	アラート用の “ENTER候補のみ” 抽出を生成
	•	すべて append-only / 再実行耐性（idempotency）を持つ

⸻

2) 入力（Input）

2.1 Step6 最新意思決定（必須）
	•	パス：{OUT}/step6_decision/decision_latest.json
	•	例：out/step6_decision/decision_latest.json

必須フィールド（decision_latest.json 内）
runner は次を参照する（存在しない場合はエラー or degraded動作）：

Top-level
	•	schema_version（任意）
	•	generated_at_utc（任意）
	•	asof_date_utc（必須：日付キー。YYYY-MM-DD想定）
	•	asof_local（任意）
	•	asof_utc（任意）
	•	risk_mode.mode（必須：RISK_ON|NEUTRAL|RISK_OFF）
	•	risk_mode.strength（任意：数値）
	•	tradable_themes（必須：list[str]。空でもOK）
	•	picks（必須：dict with keys ENTER/WATCH/AVOID）

picks.ENTER/WATCH/AVOID の各要素（dict）
	•	symbol（必須）
	•	theme（必須）
	•	action（必須：BUY/WATCH/AVOIDのどれかだが、runnerは bucket も優先）
	•	score_total（任意）
	•	score_adjusted（任意）
	•	env（任意）
	•	trend（任意）
	•	flags（任意：list）
	•	notes（任意：list）
	•	threshold_used（任意：string）
	•	rules_applied（任意：list）

runner は、通知・watchlist作成・逸脱判定用に symbol/theme/action/bucket を主に使い、scoreやflags等は補足として扱う。

⸻

2.2 Step7 行動ログ（必須）
	•	ディレクトリ：{OUT}/step7_trades/
	•	例：out/step7_trades/
	•	対象ファイル：trade_actions_*.csv（複数可）
	•	例：trade_actions_2026-01-12.csv

trade_actions_*.csv の必須カラム
runner は 最低限次を参照：
	•	theme
	•	symbol
	•	action（ENTER/WATCH/SKIP/EXIT）
	•	action_ts_jst（推奨。無い場合は “今日扱い” の誤差が出る）
	•	status（任意だが、あれば obsolete/edited を除外する）
	•	decision_id（任意）
	•	snapshot_id（任意）
	•	threshold_used（任意：Step6由来）
	•	rules_applied（任意：Step6由来）
	•	score_adjusted（任意：Step6由来）

pipeline/step7_log.py の既存仕様（statusやupdated_from_ts_jstの扱い）と矛盾しないように、runner側も “obsolete/editedは除外” を踏襲する。

⸻

2.3（任意）Step10設定
	•	パス：config/step10.yaml（任意。無ければデフォルト）
	•	通知対象（ENTERのみ）
	•	逸脱の閾値（WARN/CRITICAL）
	•	watchlist出力フォーマット等

※最小実装では不要。runner 引数で代替可。

⸻

3) 出力（Output）

3.1 日次サマリ（必須：UI/通知のソース）
	•	出力ディレクトリ：{OUT}/step10_daily/
	•	ファイル：
	•	summary_{asof_date_utc}.json
	•	summary_latest.json（同内容のコピー）

summary_*.json スキーマ（提案：step10_summary_v1）
	•	schema_version: "step10_summary_v1"
	•	generated_at_utc: string
	•	asof_date_utc: string（Step6の値をそのまま）
	•	decision_path: string（読み取った decision_latest.json のパス）
	•	trades_dir: string（step7_tradesのパス）
	•	risk_mode:
	•	mode: string
	•	strength: number
	•	tradable_themes: list[str]
	•	counts:
	•	enter: int
	•	watch: int
	•	avoid: int
	•	enter_candidates: list[object]
	•	symbol, theme, score_total, score_adjusted, env, trend, flags, threshold_used, rules_applied
	•	no_trade:
	•	is_no_trade: bool（enter_candidatesが0ならtrue）
	•	reason: string（例："ENTER candidates = 0" or risk_off等）
	•	decision_digest（idempotency用）:
	•	decision_hash: string（picks.ENTERのsymbol一覧＋asof_date等で作る）
	•	enter_symbols: list[str]

⸻

3.2 逸脱検知（必須）
	•	deviation_{asof_date_utc}.json
	•	deviation_latest.json

deviation_*.json スキーマ（提案：step10_deviation_v1）
•	schema_version: "step10_deviation_v2"
	•	generated_at_utc
	•	asof_date_utc
	•	window_days: int（例：7）
	•	decision_enter_symbols: list[str]
	•	trade_enter_symbols_today: list[str]
	•	deviations_today: list[object]
	•	type: string（例："enter_when_no_trade", "enter_not_in_enter_candidates", "enter_on_avoid"）
	•	symbol
	•	theme
	•	action_ts_jst
	•	details: dict（原因、decision側bucketなど）
	•	counts:
	•	deviation_today: int
	•	deviation_7d: int
	•	warning_level: "OK"|"WARN"|"CRITICAL"
•	warning_reason: array（例：["deviation_7d>=2"]）

⸻

3.3 アラート送信ログ（必須：append-only / idempotent）
	•	alerts.jsonl（固定名、append-only）
	•	パス：{OUT}/step10_daily/alerts.jsonl

1行1イベント（JSONL）：
	•	ts_utc
	•	asof_date_utc
	•	event_id（例："{asof_date_utc}:{decision_hash}"）
	•	kind: "ENTER_ALERT"|"NO_TRADE_NOTICE"|"DEVIATION_WARN" 等
	•	payload（通知に使った本文やenter一覧）
	•	result（送信した/しない、理由、送信先など）

runner は event_id が既に alerts.jsonl に存在する場合は再送しない（完全な二重防止）。

⸻

3.4 TradingView向け watchlist（任意）
	•	tradingview_watchlist_{asof_date_utc}.txt
	•	内容：ENTER候補の symbol を1行1銘柄（重複除去）
	•	tradingview_watchlist_latest.txt

⸻

4) runner の引数仕様（CLI）

最小想定：
	•	--out（必須）: out ディレクトリ（例：out）
	•	--window-days（任意）: 逸脱集計窓（default 7）
	•	--no-notify（任意）: 通知処理をしない（ログ生成はする）
	•	--emit-watchlist（任意）: TradingView watchlist を出す
	•	--dry-run（任意）: 書き込みはする/しない選択（必要なら）

⸻

5) 正規化ルール（重要：既存実装と整合）

5.1 decision 側の “ENTER”
	•	Step6 の picks["ENTER"] を “ENTER候補” の唯一ソースとする
（BUY判定などは見ない。bucketを信頼）

5.2 trade_actions の除外（Step7踏襲）
	•	status があれば：
	•	obsolete, edited は除外
	•	updated_from_ts_jst があり、同一tsのactiveが残るなどの複雑系は、Step7_log.pyの方針を踏襲（可能なら同ロジックで簡易再現）

5.3 “今日”の判定
	•	基準日は asof_date_utc（Step6の値）を使う
	•	trade_actions は action_ts_jst を JST として date を取り、asof_date_utc を JST に変換した日と整合させる（最小は「JST日付で比較」）

⸻

6) 逸脱判定ルール（最小）

deviation type（最低限）
	1.	enter_when_no_trade

	•	decision ENTER が 0 件なのに、trade_actions で ENTER がある

	2.	enter_not_in_enter_candidates

	•	decision ENTERに含まれない symbol が ENTER された

	3.	enter_on_avoid（できれば）

	•	decision AVOID に入っている symbol が ENTER された

warning_level（例：設定可能）
	•	OK：7日で 0〜1
	•	WARN：7日で 2〜3
	•	CRITICAL：7日で 4以上

⸻

7) 実行コマンド例（既存命名に完全一致）
	•	Step6 まで生成済み前提で Step10 runner だけ回す：
	•	python pipeline/step10_daily_runner.py --out out --window-days 7 --emit-watchlist

読み取り：
	•	out/step6_decision/decision_latest.json
	•	out/step7_trades/trade_actions_*.csv

生成：
	•	out/step10_daily/summary_YYYY-MM-DD.json
	•	out/step10_daily/deviation_YYYY-MM-DD.json
	•	out/step10_daily/alerts.jsonl
	•	（任意）out/step10_daily/tradingview_watchlist_YYYY-MM-DD.txt

⸻


1) 追加する最小実装（1ファイル）

新規ファイル
	•	pipeline/step10_daily_runner.py

依存
	•	標準ライブラリ＋pandasのみで良い（pandas無しでも可能だが CSV が楽）
	•	既存の pipeline/common.py の ensure_dir を使う（あれば）

⸻

2) Runner の入出力（既存命名に完全一致）

入力（固定）
	•	decision:
	•	out/step6_decision/decision_latest.json
	•	trades:
	•	out/step7_trades/trade_actions_*.csv

出力（固定）
	•	out/step10_daily/summary_<asof_date_utc>.json
	•	out/step10_daily/summary_latest.json
	•	out/step10_daily/deviation_<asof_date_utc>.json
	•	out/step10_daily/deviation_latest.json
	•	out/step10_daily/alerts.jsonl（append-only）

⸻

3) Runner CLI（落ちない最小）

コマンド
	•	python pipeline/step10_daily_runner.py --out out

引数（最小）
	•	--out（必須）：out など

（任意で後から）
	•	--decision（default: out/step6_decision/decision_latest.json）
	•	--trades-glob（default: out/step7_trades/trade_actions_*.csv）
	•	--dry-run（書き込みなし）

⸻

4) 必須データ最低ライン（確定）

decision_latest.json から読む最小
	•	picks.ENTER[*].symbol のみ必須
	•	asof_date は fallback で作る（後述）
	•	risk_mode は無くても "unknown" で継続

trade_actions_*.csv から読む最小
	•	symbol と action のみ必須
	•	status がある場合は obsolete/edited を除外
	•	action_ts_jst が無ければ today 扱い（最小仕様）

⸻

5) fallback ルール（落ちないための確定）

asof_date_utc の決定順
	1.	decision.asof_date_utc
	2.	decision.asof_utc を date 化
	3.	decision.generated_at_utc を date 化
	4.	それも無ければ now_utc().date()

decision が無い / 壊れている
	•	decision_status = "missing_or_invalid"
	•	enter_candidates_symbols = empty
	•	no_trade = true / reason=decision missing
	•	deviation は warning_level="UNKNOWN" で必ず出す

trades が無い / CSV が読めない
	•	読めたファイルだけ採用
	•	全部無理なら trade_enter_symbols_today = empty

“today” の判定
	•	基本：asof_date_utc を runner の “当日” とみなす（UTC基準で固定）
	•	trade側：
	•	action_ts_jst があれば JST日付にして比較
	•	無ければ runner 当日扱い（最小仕様）

⸻

6) deviation 判定（確定）

A: enter_when_no_trade

条件：
	•	decision ENTER が 0
	•	かつ trade ENTER が当日 > 0

B: enter_not_in_enter_candidates

条件：
	•	decision ENTER が > 0
	•	かつ（trade ENTER 当日 − decision ENTER）に要素がある

warning_level
	•	decision missing/invalid：UNKNOWN
	•	deviations が 0：OK
	•	deviations が 1以上：WARN

⸻

7) 出力フォーマット（実際のJSONフィールドを確定）

7.1 summary_latest.json / summary_.json

最低フィールド（これで固定）：
	•	schema_version: "step10_summary_v1"
	•	generated_at_utc: 例 "2026-01-12T16:30:00Z"
	•	asof_date_utc: "2026-01-09"
	•	decision_status: "ok" or "missing_or_invalid"
	•	risk_mode: "RISK_ON"|"RISK_OFF"|"NEUTRAL"|"unknown"
	•	enter_candidates:
	•	count: int
	•	symbols: list[str]
	•	no_trade:
	•	is_no_trade: bool
	•	reason: "no enter candidates" / "decision missing"
	•	paths:
	•	decision_latest_json: string
	•	trades_glob: string
	•	output_dir: string

任意（あれば入れる、無ければ省略でOK）：
	•	enter_candidates_detail（symbol→theme/score/threshold_used/rules_applied）

7.2 deviation_latest.json / deviation_.json

最低フィールド（固定）：
•	schema_version: "step10_deviation_v2"
	•	generated_at_utc
	•	asof_date_utc
	•	decision_status
	•	decision_enter_symbols: list[str]
	•	trade_enter_symbols_today: list[str]
	•	deviations_today: list[ {type, symbol, action_ts_jst?, decision_id?} ]
	•	warning_level: "OK"|"WARN"|"UNKNOWN"
•	warning_reason（array）

7.3 alerts.jsonl（append-only）

1行1JSON、最低フィールド（固定）：
	•	ts_utc
	•	asof_date_utc
	•	kind: "RUN_SUMMARY"|"NO_TRADE_NOTICE"|"DEVIATION_WARN"|"ENTER_ALERT"
	•	event_id
	•	dedup_key
	•	payload: dict

dedup_key の最小方針：
	•	RUN_SUMMARY: "{asof_date_utc}:RUN_SUMMARY"
	•	NO_TRADE_NOTICE: "{asof_date_utc}:NO_TRADE_NOTICE"
	•	DEVIATION_WARN: "{asof_date_utc}:DEVIATION_WARN:" + hash(sorted(deviation_symbols))
	•	ENTER_ALERT: "{asof_date_utc}:ENTER_ALERT:" + hash(sorted(enter_symbols))

runner は alerts.jsonl を読み、同一 dedup_key が既にあれば追記しない（最小でもこれで二重送信防止になる）

⸻

8) “今日は何もしない日です” の判定（確定）
	•	is_no_trade = (decision_status != "ok") OR (enter_count == 0)
	•	表示/アラート：
	•	NO_TRADE_NOTICE を 1日1回だけ（dedupで制御）
	•	payload に reason と risk_mode を入れる
