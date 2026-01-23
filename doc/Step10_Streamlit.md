Step10 Streamlit（読むだけ）設計

前提：Step10 runner が生成する JSON を読むだけで、判断・通知・ログ更新は一切しない（安全運用）。
	•	入力（固定）
	•	out/step10_daily/summary_latest.json
	•	out/step10_daily/deviation_latest.json
	•	（任意）out/step10_daily/summary_YYYY-MM-DD.json / deviation_YYYY-MM-DD.json を日付選択で参照

⸻

1) UI要件（あなたの指定を仕様化）

共通
	•	wide / 横幅いっぱい
	•	st.set_page_config(layout="wide", page_title="Step10 Daily", initial_sidebar_state="collapsed")
	•	“読むだけ”
	•	書き込み処理なし
	•	外部送信なし
	•	編集UIなし（ボタンは「再読み込み」程度）
	•	エラー耐性
	•	JSONが無い/壊れている場合も画面は落とさず「欠損表示＋原因」を出す

表示の優先順位
	1.	NO TRADE 明示（最上段で固定表示）
	2.	ENTER テーブル（強調・上部）
	3.	逸脱警告（deviation WARN 以上は目立つ）
	4.	WATCH / AVOID（折りたたみ or 2段目）
	5.	参考情報（risk_mode / tradable_themes / decision_hash / counts）

⸻

2) 画面構成（最小で完成する1ページ設計）

A. ヘッダ（1行で状況が分かる）
	•	左：asof_date_utc、generated_at_utc
	•	中：risk_mode.mode と strength
	•	右：decision_hash（短縮表示）／入力パス

B. 状態バナー（固定枠・最重要）

summary の no_trade.is_no_trade により分岐：
	•	NO TRADE の場合
	•	大きく「今日は何もしない日です」
	•	reason を併記（例：ENTER candidates = 0 / decision missing）
	•	“ENTER 0 / WATCH n / AVOID n” を併記
	•	ENTERありの場合
	•	「ENTER候補: N件」
	•	Tradable themes をチップ表示（XME/SMH/XBI）
	•	“今日見るのは ENTERだけ” を明記（WATCHは折りたたみ）

C. 逸脱警告パネル（deviation）
•	warning_level が
	•	OK：緑の小パネル（逸脱なし）
	•	WARN：黄色の警告（deviations_today件数、symbol一覧）
	•	UNKNOWN：灰色（decision missing 等）
	•	deviations_today のテーブル（ある場合のみ）
	•	columns：type, symbol, reason（details.reason）, action_ts_jst（あれば）

⸻

3) テーブル設計（ENTER強調・厳密化）

ENTERテーブル（常に表示・最上段）

データ元：summary_latest.json の enter_candidates
	•	表示カラム（左から）
	1.	symbol（太字）
	2.	theme
	3.	score_total
	4.	score_adjusted
	5.	threshold_used
	6.	rules_applied（短縮）
	7.	flags（短縮）
	•	ソート
	•	デフォルト：score_adjusted desc → score_total desc → symbol asc
	•	score が文字列になっても落ちないよう数値変換できたものだけでソート、失敗は末尾
	•	強調ルール
	•	行全体を強調（背景ハイライト）
	•	score_adjusted が threshold を下回っていたら（基本起きないはずだが）薄赤にして注意
	•	threshold_used が "20D=80.0" のように入るので、右辺数値抽出して比較（抽出失敗時はスキップ）

WATCH / AVOID（折りたたみ）
	•	summary_latest.json には WATCH/AVOID の詳細は入っていない設計なので、2案：

案1（最小）：countsだけ表示
	•	WATCH n / AVOID n の数だけ表示し、詳細は Step6 の decision_latest.json へリンク/参照導線（“Step6を見る”）

案2（推奨）：summary に WATCH/AVOID も追加している場合のみ表示
	•	もし runner を拡張して watch_candidates / avoid_candidates を summary に入れているなら
	•	WATCHは rules_applied や flags が見えるように同様のテーブルを折りたたみで表示

※現状の runner 実装では WATCH/AVOID は summary に入れていないため、Step10ダッシュボードは「ENTER中心」で問題なし。

⸻

4) サイドバー（最小の運用操作だけ）
	•	Outディレクトリ入力（デフォルト out）
	•	日付選択（任意）
	•	summary_latest.json だけでなく summary_YYYY-MM-DD.json を選べる
	•	リロードボタン（再読み込み）
	•	デバッグ情報表示（折りたたみ）
	•	decision_path, trades_dir, decision_status

⸻

5) ファイル欠損/破損時の表示ルール（落ちない）

summary_latest.json が無い
	•	画面上部に「summary_latest.json が見つからない」
	•	ENTER候補は 0 として NO TRADE 扱い
	•	deviation は deviation_latest があれば表示

deviation_latest.json が無い
	•	逸脱パネルを UNKNOWN 扱い（“deviation未生成”）
	•	ENTERテーブルは summary があれば表示

JSONが壊れている
	•	例外を握り、画面に “invalid json” とファイルパスを表示
	•	その領域だけ欠損表示にして画面全体は出す

⸻

6) 実行コマンド（固定）
	•	streamlit run pipeline/step10_dashboard.py -- --out out

（設計上のファイル名）
	•	pipeline/step10_dashboard.py（新規）
	•	依存：pandas（任意、無くても表示可能だが表整形が楽）

⸻

7) 追加で入れると効く（任意だが安全）

クリック導線（外部送信なし）
	•	symbol を TradingView の検索URL（文字列）として表示するだけ（リンククリックはユーザー操作）
	•	“コピー”用に ENTER symbols を1ボックスで表示（watchlistのコピペ用途）

“NO TRADE固定”の強化
	•	NO TRADE のときは ENTER テーブル領域を非表示にして、代わりに大きい宣言＋理由＋risk_mode だけにする（迷い遮断）

⸻

8) 最小実装の受け入れ基準（Done条件）
	1.	summary_latest.json があれば ENTER が表で出る
	2.	no_trade.is_no_trade=true のとき「今日は何もしない日です」が最上段に表示される
3.	deviation.warning_level=WARN なら警告が目立つ
4.	warning_reason（array）を表示
	4.	wideで横幅いっぱい、表が詰まらない
	5.	ファイル欠損/破損でも画面が落ちない

⸻
