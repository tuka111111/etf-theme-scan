以下は Step9（Human-in-the-loop）を“運用として回す”ための具体フロー図です。
目的は「AIの出力を採用する」のではなく、人がルールを確定し、AIをルールに従わせることです。

⸻

Human-in-the-loop 運用フロー図（全体）

【日次運用（毎営業日）】
	1.	Step1〜Step7を実行（通常運用）
　入力：市場データ
　出力：decision_log（Step7）、trade_append（Step7）
　↓
	2.	実トレード（BUYのみ）
　人が最終発注（もしくは半自動）
　↓
	3.	成果データ蓄積（翌日以降）
　prices（Step2）でリターン計測可能になる
　↓
	4.	Step8（Validation）を定期実行（週次推奨）
　decision_log × prices × env/flags を突合し集計
　出力：score_validation.csv（全期間集計）
　↓
	5.	Step9-A（閾値候補の自動算出）
　score_validation.csvから horizon別の候補閾値を出す
　出力：thresholds_final.json（候補）
　↓
	6.	Step9-C（可視化）
　horizon×bucket の avg_return/win_rate/sample を可視化
　閾値・採用領域の見え方を確認
　↓
	7.	人間のルール調整（Step9：Human-in-the-loop）
　「min_score」「flags」「env neutral/bear」等を最終確定
　成果物：config/scoring.yaml / config/rules.yaml
　↓
	8.	Step6/Step7へ反映（次回から適用）
　Step6がyamlを読みBUY/WATCHを決める
　Step7がthreshold_used/rules_appliedをログへ残す
　↓
	9.	ループ継続（ルール変更前後の効果をStep8で検証）

⸻

週次の“ルール更新サイクル”詳細（推奨）

【トリガ】毎週末またはサンプルが一定超えたら実施
A) データ準備
	•	対象期間：直近30営業日〜（運用開始直後は短くてOK）
	•	decision対象：BUYのみ（あなたの要件）

B) Step8（Validation）
	•	集計キー：全期間（asof_dateは併記か除外）
	•	bucket例：score>=80 / 70-79 / 60-69
	•	出力：bucket, avg_return, win_rate, sample（horizon別）

C) Step9-A（候補生成）
	•	ルール：sample>=min_sample（例30）を満たすものを候補に
	•	候補の出し方（例）
	•	avg_return>0 かつ win_rate>=0.55 を満たす最小bucketを採用
	•	満たさない場合は保守的に引き上げる（例80固定）

D) Step9-C（可視化レビュー）
人間のチェック観点（必須）
	•	サンプルが偏っていないか（特定週/特定テーマだけ等）
	•	20Dがまだ薄いなら暫定ルールにする（変更頻度を落とす）
	•	env neutral時に成績が落ちるなら「neutralではBUYしない」等にする
	•	flagsが当たってないなら ignore へ、当たってるなら weight調整へ

E) 人間が最終決定 → YAML更新
	•	scoring.yaml：horizon別 min_score を確定
	•	rules.yaml：
	•	ignore flags を確定
	•	weight flags を確定
	•	env neutral/bear の扱いを確定（skip/weightなど）

F) 次週から適用 + 効果検証
	•	Step6は yaml を読み判定する
	•	Step7は threshold_used/rules_applied を必ず記録
	•	Step8で「変更前後」比較できるようにする（後述）

⸻

“変更前後”を追えるようにする（運用上の要点）

Human-in-the-loop を回すには「いつ何を変えたか」が重要です。

推奨ログ項目（Step6→Step7に残す）
	•	rule_version（例：YYYYMMDD or git commit hash）
	•	threshold_used（horizonと数値）
	•	rules_applied（ignore/weight/env設定の適用結果の要約）
	•	score_adjusted（補正後スコア、任意）

これで Step8 が
	•	rule_version別
	•	threshold別
	•	env別
	•	flags別
で集計でき、改善が本物か確認できます。

⸻

運用の“意思決定ポイント”だけ抜き出し

人が決めるのはこの3つだけに絞るのが実用的です。
	1.	min_score（horizon別）

	•	例：1D=70, 5D=75, 20D=80

	2.	flags方針

	•	ignoreにする（ノイズ）
	•	weight下げ（弱いが意味はある）
	•	weight上げ（強いエッジ）

	3.	env neutral/bear時の扱い

	•	neutralはBUY禁止（skip）
	•	bearはスコアを0.7倍（weight）
など

⸻
