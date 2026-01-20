前提（あなたの流れに合わせる）：
	•	Step6 は「BUY（ENTER）判定」を出す本番判断
	•	Step8 は「スコアの妥当性検証」（統計）
	•	Step9 は「人間が閾値・flags・env扱いを確定する調整」

⸻

実行タイミングの推奨（運用として固定）

Step6（毎日）
	•	毎営業日 1回（原則）
	•	タイミング：その日の意思決定に使う価格が揃った直後
	•	1D運用なら：前日終値が確定した後（日本時間の朝〜午前）に固定が無難
	•	目的：今日の ENTER/WATCH/AVOID を出して、Step10で確認する

例（毎日）
	1.	Step1〜5（更新）
	2.	Step6（decision生成）
	3.	Step7（ログ/トレード追記）
	4.	Step10（summary/deviation/dashboard）

⸻

Step8（週次〜隔週）
	•	毎日は不要
	•	推奨：週1回（例：土曜/日曜）または 隔週
	•	目的：直近の decision_log / trade_actions と prices を突合して
	•	score_bucket別の平均リターン/勝率
	•	flagsの有効性
	•	env別の差
を更新する

条件
	•	“統計”なのでサンプルが増えてから意味が出る
→ 最低でも数回〜数十回の decision が溜まってから週次で十分

⸻

Step9（Step8の結果が出た後だけ）
	•	Step8を見た後にだけ実行（調整フェーズ）
	•	推奨：月1回、早くても 隔週
	•	目的：人間が最終決定する
	•	min_score（horizon別）
	•	flags 無視/重み
	•	env neutral/bear の扱い
	•	成果物：config/scoring.yaml / config/rules.yaml の更新（＝ルール確定）

重要
	•	Step9は頻繁に回すと “過剰最適化” になりやすい
→ 変更は小さく、頻度は低く

⸻

まとめ（最短の運用ループ）
	•	毎日：Step6 → Step7 → Step10（見る）
	•	週1：Step8（検証CSV更新）
	•	月1（または隔週）：Step9（ルール更新）→ 次の日から Step6に反映

⸻

LaunchAgentに組み込むべき範囲
	•	自動化は基本 Step6/7/10 まで（日次判断の固定化）
	•	Step8/9 は 手動実行（週次・月次のレビュータイムに実施）
