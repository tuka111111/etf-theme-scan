Sniper System – Daily Decision Guide（人間用）
===========================================

目的
----
このドキュメントは Step5 Dashboard / decision_latest.json を人間のトレード意思決定にどう組み込むかを定義する。  
・予測はしない  
・判断の「順序」と「禁止事項」を固定する  
・実行は裁量だが、逸脱を自覚できるようにする

1. 毎日の判断フロー（固定）
-----------------------------
### Step A：全体リスク判定（ETF Daily Env）
最初に必ず確認する  
IF risk_mode == RISK_OFF:  
→ 本日は新規エントリー禁止  
→ 既存ポジション管理のみ  
ELSE: → Step B へ

判断材料:  
・decision_latest.json → risk_mode / strength  
・ETF Daily Env（XME / SMH / XBI）

注意:  
・個別銘柄スコアが高くても ETF Env が bearish なら無視する  
・「今日はやりたい」は判断理由にならない

### Step B：トレード可能テーマの確定
tradable_themes = ETF Daily Env where env == bull  
・decision_latest.md の「Tradable Themes」をそのまま使用  
・ここに含まれないテーマは当日完全無視

### Step C：銘柄分類の扱い方
ENTER（即エントリー候補）  
条件: score >= min_score（例：80）, flags == none, 属するテーマが tradable  
行動: エントリー検討 OK（位置はチャートで決める）  
禁止: 成行で即入る／「高スコアだから」という理由のみで入る

WATCH（監視）  
条件: score 高いが flags あり（例: trend_not_strong_for_signal, vol_too_low, extended_move）  
行動: エントリー禁止、アラート設定・監視のみ、flags が消えるまで待つ

AVOID（回避）  
条件: data_quality_low、ETF Env 不一致、明確なリスクフラグ  
行動: 当日一切触らない

2. エントリー判断の最終責務
----------------------------
このシステムは「銘柄を絞る」までを担当する。  
必ず人間が確認する: 上位足（1H / 4H / Daily）の構造、押し・戻り・ブレイク位置、エントリー後の無効化条件。  
禁止: Step5 のスコアだけでエントリー／「AIが言っているから」。

3. 14日ロールアップの使い方（中期判断）
----------------------------------------
rollup_14d.json の役割: 日々のノイズ除去と「今週・来週の主役テーマ」把握。  
IF theme appears in rollup top consistently (≥ N days): → 重点監視テーマ  
ELSE: → デイ判断のみ  
・デイトレ: daily decision を優先  
・スイング: rollup を重視

4. この decision.md が「しないこと」
-----------------------------------
・売買タイミングの指定  
・利確・損切り価格の提示  
・勝率の保証  
（スナイパーシステムの設計思想上、意図的にやらない）

5. 判断ログの推奨（任意）
-----------------------
毎日 1 行で良いので記録:  
・今日の risk_mode  
・実行 or 見送り  
・decision.md から逸脱したか？  
・逸脱理由（感情 or 論理）  
→ 後からの改善材料になる

最後に（重要）
--------------
このシステムは「考えなくていいことを増やす」ためのもの。  
・見る銘柄を減らす  
・判断順序を固定する  
・感情での例外を可視化する  
勝つことより、間違い方を一定にするための道具である。

必要であれば次のステップとして:  
・decision.md を HTML / Notion 用に整形  
・Streamlit に「Decision View」追加  
・TradingView アラート連携ルール文書化
