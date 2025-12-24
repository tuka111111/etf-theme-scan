# readers/

このディレクトリは外部ETFホールディングス仕様の変更を
アプリケーション本体から隔離するためのもの。

- 入力: CSV / XLSX / ローカルファイル
- 出力: pandas.DataFrame（symbol列のみ）
- やらないこと:
  - スコア計算
  - テーマ判定
  - yfinance / API 呼び出し責務コメント：readers/README.md ＋ 各ファイル冒頭