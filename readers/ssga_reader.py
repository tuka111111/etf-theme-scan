"""
SSGA ETF holdings reader

責務:
- SSGA提供ETFの構成銘柄ファイルを安全に読み取る
- 対象ETF: XME, XBI

入力:
- SSGA公式 CSV / XLSX

出力:
- pandas.DataFrame（symbol 列のみ）

やらないこと:
- yfinance 呼び出し
- テーマ判定
- スコアリング

注意:
- 注意書き行・フッター・文章行は除外する
- ticker / symbol 列を自動検出する
"""