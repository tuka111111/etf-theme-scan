"""
VanEck holdings reader (SMH only)

提供API（このファイルが提供する関数）:
- read_vaneck_holdings_csv(path) -> List[str]
- read_vaneck_holdings_xlsx(path) -> List[str]

入力:
- ローカル CSV/XLSX（SMHの holdings）

出力:
- 正規化済み ticker のリスト（A-Z0-9.-）

やらないこと:
- ダウンロード
- HTML解析
- スコア計算
"""