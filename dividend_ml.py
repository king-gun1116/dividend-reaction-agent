# fix_missing_dividend.py

import pandas as pd
from utils.dart_api import parse_dividend_info

def safe_parse(html: str) -> dict:
    """
    HTML 을 parse_dividend_info 로 파싱하되,
    누락 필드는 '-' 로 채워서 반환
    """
    try:
        info = parse_dividend_info(html or "")
    except Exception:
        info = {}
    fields = [
        "div_type","div_kind",
        "per_share_common","per_share_preferred",
        "yield_common","yield_preferred",
        "total_amount","record_date",
        "payment_date","meeting_held",
        "meeting_date","board_decision_date"
    ]
    return {f: info.get(f, "-") for f in fields}

# 1) 원본 불러오기
no_html   = pd.read_csv("data/dividend_no_html.csv", dtype=str)
with_html = pd.read_csv("data/dividend_with_text.csv", dtype=str)

# 2) rcept_no별로 HTML 파싱 결과 모으기
parsed_rows = []
for _, row in with_html[["rcept_no","html"]].drop_duplicates("rcept_no").iterrows():
    parsed = safe_parse(row["html"])
    parsed["rcept_no"] = row["rcept_no"]
    parsed_rows.append(parsed)

parsed_df = pd.DataFrame(parsed_rows)

# 3) 머지 & 빈값(‘-’)만 덮어쓰기
df = no_html.merge(parsed_df, on="rcept_no", how="left", suffixes=("","_new"))

# ML 에 최종 사용할 컬럼 (html 제외)
ml_cols = [
    "corp_name","stock_code","rcept_dt","report_nm","rcept_no",
    "div_type","div_kind",
    "per_share_common","per_share_preferred",
    "yield_common","yield_preferred",
    "total_amount","record_date",
    "payment_date","meeting_held",
    "meeting_date","board_decision_date",
]

# 기존 컬럼 값이 없거나 '-' 이면 새로 파싱한 값으로 채우기
for fld in ml_cols:
    new = fld + "_new"
    if new in df.columns:
        df[fld] = df.apply(
            lambda r: r[new] if ( (r.get(fld) in [None,"-"]) and r[new] not in [None,"-"] )
                      else r.get(fld),
            axis=1
        )

# 보조 new 컬럼 제거
to_drop = [c for c in df.columns if c.endswith("_new")]
df = df.drop(columns=to_drop)

# 4) 최종 저장
df[ml_cols].to_csv("data/dividend_final_for_ml.csv", index=False, encoding="utf-8-sig")
print("✅ dividend_final_for_ml.csv 생성 완료")