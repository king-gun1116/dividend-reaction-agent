"""
run_pipeline.py (rev 2025-07-05b)
──────────────────────────────────────────────────────────────────────────────
End-to-End 배당 공시 Agent 파이프라인

● Step 1  ─ 증분 공시 수집  (utils.dart_api.collect_dividend_filings_incremental)
● Step 2  ─ ML 학습용 CSV 정제   → dividend_ml_ready.csv
● Step 3  ─ 텍스트 임베딩 & FAISS 인덱스 → dividend_faiss_index/

변경점
──────────────────────────────────────────────────────────────────────────────
1. 새 dart_api 로직 적용 (corp_code 캐싱 / document.xml 우선)
2. start_date · end_date 기본값을 최근 3년으로 단축 (속도 개선)
3. max_workers 파라미터 노출 (디폴트 10)
4. 진행 로그를 tqdm + rich.pretty 로 가독성 향상 (선택)
"""

from __future__ import annotations

import os
import pandas as pd
from dotenv import load_dotenv

# 내부 모듈
from utils.dart_api import collect_dividend_filings_incremental
from utils import embed_utils


def run_pipeline(
    start_date: str = "20130101",  
    end_date:   str = "20250630",
    data_dir:   str = "data",
    max_workers: int = 10,
) -> None:
    """배당 공시 Agent 전체 파이프라인"""

    # 0) 환경변수 로드 (DART_API_KEY 등)
    load_dotenv(dotenv_path=".env")
    os.makedirs(data_dir, exist_ok=True)

    # ── 경로 정의 ──────────────────────────────────────────
    csv_path        = os.path.join(data_dir, "dividend_with_text.csv")
    jsonl_path      = os.path.join(data_dir, "dividend_with_text.jsonl")
    ml_ready_path   = os.path.join(data_dir, "dividend_ml_ready.csv")
    faiss_index_dir = os.path.join(data_dir, "dividend_faiss_index")

    # 1️⃣ 증분 수집
    print("\n1️⃣  배당공시 증분 수집 (이미 수집된 건은 건너뜀)")
    new_records = collect_dividend_filings_incremental(
        start=start_date,
        end=end_date,
        save_csv=csv_path,
        save_jsonl=jsonl_path,
        existing_jsonl=jsonl_path,
        max_workers=max_workers,
    )
    print(f"   ▶ 신규 수집 건수: {len(new_records):,} 건")

    # 2️⃣ ML 학습용 CSV 정제
    print("\n2️⃣  ML용 데이터 준비")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # ── 2-1) HTML 원문 제거
    if "html" in df.columns:
        df = df.drop(columns=["html"])

    # ── 2-2) 결측치가 지나치게 많은 컬럼 드롭
    drop_cols = [
        "div_type", "div_kind", "per_share_preferred", "yield_preferred",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── 2-3) 필수 컬럼 결측 삭제
    df = df.dropna(subset=["per_share_common", "total_amount"]).reset_index(drop=True)

    # ── 2-4) 중앙값 대체 (yield_common)
    if "yield_common" in df.columns:
        df["yield_common"] = df["yield_common"].fillna(df["yield_common"].median())

    # ── 2-5) 문자열 → 실수 변환 (콤마 제거)
    numeric_cols = ["per_share_common", "yield_common", "total_amount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace(",", "", regex=False).astype(float)
            )

    df.to_csv(ml_ready_path, index=False, encoding="utf-8-sig")
    print(f"✅  ML용 데이터 저장 → {ml_ready_path}")

    # 3️⃣ 텍스트 임베딩 & FAISS 인덱스
    print("\n3️⃣  Agent 색인 (임베딩 및 FAISS 인덱싱)")
    embed_utils.jsonl_to_faiss(
        jsonl_path=jsonl_path,
        faiss_path=faiss_index_dir,
    )
    print(f"✅  FAISS 인덱스 저장 → {faiss_index_dir}")

    print("\n🎉  전체 파이프라인 완료")


if __name__ == "__main__":
    run_pipeline()