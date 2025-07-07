"""
run_pipeline.py (rev 2025-07-06g)
──────────────────────────────────────────────────────────────────────────────
End-to-End 배당 공시 Agent 파이프라인
  • 증분 수집
  • 日 중복 공시 통합(최신 접수번호 기준)
  • ML용 데이터 정제 (clean_ml_data 호출)
  • 주가 수집 & ±10거래일 윈도우 검증
  • 임베딩 & FAISS 색인
"""

from __future__ import annotations
import os
import pandas as pd
from dotenv import load_dotenv

# 내부 모듈
from utils.dart_api import collect_dividend_filings_incremental
from utils.data_cleaning import clean_ml_data
from utils.price_fetcher import run_price_fetching  # ✅ NEW
from utils import embed_utils


def run_pipeline(
    start_date: str = "20130101",
    end_date:   str = "20250630",
    data_dir:   str = "data",
    max_workers: int = 10,
) -> None:
    """배당 공시 Agent 전체 파이프라인"""

    # 0️. 환경변수 로드
    load_dotenv(dotenv_path=".env")
    os.makedirs(data_dir, exist_ok=True)

    # ── 주요 경로 정의
    csv_path      = os.path.join(data_dir, "dividend_with_text.csv")
    jsonl_path    = os.path.join(data_dir, "dividend_with_text.jsonl")
    ml_ready_path = os.path.join(data_dir, "dividend_ml_ready.csv")
    faiss_dir     = os.path.join(data_dir, "dividend_faiss_index")
    hist_path     = os.path.join(data_dir, "price_history.csv")
    check_path    = os.path.join(data_dir, "window_check_result.csv")
    cache_dir     = os.path.join(data_dir, "price_cache")

    # 1️. 증분 수집
    print("\n 1. 배당공시 증분 수집 (이미 수집된 건은 건너뜀)")
    new_records = collect_dividend_filings_incremental(
        start=start_date,
        end=end_date,
        save_csv=csv_path,
        save_jsonl=jsonl_path,
        existing_jsonl=jsonl_path,
        max_workers=max_workers,
    )
    print(f"▶ 신규 수집 건수: {len(new_records):,}건")

    # 2️. ML용 데이터 정제
    print("\n 2. ML용 데이터 준비 및 정제")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"원본 shape: {df.shape}")
    df = clean_ml_data(df)
    print(f"정제 후 shape: {df.shape}")
    df.to_csv(ml_ready_path, index=False, encoding="utf-8-sig")
    print(f"✅ ML용 데이터 저장 → {ml_ready_path}")

    # 2.1 주가 수집 & 윈도우 검증
    print("\n 2.1 주가 수집 & 윈도우 검증")
    run_price_fetching(
        div_path=ml_ready_path,
        hist_path=hist_path,
        check_path=check_path,
        cache_dir_path=cache_dir,
        window_days=30,
        max_workers=max_workers,
    )

    # 3️. 임베딩 & FAISS 인덱싱
    print("\n 3. Agent 색인 (임베딩 및 FAISS 인덱싱)")
    embed_utils.jsonl_to_faiss(
        jsonl_path=jsonl_path,
        faiss_path=faiss_dir,
    )
    print(f"✅ FAISS 인덱스 저장 → {faiss_dir}")

    print("\n🎉 전체 파이프라인 완료")


if __name__ == "__main__":
    run_pipeline()
