# run_pipeline.py  (rev 2025-07-10)
# ──────────────────────────────────────────────────────────────────────────────
# End-to-End 배당 공시 Agent & ML 파이프라인
#   1. 배당 공시 증분 수집 (DART)
#   2. ML 학습용 정제 → dividend_ml_ready.csv
#   2-1. 주가 수집 + ±30일 윈도우 검증
#   3. 공통 피처 생성 & 모듈별 분할 (classification / regression / clustering)
#   4. 문서 임베딩 & FAISS 인덱스 구축
#   5. Notebook 기반 모델 학습 (04~06)  ⎯ papermill 실행
#   6. 앙상블 & Master CSV 생성 (07_ensemble.ipynb or inline function)
#   7. 추가 후처리 노트북(08_dividend.ipynb) – 선택적 실행
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from typing import List

import pandas as pd
import papermill as pm
from dotenv import load_dotenv

# ── 내부 유틸
from utils.dart_api import collect_dividend_filings_incremental
from utils.data_cleaning import clean_ml_data
from utils.price_fetcher import run_price_fetching
from utils import embed_utils

# ──────────────────────────────────────────────────────────────────────────────
# Helper: 앙상블 & Master CSV 빌더
# ──────────────────────────────────────────────────────────────────────────────

def _build_master_csv(
    module_dir: str,
    data_dir: str,
    master_csv_path: str,
    n_clusters: int = 4,
) -> None:
    """classificationㆍregression 결과를 통합하여 Master CSV 생성

    Parameters
    ----------
    module_dir       : str  – module_datasets 디렉토리 (classification / regression csv 위치)
    data_dir         : str  – 프로젝트 최상위 data 디렉토리
    master_csv_path  : str  – 최종 저장 경로
    n_clusters       : int  – K-Means 클러스터 개수 (default=4)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import joblib

    # ── 파일 경로
    reg_data_fp  = os.path.join(module_dir, "regression_enriched.csv")
    clf_data_fp  = os.path.join(module_dir, "classification_with_text.csv")
    pred_fp      = os.path.join(
        data_dir,
        "results",
        "regression",
        "regression_predictions_for_ensemble.csv",
    )
    clf_model_fp = os.path.join(data_dir, "models", "lgbm_classifier.pkl")

    # ── 데이터 로드
    df_reg  = pd.read_csv(reg_data_fp, parse_dates=["rcept_dt"], dtype={"stock_code": str})
    df_clf  = pd.read_csv(clf_data_fp, parse_dates=["rcept_dt"], dtype={"stock_code": str})
    df_pred = pd.read_csv(pred_fp,   parse_dates=["rcept_dt"], dtype={"stock_code": str})

    # ── 분류 확률(p_up) 계산
    clf_model = joblib.load(clf_model_fp)
    X_clf     = df_clf.drop(columns=[
        "up_1d", "corp_name", "stock_code", "rcept_dt"], errors="ignore"
    )
    df_clf["p_up"] = clf_model.predict_proba(X_clf)[:, 1]

    # ── 회귀 residual + y_pred 를 이용한 클러스터 라벨 생성
    scaler        = StandardScaler()
    scaled        = scaler.fit_transform(df_pred[["y_pred", "residual"]])
    df_pred["cluster"] = (
        KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        .fit_predict(scaled)
        .astype("int8")
    )

    # ── 마스터 병합
    df_master = (
        df_pred
        .merge(
            df_clf[["stock_code", "rcept_dt", "p_up"]],
            on=["stock_code", "rcept_dt"],
            how="left",
        )
        .merge(
            df_reg,
            on=["stock_code", "rcept_dt"],
            how="left",
            suffixes=("", "_orig"),
        )
    )

    # ── 저장
    df_master.to_csv(master_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Master CSV saved → {master_csv_path}  (rows: {len(df_master):,})")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline function
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    start_date: str = "20130101",
    end_date:   str = datetime.today().strftime("%Y%m%d"),
    data_dir:   str = "data",
    max_workers: int = 10,
    skip_notebooks: List[str] | None = None,
) -> None:
    """배당 공시 Agent 전체 파이프라인

    Parameters
    ----------
    start_date     : str   – DART 조회 시작일 (YYYYMMDD)
    end_date       : str   – DART 조회 종료일 (YYYYMMDD)
    data_dir       : str   – 프로젝트 데이터 루트
    max_workers    : int   – 멀티스레드 병렬 수집 워커 수
    skip_notebooks : list  – 실행을 건너뛰고 싶은 노트북 파일명 목록 (optional)
    """

    skip_notebooks = skip_notebooks or []

    # 0. env & dirs
    load_dotenv(dotenv_path=".env")
    os.makedirs(data_dir, exist_ok=True)

    csv_path      = os.path.join(data_dir, "dividend_with_text.csv")
    jsonl_path    = os.path.join(data_dir, "dividend_with_text.jsonl")
    ml_ready_path = os.path.join(data_dir, "dividend_ml_ready.csv")
    hist_path     = os.path.join(data_dir, "price_history.csv")
    check_path    = os.path.join(data_dir, "window_check_result.csv")
    cache_dir     = os.path.join(data_dir, "price_cache")
    module_dir    = os.path.join(data_dir, "module_datasets")
    results_dir   = os.path.join(data_dir, "results")
    artifacts_dir = os.path.join("artifacts")

    for d in [module_dir, results_dir, artifacts_dir, cache_dir]:
        os.makedirs(d, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # 1. 증분 공시 수집
    # ──────────────────────────────────────────────────────────────
    print("\n1⃣  배당공시 증분 수집")
    new_records = collect_dividend_filings_incremental(
        start=start_date,
        end=end_date,
        save_csv=csv_path,
        save_jsonl=jsonl_path,
        existing_jsonl=jsonl_path,
        max_workers=max_workers,
    )
    print(f"   ▶ 신규 수집 건수: {len(new_records):,}건")

    # ──────────────────────────────────────────────────────────────
    # 2. ML 데이터 정제 → dividend_ml_ready.csv
    # ──────────────────────────────────────────────────────────────
    print("\n2⃣  ML용 데이터 정제 & 저장")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"   원본 shape: {df.shape}")
    df = clean_ml_data(df)
    print(f"   정제 후 shape: {df.shape}")
    df.to_csv(ml_ready_path, index=False, encoding="utf-8-sig")
    print(f"   ✅ 저장 완료 → {ml_ready_path}")

    # 2-1. Price fetch & window check
    print("\n2.1 주가 수집 & 윈도우 검증")
    run_price_fetching(
        div_path       = ml_ready_path,
        hist_path      = hist_path,
        check_path     = check_path,
        cache_dir_path = cache_dir,
        window_days    = 30,
        max_workers    = max_workers,
    )

    # ──────────────────────────────────────────────────────────────
    # 3. Feature Engineering (03_feature_splits.ipynb)
    # ──────────────────────────────────────────────────────────────
    print("\n3⃣  공통 피처 생성 & 데이터 분할 (papermill)")
    try:
        pm.execute_notebook(
            input_path  = os.path.join("notebooks", "03_feature_splits.ipynb"),
            output_path = os.path.join(artifacts_dir, "03_feature_splits.out.ipynb"),
            parameters  = {
                "data_dir":   data_dir,
                "out_dir":    module_dir,
                "clf_window":      1,
                "reg_window":     10,
                "cluster_window": 10,
            },
            kernel_name=None,
        )
    except Exception:
        print("   ⚠️  03_feature_splits.ipynb 실행 실패 — 스택트레이스 출력")
        traceback.print_exc()
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────
    # 4. Embedding & FAISS
    # ──────────────────────────────────────────────────────────────
    print("\n4⃣  문서 임베딩 & FAISS 색인")
    embed_utils.jsonl_to_faiss(
        jsonl_path=jsonl_path,
        faiss_path=os.path.join(data_dir, "dividend_faiss_index"),
    )
    print("   ✅ FAISS 인덱스 저장 완료")

    # ──────────────────────────────────────────────────────────────
    # 5. Notebook-based model training (04-06)
    # ──────────────────────────────────────────────────────────────
    nb_seq = [
        "04_classification.ipynb",
        "05_regression.ipynb",
        "06_clustering.ipynb",
    ]

    for nb in nb_seq:
        if nb in skip_notebooks:
            print(f"   ⏭️  {nb}  건너뜀 (skip_notebooks 설정) ")
            continue
        print(f"\n5⃣  papermill 실행 → {nb}")
        try:
            pm.execute_notebook(
                input_path  = os.path.join("notebooks", nb),
                output_path = os.path.join(artifacts_dir, nb.replace(".ipynb", ".out.ipynb")),
                parameters  = {
                    "data_dir":  data_dir,
                    "module_dir": module_dir,
                },
            )
        except FileNotFoundError:
            print(f"   ⚠️  {nb} 파일이 존재하지 않습니다 — 건너뜀")
        except Exception:
            print(f"   ⚠️  {nb} 실행 오류 — 계속 진행")
            traceback.print_exc()

    # ──────────────────────────────────────────────────────────────
    # 6. Ensemble & Master CSV
    # ──────────────────────────────────────────────────────────────
    master_csv_path = os.path.join(data_dir, "all_stocks_master.csv")

    if "07_ensemble.ipynb" not in skip_notebooks and os.path.exists(os.path.join("notebooks", "07_ensemble.ipynb")):
        print("\n6⃣  papermill 실행 → 07_ensemble.ipynb")
        try:
            pm.execute_notebook(
                input_path  = os.path.join("notebooks", "07_ensemble.ipynb"),
                output_path = os.path.join(artifacts_dir, "07_ensemble.out.ipynb"),
                parameters  = {
                    "module_dir": module_dir,
                    "data_dir":   data_dir,
                    "master_csv": master_csv_path,
                },
            )
        except Exception:
            print("   ⚠️  07_ensemble.ipynb 실행 실패 — 코드 fallback 으로 전환")
            traceback.print_exc()
            _build_master_csv(module_dir, data_dir, master_csv_path)
    else:
        # 노트북 스킵 또는 없음 → inline 함수 이용
        print("\n6⃣  Ensemble 노트북 건너뜀 — inline 함수로 Master CSV 생성")
        _build_master_csv(module_dir, data_dir, master_csv_path)

    # ──────────────────────────────────────────────────────────────
    # 7. Optional: 08_dividend.ipynb 후처리
    # ──────────────────────────────────────────────────────────────
    nb08 = "08_dividend.ipynb"
    if nb08 not in skip_notebooks and os.path.exists(os.path.join("notebooks", nb08)):
        print("\n7⃣  papermill 실행 → 08_dividend.ipynb")
        try:
            pm.execute_notebook(
                input_path  = os.path.join("notebooks", nb08),
                output_path = os.path.join(artifacts_dir, nb08.replace(".ipynb", ".out.ipynb")),
                parameters  = {
                    "data_dir": data_dir,
                    "master_csv": master_csv_path,
                },
            )
        except Exception:
            print("   ⚠️  08_dividend.ipynb 실행 오류 — 계속 진행")
            traceback.print_exc()
    else:
        print("   ⏭️  08_dividend.ipynb  건너뜀")

    print("\n🎉  전체 파이프라인 완료!")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 예시: python run_pipeline.py 20130101 20250630 data 8 04_classification.ipynb
    import argparse

    parser = argparse.ArgumentParser(description="Dividend Agent End-to-End Pipeline")
    parser.add_argument("--start",  type=str, default="20130101", help="시작일 (YYYYMMDD)")
    parser.add_argument("--end",    type=str, default=datetime.today().strftime("%Y%m%d"), help="종료일 (YYYYMMDD)")
    parser.add_argument("--data",   type=str, default="data", help="데이터 디렉토리")
    parser.add_argument("--workers",type=int, default=10, help="max_workers")
    parser.add_argument("--skip",   nargs="*", default=[], help="건너뛸 노트북 파일명 목록")
    args = parser.parse_args()

    run_pipeline(
        start_date=args.start,
        end_date=args.end,
        data_dir=args.data,
        max_workers=args.workers,
        skip_notebooks=args.skip,
    )