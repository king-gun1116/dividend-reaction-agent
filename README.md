# 배당 리액션 에이전트

## 📖 프로젝트 개요

본 저장소는 배당 공시 이벤트에 대한 반응을 예측하기 위한 파이프라인을 담고 있습니다:

1. **데이터 수집**  
   - 배당 공시 원본(`dividend_ml_ready.csv`)  
   - 종목별 전체 기간 주가(`full_price_history.csv`)  
   - 이벤트별 ±_n_거래일 슬라이스 주가(`price_history.csv`)  
   - 섹터 정보(`sector_info.csv`)  
2. **피처 엔지니어링 & 윈도우 스캔**  
   - 공통 피처 생성(배당 규모, 배당 수익률, 순위, 월, 연말 여부, 섹터 등)  
   - ±_n_거래일 윈도우 보존율 계산  
   - `features_common.csv` 출력  
3. **모듈별 데이터 분할**  
   - 분류(classification.csv)  
   - 회귀(regression.csv)  
   - 군집화(clustering.csv)  
4. **윈도우 최적화**  
   - 이벤트 보존율 vs. 성능(AUC, R², 실루엣) 비교  
5. **모델링 & 평가**  
   - 분류 · 회귀 · 군집화 모델 학습 및 지표 확인

---

## 📂 디렉터리 구조

data/
├── dividend_ml_ready.csv        # 배당 공시 원본
├── full_price_history.csv       # 종목별 전체 주가 (Git LFS)
├── price_history.csv            # 이벤트별 ±윈도우 주가 (Git LFS)
└── sector_info.csv              # 종목→섹터 매핑

module_datasets/
├── features_common.csv          # 공통 피처
├── classification.csv           # 분류용 데이터
├── regression.csv               # 회귀용 데이터
└── clustering.csv               # 군집화용 데이터

notebooks/
├── 02_price_fetching.ipynb      # 주가 수집 및 슬라이스
├── 03_feature_engineering_refactored.ipynb
│     • 윈도우별 보존율 스캔 & features_common.csv
├── 03_features_common.ipynb     # 모듈별 분할
├── 04_classification.ipynb      # 분류 모델링
├── 05_regression.ipynb          # 회귀 모델링
└── 06_clustering.ipynb          # 군집화 모델링

scripts/
└── run_pipeline.py              # 파이프라인 전체 자동화 스크립트

README.md
.gitignore
.gitattributes                    # Git LFS 설정

---

## 🚀 빠른 시작

1. **환경 설치**  
   ```bash
   conda create -n divagent python=3.10
   conda activate divagent
   pip install -r requirements.txt

2.	Git LFS 설정 (대용량 CSV 관리)
git lfs install
git lfs track "data/full_price_history.csv"
git lfs track "data/price_history.csv"
git lfs track "data/html/dividend_with_text.csv"
git lfs track "data/html/dividend_with_text.jsonl"
git add .gitattributes
git commit -m "대용량 파일 Git LFS로 관리 설정"

