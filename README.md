# 💰 Dividend Reaction Agent

미래에셋 AI 페스티벌 TECH 부문 참가 프로젝트  
**배당 공시 → 주가 반응**을 자동으로 예측하는 Agent 시스템입니다.

---

## 📌 주요 기능

1. **DART API 기반 배당 공시 수집** (병렬 처리 + 증분 업데이트)
2. **본문 HTML 파싱 및 배당 데이터 전처리**
3. **ML 학습용 CSV + FAISS 임베딩 색인 자동 생성**
4. **HyperCLOVA Agent 검색 대응을 위한 인덱스 구축**

---

## 📂 프로젝트 구조

```bash
├── run_pipeline.py              # 전체 파이프라인 실행 스크립트
├── data/                        # 데이터 저장 디렉토리
│   ├── dividend_with_text.csv
│   ├── dividend_ml_ready.csv
│   └── ...
├── utils/                       # 기능별 모듈화 코드
│   ├── dart_api.py              # 배당 공시 수집 + 전처리
│   └── embed_utils.py          # 임베딩 및 FAISS 색인
├── notebooks/                   # 실험용 노트북
├── results/                     # 결과물 저장 폴더
└── requirements.txt             # 의존성 패키지