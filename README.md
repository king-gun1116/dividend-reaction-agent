🤖 배당 공시 반응 예측 에이전트 (Dividend Reaction Agent)

DART에서 배당 공시가 올라오면, 해당 공시에 대해 주가가 어떻게 반응할지 예측합니다.
클래식 재무 팩터 + LightGBM 기반 모델 + 임베딩 검색 + 앙상블까지 포함된 배당 예측 통합 시스템입니다.

⸻

✨ 주요 기능 요약

단계	모듈	설명
1	배당 공시 수집기	DART API를 활용하여 ‘현금·현물배당결정’ 공시를 다중 스레드로 증분 수집
2	데이터 정제기	불필요 공시 필터링, 결측값 제거 등 ML 학습용 데이터 생성
3	주가 수집기	±30일 윈도우의 OHLCV 데이터를 FinanceDataReader, yfinance로 수집
4	피처 생성기	분류/회귀/클러스터링에 맞게 공통 피처를 가공 (03_feature_splits.ipynb)
5	ML 모델	04~06번 노트북에서 각각 분류(LGBM), 회귀(LGBM), 클러스터링(KMeans) 수행
6	앙상블 모듈	p_up, y_pred, residual 등을 병합해 Master CSV 생성
7	문서 임베딩	Sentence-Transformer로 벡터화 후 FAISS 인덱스 저장


⸻

📁 폴더 구조

📁 dividend-reaction-agent
├── scripts/          # 전체 파이프라인 실행 스크립트 (run_pipeline.py)
├── notebooks/        # 03~08 모델 학습용 노트북
├── utils/            # dart_api, data_cleaning 등 재사용 함수
├── data/             # 생성되는 데이터 (Git에 포함되지 않음)
├── artifacts/        # 실행 결과 노트북 로그 저장
├── docs/             # 이미지/도표 등 문서
├── requirements.txt  # 패키지 의존성
├── .env.example      # 환경변수 템플릿 (.env 복사용)
└── .gitignore        # 무거운 파일 및 캐시 제외


⸻

🚀 실행 방법

# 1. 클론 & 패키지 설치
$ git clone https://github.com/king-gun1116/dividend-reaction-agent.git
$ cd dividend-reaction-agent
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# 2. 환경변수 설정
$ cp .env.example .env  # DART_API_KEY를 .env에 입력

# 3. 전체 파이프라인 실행
$ python scripts/run_pipeline.py \
    --start 20130101 \
    --end   20250630 \
    --data  data \
    --workers 10


⸻

📦 출력 결과 요약

파일 경로	설명
data/module_datasets/*	분류/회귀/클러스터링용 전처리 완료 데이터셋
data/results/...	예측 결과 (y_pred, residual 등)
all_stocks_master.csv	최종 추천 종목 마스터 파일
data/dividend_faiss_index	공시 본문 임베딩 벡터 검색용 인덱스


⸻

🔐 환경 변수 설정

.env 파일에는 아래 값을 반드시 설정해야 합니다.

변수명	설명
DART_API_KEY	DART Open API 키
OPENAI_API_KEY	(선택) HyperCLOVA 임베딩에 사용

.env.example을 복사해 .env로 만든 후 키만 입력하세요.

⸻

📂 Git 제외 파일 (.gitignore)
	•	data/, artifacts/ 등 무거운 파일은 Git에서 제외 처리됩니다.
	•	필요 시 외부 업로드(Google Drive, S3 등)나 Git LFS를 사용하세요.

⸻

🧑‍💻 개발 환경

# 코드 포맷팅
$ black . && isort .

# 테스트 실행 (선택)
$ pytest


⸻

📄 라이선스

MIT License (자세한 내용은 LICENSE 파일 참조)

⸻

만든 사람: 김건 
키워드: 배당 공시, 주가 반응 예측, 머신러닝, DART API, FAISS