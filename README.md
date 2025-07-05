## 📈 배당 공시 기반 주가 반응 예측 Agent

이 프로젝트는 배당 공시를 입력받아 요약하고, 과거 유사 사례를 탐색하며, 공시 이후의 주가 반응(상승/하락/유지)을 예측하는 자동화 Agent를 구축하는 것입니다.

## 🚀 실행 방법

다음 명령어로 전체 전처리 및 학습 파이프라인을 실행할 수 있습니다:

```bash
python run_pipeline.py

🗂 폴더 구조
dividend_reaction_agent/
├── data/                  # 원본 및 처리된 데이터
├── notebooks/             # 실험용 Jupyter 파일
├── models/                # 학습된 모델
├── utils/                 # 기능 함수 모듈
├── run_pipeline.py        # 전체 실행 스크립트
├── requirements.txt       # 필수 패키지
└── README.md              # 프로젝트 설명서

🧠 주요 기술 스택
 • 📄 DART API로 배당 공시 수집
 • 📊 yfinance로 주가 데이터 수집
 • 🤖 LSTM 기반 주가 반응 예측
 • 🧠 HyperCLOVA 기반 요약 및 정책 분류 (외부)
 • 🧾 Flask 기반 Agent API 서버


🔗 평가용 API 형식

http
GET /agent?question=거래량이 전날 대비 15% 이상 오른 종목을 모두 보여줘

json
{
  "answer": "아난티, 엠에프엠코리아입니다."
}


## ✅ 2. `requirements.txt` (패키지 목록)

```txt
pandas
numpy
yfinance
requests
scikit-learn
tensorflow
flask
beautifulsoup4
lxml
tqdm
