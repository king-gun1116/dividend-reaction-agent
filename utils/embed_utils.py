# utils/embed_utils.py

import os
import json
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 반드시 환경변수로 설정

def jsonl_to_faiss(
    jsonl_path="data/dividend_with_text.jsonl",
    faiss_path="data/dividend_faiss_index"
):
    if not OPENAI_API_KEY:
        print("❌ 환경변수 'OPENAI_API_KEY'가 설정되지 않았습니다.")
        return

    if not os.path.exists(jsonl_path):
        print(f"❌ 파일 없음: {jsonl_path}")
        return

    docs, metadatas = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("report_text"): continue
            docs.append(rec["report_text"])
            metadatas.append({
                "corp_code":  rec.get("corp_code", ""),
                "stock_code": rec.get("stock_code", ""),
                "rcept_dt":   rec.get("rcept_dt", ""),
                "title":      rec.get("report_name", "")
            })

    if len(docs) == 0:
        print("⚠️ 유효한 본문 문서가 없어 임베딩을 건너뜁니다.")
        return

    print(f"🔍 {len(docs)}건 문서 임베딩 시작...")
    embedder = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_texts(docs, embedder, metadatas=metadatas)
    db.save_local(faiss_path)
    print(f"✅ FAISS 인덱스 저장 완료 → {faiss_path}")

if __name__ == "__main__":
    jsonl_to_faiss()