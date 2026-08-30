# -*- coding: utf-8 -*-
"""逐题检索分析：找出清洗版 MRR 失分点"""
import os, json, sys
from pathlib import Path

proj = Path(r"C:\Users\lizhihao\w1-day5\device-rag-44653")
sys.path.insert(0, str(proj / "backend" / "src"))
BGE = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
os.environ["BGE_M3_MODEL"] = BGE

from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer(BGE)
client = chromadb.PersistentClient(path=str(proj / "data" / "chroma"))
col = client.get_collection("sf6_kb_clean")

rs = json.load(open(proj / "eval" / "retrieval_set.json", encoding="utf-8"))
pos = [q for q in rs if not q.get("should_refuse")]

hits_list = []
for i, q in enumerate(pos):
    qv = model.encode([q["question"]], normalize_embeddings=True)
    hits = col.query(query_embeddings=qv.tolist(), n_results=5)
    docs, metas = hits["documents"][0], hits["metadatas"][0]
    ep = q["expected_page"]
    rank = None
    for j, (doc, meta) in enumerate(zip(docs, metas)):
        if abs(meta.get("page", 0) - ep) <= 1:
            rank = j + 1
            break
    hits_list.append((rank, q))

mrr = sum(1.0 / r for r, _ in hits_list if r) / len(hits_list)
print(f"MRR = {mrr:.4f} ({len(hits_list)} 条)")
print()
print("=== 非首位命中的题（MRR 失分点） ===")
for r, q in hits_list:
    if r != 1:
        print(f"\nrank={r} | 期望p{q['expected_page']} | {q['question']}")
        # top-3 实际命中
        qv = model.encode([q["question"]], normalize_embeddings=True)
        hits = col.query(query_embeddings=qv.tolist(), n_results=3)
        for j, (doc, meta) in enumerate(zip(hits["documents"][0], hits["metadatas"][0])):
            print(f"    {j+1}. p{meta.get('page')} [{meta.get('section','')[:20]}] {doc[:50]}...")
