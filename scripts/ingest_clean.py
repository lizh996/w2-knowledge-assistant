# -*- coding: utf-8 -*-
"""清洗版 chunks 入库：chunks.json -> sf6_kb_clean 集合
用法: PYTHONPATH=backend D:/an/envs/langchain/python.exe scripts/ingest_clean.py
"""
from __future__ import annotations
import os, json, sys
from pathlib import Path

OUT = Path(r"C:\Users\lizhihao\w1-day5\device-rag-44653\data\mineru_out")
CHROMA = r"C:\Users\lizhihao\w1-day5\device-rag-44653\data\chroma"
BGE = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
SOURCE = "GB/T 44653-2024 \u516d\u6c1f\u5316\u786b\u6c14\u4f53\u73b0\u573a\u5faa\u73af\u518d\u5229\u7528\u5bfc\u5219"

chunks = json.load(open(OUT / "chunks.json", encoding="utf-8"))
print(f"\u8bfb\u53d6 chunks: {len(chunks)} \u4e2a")

os.environ["BGE_M3_MODEL"] = BGE
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer(BGE)
client = chromadb.PersistentClient(path=CHROMA)
# 重建集合（确保删除旧数据，避免 add 覆盖不了旧 id）
try:
    client.delete_collection("sf6_kb_clean")
except Exception:
    pass
col = client.get_or_create_collection("sf6_kb_clean", metadata={"hnsw:space": "cosine"})

texts = [c["content"] for c in chunks]
vectors = model.encode(texts, normalize_embeddings=True, batch_size=16).tolist()
metadatas = [{"page": c.get("page", 0), "section": c["section"], "doc": SOURCE} for c in chunks]
ids = [f"gb44653-clean-{i:03d}" for i in range(len(chunks))]

col.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
print(f"\u2705 \u5165\u5e93\u5b8c\u6210: sf6_kb_clean = {col.count()} chunks")
