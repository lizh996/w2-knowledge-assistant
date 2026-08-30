# -*- coding: utf-8 -*-
"""chunks_v7_final.json 向量化入库 -> sf6_kb_v7（dense + sparse）。

对齐《向量化方案》Day4：
- 嵌入模型 BGE-M3（BGEM3FlagModel，dense 1024 维 + sparse lexical_weights）
- 编码文本 = content + important_kwd + question_kwd（Q2Q 参与向量化）
- dense 归一化入库 ChromaDB（cosine）
- sparse lexical_weights 存 json，供 Day4 复合检索加载（ChromaDB 不支持稀疏向量）
- documents 只存纯 content；元数据存 page/type/section/semantic_type/document_id/ingest_version

用法:
    D:/an/envs/langchain/python.exe scripts/ingest_v7.py
"""
from __future__ import annotations
import os
import json
from pathlib import Path

BASE = r"C:\Users\lizhihao\w1-day5\device-rag-44653"
CHUNKS = Path(BASE) / "data" / "mineru_out" / "chunks_v7_final.json"
SPARSE_OUT = Path(BASE) / "data" / "mineru_out" / "sparse_weights.json"
CHROMA = Path(BASE) / "data" / "chroma"
BGE = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
COLLECTION = "sf6_kb_v7"


def build_embed_text(c: dict) -> str:
    """编码文本 = content + important_kwd + question_kwd（Q2Q 参与向量化）。"""
    parts = [c["content"]]
    ik = " ".join(c.get("important_kwd", []))
    qk = " ".join(c.get("question_kwd", []))
    if ik:
        parts.append(ik)
    if qk:
        parts.append(qk)
    return "\n".join(parts)


def main() -> None:
    chunks = json.load(open(CHUNKS, encoding="utf-8"))
    print(f"读取 chunks_v7_final.json: {len(chunks)} 块")

    from FlagEmbedding import BGEM3FlagModel
    import numpy as np
    import chromadb

    print("加载 BGE-M3（BGEM3FlagModel，CPU 推理）...")
    model = BGEM3FlagModel(BGE, use_fp16=False)

    embed_texts = [build_embed_text(c) for c in chunks]
    print(f"编码 {len(embed_texts)} 条（dense + sparse）...")
    out = model.encode(
        embed_texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        batch_size=16,
    )
    dense = out["dense_vecs"]            # (n, 1024)
    sparse = out["lexical_weights"]      # list[dict{token_id: weight}]

    # dense 归一化（BGEM3FlagModel 已归一，此处双保险），转 list 供 ChromaDB
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    dense = (dense / np.maximum(norms, 1e-12)).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA))
    try:
        client.delete_collection(COLLECTION)
        print(f"已删除旧集合 {COLLECTION}")
    except Exception:
        pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids = [f"gb44653-v7-{i:03d}" for i in range(len(chunks))]
    documents = [c["content"] for c in chunks]
    metadatas = [
        {
            "page": c["page"],
            "type": c["type"],
            "section": c["section"],
            "semantic_type": c["semantic_type"],
            "document_id": c.get("document_id", "gb_t_44653_2024"),
            "ingest_version": c.get("ingest_version", "20260825_v1"),
        }
        for c in chunks
    ]

    col.add(ids=ids, embeddings=dense, documents=documents, metadatas=metadatas)

    # sparse 存 json（ChromaDB 不支持稀疏向量，Day4 复合检索时加载）
    sparse_data = {
        ids[i]: {str(k): float(v) for k, v in sparse[i].items()}
        for i in range(len(chunks))
    }
    SPARSE_OUT.write_text(json.dumps(sparse_data, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] 入库完成: {COLLECTION} = {col.count()} 块")
    print(f"dense 维度: {len(dense[0])} | sparse 存至: {SPARSE_OUT.name}")


if __name__ == "__main__":
    main()
