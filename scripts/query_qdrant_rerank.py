# -*- coding: utf-8 -*-
"""Day5：Qdrant 检索 → bge-reranker 粗排（两段式）。

检索：Qdrant FusionQuery(RRF) 取 top-k 候选
重排：FlagReranker(bge-reranker-v2-m3) 对 (query, chunk_text) 打分 → 取 top-N

用法:
    D:/an/envs/langchain/python.exe scripts/query_qdrant_rerank.py
"""
from __future__ import annotations
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(r"C:\Users\lizhihao\w1-day5\device-rag-44653")
QDRANT_DIR = BASE / "data" / "qdrant"
BGE = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
RERANKER = r"D:\models\bge-reranker-v2-m3"
COLLECTION = "device_knowledge_v7"
TOP_K = 20      # 检索候选数
TOP_N = 5       # 重排后返回数


def to_sparse_vector(weights: dict):
    from qdrant_client.models import SparseVector
    items = sorted(weights.items(), key=lambda kv: int(kv[0]))
    return SparseVector(indices=[int(k) for k, _ in items],
                        values=[float(v) for _, v in items])


def main() -> None:
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
    from qdrant_client import QdrantClient
    from qdrant_client.models import Prefetch, FusionQuery, Fusion

    model = BGEM3FlagModel(BGE, use_fp16=False)
    reranker = FlagReranker(RERANKER, use_fp16=False)
    client = QdrantClient(path=str(QDRANT_DIR))

    samples = [
        "SF6气体回收后检测不合格应该怎么处理？",
        "SF6气体现场循环再利用的流程是什么？",
        "回充前要对电气设备做什么处理？",
    ]

    for q in samples:
        out = model.encode([q], return_dense=True, return_sparse=True,
                           return_colbert_vecs=False, batch_size=1)
        dvec = out["dense_vecs"][0].tolist()
        svec = to_sparse_vector(out["lexical_weights"][0])

        resp = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(query=svec, using="sparse", limit=TOP_K),
                Prefetch(query=dvec, using="dense", limit=TOP_K),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=TOP_K,
        )

        # 候选 (query, chunk_text) 对，交给 reranker 打分
        pairs = [(q, h.payload["text"]) for h in resp.points]
        scores = reranker.compute_score(pairs, batch_size=8)

        # 按 rerank 分降序重排
        ranked = sorted(zip(resp.points, scores), key=lambda x: -x[1])
        print(f"【{q}】")
        for i, (h, s) in enumerate(ranked[:TOP_N], 1):
            p = h.payload
            print(f"  {i}. p{p['page']} [{p['semantic_type']}] {p['section'][:24]} | rerank分={s:.4f}")
        print()


if __name__ == "__main__":
    main()
