# -*- coding: utf-8 -*-
"""Day4：Qdrant 复合检索验证（FusionQuery RRF 融合 dense + sparse）。

对齐讲师《向量库选型》2.3：FusionQuery(RRF) 一次查 dense+sparse。
含精确词型样例，验证 sparse 路生效（关键词精确命中）。

用法:
    D:/an/envs/langchain/python.exe scripts/query_qdrant_v7.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = r"C:\Users\lizhihao\w1-day5\device-rag-44653"
QDRANT_DIR = Path(BASE) / "data" / "qdrant"
BGE = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
COLLECTION = "device_knowledge_v7"

# 样例：前 3 个精确词型（验证 sparse），后 2 个语义型（验证 dense）
SAMPLES = [
    ("DL/T 920 检测空气和四氟化碳", "精确词型"),
    ("133 Pa 抽真空", "精确词型"),
    ("T/CEC 140 纯度检测", "精确词型"),
    ("SF6气体回收后不合格怎么处理", "语义型"),
    ("六氟化硫的湿度限值是多少", "语义型"),
]


def to_sparse_vector(weights: dict):
    from qdrant_client.models import SparseVector
    items = sorted(weights.items(), key=lambda kv: int(kv[0]))
    return SparseVector(indices=[int(k) for k, _ in items], values=[float(v) for _, v in items])


def main() -> None:
    from FlagEmbedding import BGEM3FlagModel
    from qdrant_client import QdrantClient
    from qdrant_client.models import FusionQuery, Fusion, Prefetch

    model = BGEM3FlagModel(BGE, use_fp16=False)
    client = QdrantClient(path=str(QDRANT_DIR))
    print(f"集合 {COLLECTION} 点数: {client.count(collection_name=COLLECTION).count}\n")

    queries = [q for q, _ in SAMPLES]
    # 逐条 encode（避开 batch padding bug，保证 query 的 sparse 正确）
    dense_q = []
    sparse_q = []
    for q in queries:
        out = model.encode([q], return_dense=True, return_sparse=True,
                           return_colbert_vecs=False, batch_size=1)
        dense_q.append(out["dense_vecs"][0])
        sparse_q.append(out["lexical_weights"][0])

    for (q, kind), dvec, svec in zip(SAMPLES, dense_q, sparse_q):
        resp = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(query=to_sparse_vector(svec), using="sparse", limit=20),
                Prefetch(query=dvec.tolist(), using="dense", limit=20),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=3,
        )
        print(f"【{kind}】{q}")
        print(f"  查询 sparse 词数: {len(svec)}")
        for i, hit in enumerate(resp.points, 1):
            p = hit.payload
            print(f"    {i}. p{p['page']} [{p['semantic_type']}] {p['section'][:26]} | RRF分={hit.score:.4f}")
        print()


if __name__ == "__main__":
    main()
