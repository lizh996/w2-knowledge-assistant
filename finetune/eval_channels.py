# -*- coding: utf-8 -*-
"""任务1：Dense / Sparse / RRF-Fusion 分组评测（讲师 V2.0 §8.3）
对 264 干净集，base 与 ft(E1) 各输出三种检索通道的 recall@10：
  - dense-only  ：BGE-M3 CLS 池化 + L2 归一化 → 余弦
  - sparse-only ：lexical_weights + Qdrant local mode IDF 归一化
  - rrf-fusion  ：RRF(k=60) 融合 dense + sparse 排名
候选库：88 块 chunks_all.json（与线上 transformer_kb_v1 一致）
用法: D:\\an\\envs\\mineru\\python.exe eval_channels.py
输出: eval_channels.json（三组 × 两模型 = 6 个 recall@10）
"""
import os
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _eval_common import (
    BASE_DIR, load_model, encode, compute_idf,
    dense_scores, sparse_scores, ranks_from_scores, rrf_scores, recall_at,
)

CHUNKS = json.load(open(os.path.join(BASE_DIR, "chunks_all.json"), encoding="utf-8"))
EVAL_RAW = json.load(open(os.path.join(BASE_DIR, "eval_retrieval_set.json"), encoding="utf-8"))
EVAL = [e for e in EVAL_RAW if e.get("source") != "train-query"]  # 干净集 264

if __name__ == "__main__":
    chunk_texts = [c["text"] for c in CHUNKS]
    q_texts = [e["question"] for e in EVAL]
    gold_idx = {c["chunk_id"]: i for i, c in enumerate(CHUNKS)}
    gold_list = [gold_idx.get(e["gold_chunk_id"], -1) for e in EVAL]
    n_q = len(EVAL)

    result = {"clean_n": n_q, "chunks": len(CHUNKS), "rrf_k": 60, "results": {}}

    for mode in ("base", "ft"):
        print(f"\n=== [{mode}] 编码 {len(chunk_texts)} 块 chunk + {len(q_texts)} 条 query ===")
        model = load_model(mode)
        c_dense, c_sparse = encode(model, chunk_texts)
        q_dense, q_sparse = encode(model, q_texts)

        idf = compute_idf(c_sparse)
        d_scores = dense_scores(q_dense, c_dense)
        s_scores = sparse_scores(q_sparse, c_sparse, idf)

        d_rank = ranks_from_scores(d_scores)
        s_rank = ranks_from_scores(s_scores)
        rrf = rrf_scores(d_rank, s_rank, k=60)

        r_dense, h_dense, _ = recall_at(d_scores, gold_list, 10)
        r_sparse, h_sparse, _ = recall_at(s_scores, gold_list, 10)
        r_fusion, h_fusion, _ = recall_at(rrf, gold_list, 10)

        result["results"][mode] = {
            "dense_recall@10": round(r_dense, 4),
            "sparse_recall@10": round(r_sparse, 4),
            "fusion_recall@10": round(r_fusion, 4),
            "hits": {"dense": h_dense, "sparse": h_sparse, "fusion": h_fusion},
        }
        print(f"  dense  recall@10 = {r_dense:.4f} ({h_dense}/{n_q})")
        print(f"  sparse recall@10 = {r_sparse:.4f} ({h_sparse}/{n_q})")
        print(f"  fusion recall@10 = {r_fusion:.4f} ({h_fusion}/{n_q})")

    out = os.path.join(BASE_DIR, "eval_channels.json")
    json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[OK] 结果写入 {out}")
    print(json.dumps(result, ensure_ascii=False, indent=1))
