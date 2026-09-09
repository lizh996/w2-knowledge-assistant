# -*- coding: utf-8 -*-
"""共享评测工具：加载 base / ft(E1) 模型（FlagEmbedding BGEM3FlagModel），
统一 dense / sparse 编码；sparse 打分对齐 Qdrant local mode 的 IDF 归一化；
RRF(k=60) 融合。供 eval_channels / eval_unseen / eval_ood 共用。

关键口径（与线上 backend/src/sf6_rag/retrieve.py 对齐）：
- dense：BGE-M3 CLS 池化 + L2 归一化 → 余弦（点积）
- sparse：模型 lexical_weights（token_id: weight），查询侧做 IDF 加权
          idf = ln((n - df + 0.5) / (df + 0.5) + 1)   # Qdrant local mode _compute_idf
- fusion：RRF，k=60（对齐 retrieve.py _RRF_K = 60）
- ft(E1)：LoRA 只训 query/value attention，sparse_linear 未训，故 ft 的 sparse 走
          merge 后的主干 + base 原始 sparse_linear（与线上部署模型口径一致）
"""
import os
import json
import math
import shutil

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
FT_ADAPTER = os.path.join(BASE_DIR, "output_ft_v1")
MERGED_FT = os.path.join(BASE_DIR, "merged_ft_v1")
RRF_K = 60


def ensure_merged_ft():
    """把 LoRA adapter 合并进 base 主干，补齐 sparse/colbert linear（复用 base 权重），
    产出可直接被 BGEM3FlagModel 加载的完整目录（幂等）。"""
    marker = os.path.join(MERGED_FT, "adapter_config.json")
    if os.path.exists(marker):
        # 已合并：校验 sparse_linear.pt 存在
        if os.path.exists(os.path.join(MERGED_FT, "sparse_linear.pt")):
            return MERGED_FT
    from transformers import AutoTokenizer, AutoModel
    from peft import PeftModel

    print(f"[merge] 合并 LoRA {os.path.basename(FT_ADAPTER)} → {MERGED_FT} ...")
    base = AutoModel.from_pretrained(BASE_MODEL)
    ft = PeftModel.from_pretrained(base, FT_ADAPTER)
    merged = ft.merge_and_unload()
    os.makedirs(MERGED_FT, exist_ok=True)
    merged.save_pretrained(MERGED_FT)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.save_pretrained(MERGED_FT)
    # sparse_linear / colbert_linear：LoRA 未训练，复用 base 原始权重
    for f in ("sparse_linear.pt", "colbert_linear.pt"):
        shutil.copy(os.path.join(BASE_MODEL, f), os.path.join(MERGED_FT, f))
    print(f"[merge] 完成 → {MERGED_FT}")
    return MERGED_FT


def load_model(mode):
    """mode: 'base' | 'ft'，返回 BGEM3FlagModel。"""
    from FlagEmbedding import BGEM3FlagModel
    path = ensure_merged_ft() if mode == "ft" else BASE_MODEL
    return BGEM3FlagModel(path, use_fp16=False)


def encode(model, texts, batch_size=16):
    """返回 (dense[N,1024] np.float32, sparse list[dict{token_id: weight}])。"""
    out = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        batch_size=batch_size,
    )
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    sparse = out["lexical_weights"]
    return dense, sparse


def compute_idf(sparse_list):
    """按候选库 sparse 统计每个 token 的 idf（对齐 Qdrant local mode）。"""
    n = len(sparse_list)
    df = {}
    for sw in sparse_list:
        for tok in sw.keys():
            df[tok] = df.get(tok, 0) + 1
    idf = {}
    for tok, d in df.items():
        idf[tok] = math.log((n - d + 0.5) / (d + 0.5) + 1)
    return idf


def dense_scores(q_dense, c_dense):
    """dense 余弦相似度矩阵 [Q, C]（dense 已 L2 归一化 → 点积即余弦）。"""
    return q_dense @ c_dense.T


def sparse_scores(q_sparse, c_sparse, idf):
    """sparse 检索分 [Q, C]：score = Σ_tok (q_w[tok]*idf[tok]) * c_w[tok]。"""
    Q, C = len(q_sparse), len(c_sparse)
    scores = np.zeros((Q, C), dtype=np.float32)
    for qi, qw in enumerate(q_sparse):
        # 只对 query 侧做 IDF 加权
        weighted = {tok: w * idf.get(tok, 0.0) for tok, w in qw.items()}
        for ci, cw in enumerate(c_sparse):
            s = 0.0
            # 遍历较小的一侧，减少内积开销
            if len(weighted) <= len(cw):
                for tok, w in weighted.items():
                    if tok in cw:
                        s += w * cw[tok]
            else:
                for tok, w in cw.items():
                    if tok in weighted:
                        s += weighted[tok] * w
            scores[qi, ci] = s
    return scores


def ranks_from_scores(scores):
    """分数矩阵 → 排名矩阵（1-based，分数越高 rank 越小）。"""
    order = np.argsort(-scores, axis=1)  # 每行降序
    ranks = np.empty_like(order, dtype=np.int64)
    for i in range(order.shape[0]):
        ranks[i, order[i]] = np.arange(1, order.shape[1] + 1)
    return ranks


def rrf_scores(dense_rank, sparse_rank, k=RRF_K):
    """RRF 融合分 [Q, C]：1/(k+dense_rank) + 1/(k+sparse_rank)。"""
    return 1.0 / (k + dense_rank) + 1.0 / (k + sparse_rank)


def recall_at(scores, gold_idx_list, k):
    """scores [Q, C]；gold_idx_list 为每条 query 的 gold chunk 列索引。
    返回 recall@k（gold 落入 top-k 的比例）。"""
    hits = 0
    n = 0
    for qi, gi in enumerate(gold_idx_list):
        if gi < 0:
            continue
        n += 1
        topk = np.argsort(-scores[qi])[:k]
        if gi in topk:
            hits += 1
    return hits / n if n else 0.0, hits, n


def mrr_at(scores, gold_idx_list, k=10):
    """MRR@k：gold 排名的倒数均值（rank>k 计 0）。"""
    total = 0.0
    n = 0
    for qi, gi in enumerate(gold_idx_list):
        if gi < 0:
            continue
        n += 1
        order = np.argsort(-scores[qi])
        pos = int(np.where(order == gi)[0][0]) + 1
        if pos <= k:
            total += 1.0 / pos
    return total / n if n else 0.0
