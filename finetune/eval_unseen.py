# -*- coding: utf-8 -*-
"""任务2（评测部分）：Unseen Knowledge 子集 —— 46131 未参与训练，验证 FT 未破坏泛化。
候选库 = 88（chunks_all）+ 61（chunks_46131）= 149 块。
gold 全部为 46131 块（未见知识）。base vs ft(E1) 各跑 recall@5 / MRR@10（dense 检索）。

问题来源：eval_unseen_46131.json（DeepSeek 生成，含 gold_chunk_id）。
若该文件为空/缺失，回退规则生成：取 chunks_46131 的 question_kwd 每块 1 条。

用法: D:\\an\\envs\\mineru\\python.exe eval_unseen.py
输出: eval_unseen_result.json
"""
import os
import sys
import json
import random

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _eval_common import (
    BASE_DIR, load_model, encode, dense_scores, recall_at, mrr_at,
)

CHUNKS_88 = json.load(open(os.path.join(BASE_DIR, "chunks_all.json"), encoding="utf-8"))
CHUNKS_61 = json.load(open(os.path.join(BASE_DIR, "chunks_46131.json"), encoding="utf-8"))
EVAL_FILE = os.path.join(BASE_DIR, "eval_unseen_46131.json")


def load_questions():
    """优先读 LLM 生成的评测集；空则规则回退（每块 question_kwd 取 1 条）。"""
    if os.path.exists(EVAL_FILE):
        qs = json.load(open(EVAL_FILE, encoding="utf-8"))
        if qs:
            return qs, "llm"
    random.seed(42)
    qs = []
    for c in CHUNKS_61:
        kwds = c.get("question_kwd") or []
        if not kwds:
            continue
        kwds = [k for k in kwds if k and not k.startswith(c["source"])]
        if not kwds:
            kwds = [f"{c['source']} 关于 {c['semantic_type']} 的要求是什么？"]
        qs.append({"question": kwds[0], "gold_chunk_id": c["chunk_id"],
                   "page": c["page"], "source": c["source"],
                   "semantic_type": c.get("semantic_type", "")})
    return qs, "rule-fallback"


if __name__ == "__main__":
    questions, src = load_questions()
    print(f"Unseen 评测集: {len(questions)} 条（来源={src}）")

    # 候选库：88 + 61，gold 均为 46131 块
    all_chunks = CHUNKS_88 + CHUNKS_61
    chunk_texts = [c["text"] for c in all_chunks]
    gold_idx = {c["chunk_id"]: i for i, c in enumerate(all_chunks)}
    gold_list = [gold_idx.get(q["gold_chunk_id"], -1) for q in questions]
    n_missing = sum(1 for g in gold_list if g < 0)
    q_texts = [q["question"] for q in questions]

    print(f"候选库 {len(all_chunks)} 块（88 训练域 + 61 未见），gold 缺失 {n_missing} 条")
    if n_missing:
        print("[WARN] 部分 gold_chunk_id 不在候选库中，将跳过计分")

    result = {"unseen_n": len(questions), "source": src,
              "candidate_chunks": len(all_chunks), "results": {}}

    for mode in ("base", "ft"):
        print(f"\n=== [{mode}] 编码 {len(chunk_texts)} 块 + {len(q_texts)} 条 query ===")
        model = load_model(mode)
        c_dense, _ = encode(model, chunk_texts)
        q_dense, _ = encode(model, q_texts)
        scores = dense_scores(q_dense, c_dense)

        r5, h5, n5 = recall_at(scores, gold_list, 5)
        mrr = mrr_at(scores, gold_list, 10)
        result["results"][mode] = {
            "recall@5": round(r5, 4),
            "mrr@10": round(mrr, 4),
            "hits@5": h5,
            "valid_n": n5,
        }
        print(f"  recall@5 = {r5:.4f} ({h5}/{n5})")
        print(f"  mrr@10   = {mrr:.4f}")

    # 结论：ft 在未见知识上是否不显著低于 base。
    # 小样本(n≈28)下单条即 3.6pp，故用「命中题数差」而非比例差判定：
    # 差 ≤1 题 + MRR 降幅 <2pp → 视为噪声，未破坏泛化。
    r5_base = result["results"]["base"]["recall@5"]
    r5_ft = result["results"]["ft"]["recall@5"]
    h5_base = result["results"]["base"]["hits@5"]
    h5_ft = result["results"]["ft"]["hits@5"]
    mrr_base = result["results"]["base"]["mrr@10"]
    mrr_ft = result["results"]["ft"]["mrr@10"]
    result["delta"] = {
        "recall@5": round(r5_ft - r5_base, 4),
        "mrr@10": round(mrr_ft - mrr_base, 4),
        "hits@5": h5_ft - h5_base,
    }
    result["conclusion"] = (
        "FT 未破坏泛化（未见知识召回基本持平，差异在单样本噪声内）"
        if (h5_ft - h5_base >= -1 and mrr_ft >= mrr_base - 0.02)
        else "FT 在未见知识上明显退化，需复查"
    )

    out = os.path.join(BASE_DIR, "eval_unseen_result.json")
    json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[OK] 结果写入 {out}")
    print(json.dumps(result, ensure_ascii=False, indent=1))
