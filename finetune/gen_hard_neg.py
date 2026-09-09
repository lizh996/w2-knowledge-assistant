# -*- coding: utf-8 -*-
"""
生成干净训练数据 train_clean.jsonl:
  1) 删除歧义 query (同一 query 对应多个不同 gold，LLM 生成撞车，无法唯一确定正例)
  2) 用 base BGE-M3 检索, 取与 query 语义相近但非 gold 的 chunk 作为 hard negative (替换随机 neg)
  3) pos/neg 统一截断 512 (对齐 P1-7)
用法: D:\\an\\envs\\mineru\\python.exe gen_hard_neg.py
"""
import json, os, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
TRAIN_RAW = os.path.join(BASE, "train.jsonl")
CHUNKS = json.load(open(os.path.join(BASE, "chunks_all.json"), encoding="utf-8"))
OUT = os.path.join(BASE, "train_clean.jsonl")
N_HARD_NEG = 2      # 每条取 top 2 非 gold 块作 hard neg
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def drop_ambiguous(rows):
    """删除同一 query 对应多个不同 gold 的歧义样本。"""
    from collections import defaultdict
    gold_by_q = defaultdict(set)
    for r in rows:
        gold_by_q[r["query"]].add(r["knowledge_id"])
    amb = {q for q, ks in gold_by_q.items() if len(ks) > 1}
    clean = [r for r in rows if r["query"] not in amb]
    return clean, len(amb)


def encode(model, tok, texts, batch=32):
    vecs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i+batch], max_length=512, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            h = model(input_ids=enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device)).last_hidden_state
        vecs.append(F.normalize(h[:, 0], p=2, dim=-1).cpu())
    return torch.cat(vecs)


if __name__ == "__main__":
    rows = load_rows(TRAIN_RAW)
    clean, n_amb = drop_ambiguous(rows)
    print(f"原始 {len(rows)} 条 → 删除 {n_amb} 个歧义 query ({len(rows)-len(clean)} 条记录) → {len(clean)} 条")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model = AutoModel.from_pretrained(MODEL_PATH).to(device).eval()
    print(f"base 模型编码 {len(CHUNKS)} 块 chunk + {len(clean)} 条 query ...")
    chunk_vecs = encode(model, tok, [c["text"] for c in CHUNKS])
    q_vecs = encode(model, tok, [r["query"] for r in clean])
    sim = q_vecs @ chunk_vecs.T  # [n_query, 88]

    gold_idx = {c["chunk_id"]: i for i, c in enumerate(CHUNKS)}
    chunk_texts = {c["chunk_id"]: c["text"][:512] for c in CHUNKS}

    out_rows = []
    for qi, r in enumerate(clean):
        gi = gold_idx.get(r["knowledge_id"], -1)
        # 取非 gold 的 top N 作为 hard neg
        order = sim[qi].argsort(descending=True).tolist()
        neg_ids = [CHUNKS[j]["chunk_id"] for j in order if j != gi][:N_HARD_NEG]
        negs = [chunk_texts[cid] for cid in neg_ids]
        out_rows.append({
            "query": r["query"],
            "pos": [r["pos"][0][:512]],
            "neg": negs,
            "knowledge_id": r["knowledge_id"],
            "question_type": r["question_type"],
        })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] 干净训练数据: {len(out_rows)} 条 -> train_clean.jsonl "
          f"(hard neg: base 模型检索 top {N_HARD_NEG} 非 gold 块)")
