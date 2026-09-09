# -*- coding: utf-8 -*-
"""完整评测（对齐讲师 V2.0 §8/§9）：264 干净集上 base vs E1
指标：Recall@5/10/20、MRR@10、nDCG@10、Top1 + 逐条命中（Saved/Degraded 四象限）
用法: D:\\an\\envs\\mineru\\python.exe eval_full.py [base|ft] [ft_dir]
输出: eval_full_{mode}.json + pos 记录（供四象限对比）
"""
import json, os, sys, math, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
FT_PATH = os.path.join(BASE, sys.argv[2]) if len(sys.argv) > 2 else os.path.join(BASE, "output_ft_v1")
CHUNKS = json.load(open(os.path.join(BASE, "chunks_all.json"), encoding="utf-8"))
EVAL_RAW = json.load(open(os.path.join(BASE, "eval_retrieval_set.json"), encoding="utf-8"))
EVAL = [e for e in EVAL_RAW if e.get("source") != "train-query"]  # 干净集 264
device = "cuda" if torch.cuda.is_available() else "cpu"

def load(mode):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    base = AutoModel.from_pretrained(MODEL_PATH)
    if mode == "ft":
        model = PeftModel.from_pretrained(base, FT_PATH)
    else:
        model = base
    return tok, model.to(device).eval()

def encode(model, tok, texts, batch=32):
    vecs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i+batch], max_length=512, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            h = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).last_hidden_state
        vecs.append(F.normalize(h[:, 0], p=2, dim=-1).cpu())
    return torch.cat(vecs)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    tok, model = load(mode)
    chunk_vecs = encode(model, tok, [c["text"] for c in CHUNKS])
    q_vecs = encode(model, tok, [e["question"] for e in EVAL])
    sim = q_vecs @ chunk_vecs.T

    gold_idx = {c["chunk_id"]: i for i, c in enumerate(CHUNKS)}
    hits = {5: 0, 10: 0, 20: 0}
    mrr_sum = 0.0; ndcg_sum = 0.0; top1 = 0
    pos_records = {}
    for qi, e in enumerate(EVAL):
        gi = gold_idx.get(e["gold_chunk_id"], -1)
        if gi < 0:
            continue
        ranks = sim[qi].argsort(descending=True).tolist()
        pos = ranks.index(gi) + 1
        pos_records[qi] = pos
        for k in (5, 10, 20):
            if pos <= k:
                hits[k] += 1
        if pos <= 10:
            mrr_sum += 1.0 / pos
            ndcg_sum += 1.0 / math.log2(pos + 1)   # rel=1 的 nDCG@10
        if pos == 1:
            top1 += 1
    n = len(EVAL)
    res = {"mode": mode, "adapter": os.path.basename(FT_PATH) if mode == "ft" else "base",
           "clean_n": n,
           "recall@5": hits[5]/n, "recall@10": hits[10]/n, "recall@20": hits[20]/n,
           "mrr@10": mrr_sum/n, "ndcg@10": ndcg_sum/n, "top1": top1/n}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    out = os.path.join(BASE, f"eval_full_{mode}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(pos_records, open(os.path.join(BASE, f"eval_full_pos_{mode}.json"), "w", encoding="utf-8"))
    print(f"✅ {out}")
