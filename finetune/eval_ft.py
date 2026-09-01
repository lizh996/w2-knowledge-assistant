# -*- coding: utf-8 -*-
"""
评测: E0(Base BGE-M3) vs E1(LoRA微调) — 350条评测集 recall@5 / MRR
用法: D:\\an\\envs\\mineru\\python.exe eval_ft.py [base|ft]
"""
import json, os, sys, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
FT_PATH = os.path.join(BASE, "output_ft_v1")
CHUNKS = json.load(open(os.path.join(BASE, "chunks_all.json"), encoding="utf-8"))
EVAL = json.load(open(os.path.join(BASE, "eval_retrieval_set.json"), encoding="utf-8"))
device = "cuda" if torch.cuda.is_available() else "cpu"

def load(mode):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    base = AutoModel.from_pretrained(MODEL_PATH)
    if mode == "ft":
        model = PeftModel.from_pretrained(base, FT_PATH)  # base + LoRA
        print(f"[{mode}] 已加载 LoRA adapter: {FT_PATH}")
    else:
        model = base
        print(f"[{mode}] 已加载 Base BGE-M3")
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
    chunk_texts = [c["text"] for c in CHUNKS]
    q_texts = [e["question"] for e in EVAL]
    print(f"编码 {len(chunk_texts)} 块 chunk + {len(q_texts)} 条 query...")
    chunk_vecs = encode(model, tok, chunk_texts)
    q_vecs = encode(model, tok, q_texts)
    sim = q_vecs @ chunk_vecs.T  # [350, 88]

    gold_idx = {c["chunk_id"]: i for i, c in enumerate(CHUNKS)}
    hits5 = 0; mrr_sum = 0.0; top1 = 0
    for qi, e in enumerate(EVAL):
        gi = gold_idx.get(e["gold_chunk_id"], -1)
        if gi < 0: continue
        ranks = sim[qi].argsort(descending=True).tolist()
        pos = ranks.index(gi) + 1
        if pos <= 5: hits5 += 1
        if pos <= 10: mrr_sum += 1.0 / pos
        if pos == 1: top1 += 1
    n = len(EVAL)
    print(f"\n=== {mode.upper()} 评测结果 ({n} 条) ===")
    print(f"Recall@5  = {hits5/n:.4f}  ({hits5}/{n})")
    print(f"MRR@10    = {mrr_sum/n:.4f}")
    print(f"Top1      = {top1/n:.4f}  ({top1}/{n})")
    json.dump({"mode": mode, "recall@5": hits5/n, "mrr@10": mrr_sum/n, "top1": top1/n},
              open(os.path.join(BASE, f"eval_{mode}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
