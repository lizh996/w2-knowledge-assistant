# -*- coding: utf-8 -*-
"""
错误分析: 找出评测集漏检题 + 输出详情用于分类
用法: D:\\an\\envs\\mineru\\python.exe error_analysis.py
"""
import json, os, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
FT_PATH = os.path.join(BASE, "output_ft_v1")
CHUNKS = json.load(open(os.path.join(BASE, "chunks_all.json"), encoding="utf-8"))
EVAL = json.load(open(os.path.join(BASE, "eval_retrieval_set.json"), encoding="utf-8"))
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
base = AutoModel.from_pretrained(MODEL_PATH)
model = PeftModel.from_pretrained(base, FT_PATH).to(device).eval()
print(f"模型: 微调版 (LoRA) | 设备: {device}")

def encode(model, tok, texts, batch=32):
    vecs = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i+batch], max_length=512, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            h = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).last_hidden_state
        vecs.append(F.normalize(h[:, 0], p=2, dim=-1).cpu())
    return torch.cat(vecs)

if __name__ == "__main__":
    chunk_texts = [c["text"] for c in CHUNKS]
    chunk_vecs = encode(model, tok, chunk_texts)
    q_vecs = encode(model, tok, [e["question"] for e in EVAL])
    sim = q_vecs @ chunk_vecs.T
    gold_idx = {c["chunk_id"]: i for i, c in enumerate(CHUNKS)}

    misses = []
    for qi, e in enumerate(EVAL):
        gi = gold_idx.get(e["gold_chunk_id"], -1)
        if gi < 0: continue
        ranks = sim[qi].argsort(descending=True).tolist()
        rank = ranks.index(gi) + 1
        if rank > 5:  # 漏检
            top5 = [(r+1, CHUNKS[r]["page"], round(sim[qi][r].item(), 4), CHUNKS[r]["text"][:80])
                    for r in ranks[:5]]
            misses.append({
                "question": e["question"],
                "gold_chunk_id": e["gold_chunk_id"],
                "gold_page": e["page"],
                "gold_sim": round(sim[qi][gi].item(), 4),
                "gold_text": CHUNKS[gi]["text"][:120],
                "rank": rank,
                "top5": top5,
            })

    print(f"\n=== 错误分析: {len(misses)}/350 漏检 ===")
    for m in misses:
        print(f"\nQ: {m['question'][:50]}")
        print(f"  gold: rank={m['rank']} sim={m['gold_sim']} p{m['gold_page']} | {m['gold_text'][:60]}...")
        print(f"  top5: {[(r, p, s) for r, p, s, _ in m['top5']]}")
    json.dump(misses, open(os.path.join(BASE, "error_analysis.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n✅ 详情已保存 error_analysis.json")
