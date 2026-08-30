
# -*- coding: utf-8 -*-
"""V1 基线评测：检索集 recall@5 / MRR（page 级 ±1 判定）。"""
import json, os, sys
sys.path.insert(0, r"C:\Users\lizhihao\w2-knowledge-assistant\backend\src")
os.environ.setdefault("SF6_RAG_RUNTIME_DIR", r"C:\Users\lizhihao\w2-knowledge-assistant\data")

from sf6_rag.retrieve import retrieve, search_collection_names
from sf6_rag.pipeline import estimate_tokens

def norm_page(p):
    try: return int(p)
    except: return None

def hit_page(question, gold_pages, top_k=5):
    """检索 top5，看是否命中金标准页码（±1 容差）。返回 (命中, 首个命中排名, 命中页列表)"""
    try:
        results = retrieve(question, top_k=top_k)
    except Exception as e:
        return False, None, f"ERR:{e}"
    if not results:
        return False, None, []
    hit_pages = [norm_page(r.get("page")) for r in results]
    golds = set(gold_pages)
    for rank, p in enumerate(hit_pages, 1):
        if p is not None and any(abs(p - g) <= 1 for g in golds):
            return True, rank, hit_pages
    return False, None, hit_pages

# 载入检索集
with open(r"C:\Users\lizhihao\w2-knowledge-assistant\eval\retrieval_set.json", encoding="utf-8") as f:
    retrieval_set = json.load(f)

print(f"集合: {search_collection_names()}")
print(f"评测集: {len(retrieval_set)} 题\n")

hits = 0
rr_sum = 0.0
for i, item in enumerate(retrieval_set, 1):
    ok, rank, pages = hit_page(item["question"], item["pages"])
    gold = item["pages"]
    # 简化显示：只显示前5命中页
    pages_str = ",".join(str(p) for p in pages[:5]) if isinstance(pages, list) else pages
    mark = "✅" if ok else "❌"
    print(f"{mark} #{i} 金标p{gold} 命中p[{pages_str}] rank={rank}")
    if ok:
        hits += 1
        rr_sum += 1.0 / rank

recall = hits / len(retrieval_set)
mrr = rr_sum / len(retrieval_set)
print(f"\n=== V1 基线 ===")
print(f"recall@5 = {recall:.4f} ({hits}/{len(retrieval_set)})")
print(f"MRR     = {mrr:.4f}")
