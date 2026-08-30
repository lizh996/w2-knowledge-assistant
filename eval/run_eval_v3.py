
# -*- coding: utf-8 -*-
"""V3 评测：rerank + 查询改写（环境变量 RERANK_ENABLED=1 QUERY_REWRITE_ENABLED=1）。"""
import json, os, sys
sys.path.insert(0, r"C:\Users\lizhihao\w2-knowledge-assistant\backend\src")
os.environ.setdefault("SF6_RAG_RUNTIME_DIR", r"C:\Users\lizhihao\w2-knowledge-assistant\data")
os.environ["RERANK_ENABLED"] = "1"
os.environ["QUERY_REWRITE_ENABLED"] = "1"

from sf6_rag.retrieve import retrieve, search_collection_names, _get_reranker
from sf6_rag.api import _rewrite_query

def norm_page(p):
    try: return int(p)
    except: return None

def hit_page_v3(question, gold_pages, top_k=5):
    try:
        # 查询改写 → 检索 → 精排
        q = _rewrite_query(question)
        results = retrieve(q, top_k=15)
        reranker = _get_reranker()
        if reranker is not None and results:
            pairs = [(q, c["content"]) for c in results]
            scores = reranker.compute_score(pairs, batch_size=8)
            order = sorted(range(len(results)), key=lambda i: -float(scores[i]))
            results = [results[i] for i in order][:top_k]
        else:
            results = results[:top_k]
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

with open(r"C:\Users\lizhihao\w2-knowledge-assistant\eval\retrieval_set.json", encoding="utf-8") as f:
    retrieval_set = json.load(f)

print("集合:", search_collection_names())
print("reranker:", "已启用" if _get_reranker() is not None else "未加载")
print(f"评测集: {len(retrieval_set)} 题\n")

hits = 0; rr_sum = 0.0
for i, item in enumerate(retrieval_set, 1):
    ok, rank, pages = hit_page_v3(item["question"], item["pages"])
    pages_str = ",".join(str(p) for p in pages[:5]) if isinstance(pages, list) else pages
    mark = "✅" if ok else "❌"
    print(f"{mark} #{i} 金标p{item['pages']} 命中p[{pages_str}] rank={rank}")
    if ok:
        hits += 1
        rr_sum += 1.0/rank

print(f"\n=== V3 结果 ===")
print(f"recall@5 = {hits/len(retrieval_set):.4f} ({hits}/{len(retrieval_set)})")
print(f"MRR     = {rr_sum/len(retrieval_set):.4f}")
