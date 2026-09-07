# -*- coding: utf-8 -*-
"""三版本链路消融评测：同一 FT 模型 + 当前库(149块)，只变检索链路配置。
V1=dense-only top5 | V2=RRF(dense+sparse) top5 | V3=RRF粗排15+rerank取5"""
import json, os, sys, time
sys.path.insert(0, r"C:\\Users\\lizhihao\\w2-knowledge-assistant\\backend\\src")
os.environ.setdefault("SF6_RAG_RUNTIME_DIR", r"C:\\Users\\lizhihao\\w2-knowledge-assistant\\data")
os.environ["RERANK_ENABLED"] = "1"

from sf6_rag.retrieve import _encode_query, _get_client, search_collection_names, _get_reranker, retrieve

client = _get_client()
collections = search_collection_names(client)

def norm(p):
    try: return int(p)
    except: return None

def hit(pages_top5, golds):
    """命中判定：top5 页任一与金标 ±1"""
    for p in pages_top5[:5]:
        if p is not None and any(abs(p - g) <= 1 for g in golds):
            return True
    return False

def v1_dense(query):
    """V1: 纯 dense top5"""
    dense_vec, _ = _encode_query(query)
    pts = []
    for col in collections:
        try:
            r = client.query_points(collection_name=col, query=dense_vec, using="dense", limit=5)
            pts.extend(list(r.points))
        except Exception:
            continue
    pts.sort(key=lambda p: -p.score)
    return [norm(getattr(p, "payload", {}).get("page")) for p in pts[:5]]

def v2_rrf(query):
    """V2(修复版): hybrid 召回 + dense 余弦全局排序 top5（线上当前 retrieve 逻辑）"""
    fused = retrieve(query, top_k=5)
    return [norm(h.get("page")) for h in fused]

def v3_rerank(query):
    """V3: 修复版粗排 15 → rerank 取 5"""
    fused = retrieve(query, top_k=15)
    rr = _get_reranker()
    if rr is not None and fused:
        pairs = [(query, h["content"]) for h in fused]
        scores = rr.compute_score(pairs, batch_size=8)
        order = sorted(range(len(fused)), key=lambda i: -float(scores[i]))
        # dense 高分保底（与线上 api.py 一致）
        d_order = sorted(range(len(fused)), key=lambda i: -(float(fused[i].get("dense_score") or 0)))
        protected = [i for i in d_order[:2] if i not in order[:2]]
        if protected:
            order = protected + [i for i in order if i not in protected]
        fused = [fused[i] for i in order][:5]
    else:
        fused = fused[:5]
    return [norm(h.get("page")) for h in fused]

with open(r"C:\\Users\\lizhihao\\w2-knowledge-assistant\\eval\\retrieval_set.json", encoding="utf-8") as f:
    R = json.load(f)

for ver, fn in [("V1(dense)", v1_dense), ("V2(RRF)", v2_rrf), ("V3(+rerank)", v3_rerank)]:
    hits = 0; rr_sum = 0.0
    print(f"\\n===== {ver} =====")
    for i, it in enumerate(R, 1):
        t0 = time.time()
        pages = fn(it["question"])
        ok = hit(pages, it["pages"])
        # rank = 首个命中位置
        rank = next((j+1 for j, p in enumerate(pages[:5]) if p is not None and any(abs(p-g)<=1 for g in it["pages"])), None)
        if ok: hits += 1
        if rank: rr_sum += 1.0/rank
        print(f"{'✅' if ok else '❌'} #{i:>2} 金标p{it['pages']} 前5页{pages} rank={rank} ({time.time()-t0:.1f}s)")
    n = len(R)
    print(f"recall@5={hits/n:.4f} ({hits}/{n})  MRR={rr_sum/n:.4f}")
