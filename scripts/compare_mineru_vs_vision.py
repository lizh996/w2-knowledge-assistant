"""对照实验：mineru 版 vs vision 版（同一份评测集对比）。

用法：
    D:/an/envs/langchain/python.exe scripts/compare_mineru_vs_vision.py

流程：
    1. 读 mineru content_list（带 page_idx）→ 按页合并 chunk → 入库 sf6_kb_mineru
    2. 用同一份 retrieval_set.json 评测 mineru 集合
    3. 对比 vision 版指标（读 report_day1_baseline.json）
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

BGE = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
CHROMA = r"C:\Users\lizhihao\w1-day5\device-rag-44653\data\chroma"
MINERU_OUT = Path(r"C:\Users\lizhihao\w1-day5\device-rag-44653\data\mineru_out")
EVAL_SET = Path(r"C:\Users\lizhihao\w1-day5\device-rag-44653\eval\retrieval_set.json")
SOURCE = "GB/T 44653-2024 六氟化硫气体现场循环再利用导则"

COLL_MINERU = "sf6_kb_mineru"
COLL_VISION = "sf6_kb"


def load_content_list() -> list[dict]:
    """读 mineru content_list（二次 json.loads 已处理）。"""
    for f in MINERU_OUT.glob("*content_list*"):
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, str):
            return json.loads(data)
    raise FileNotFoundError("content_list 未找到")


def build_chunks_by_page(content_list: list[dict]) -> list[dict]:
    """按页合并文本元素为 chunk（模拟讲师 mineru 整页 chunk）。"""
    pages: dict[int, list[str]] = {}
    for el in content_list:
        if el.get("type") in ("text", "list", "table", "equation"):
            page = el.get("page_idx")
            if page is not None and el.get("text"):
                pages.setdefault(page, []).append(el["text"])
    chunks = []
    for page in sorted(pages):
        text = "\n".join(pages[page])
        chunks.append({
            "id": f"mineru-p{page + 1:02d}",
            "text": text,
            "page": page + 1,  # page_idx 0-based → 物理页 1-based
            "section": "mineru",
        })
    return chunks


def ingest(chunks: list[dict]) -> None:
    """入库 mineru 集合。"""
    os.environ["BGE_M3_MODEL"] = BGE
    from sentence_transformers import SentenceTransformer
    import chromadb

    model = SentenceTransformer(BGE)
    client = chromadb.PersistentClient(path=CHROMA)
    # 清掉旧 mineru 集合（重跑幂等）
    try:
        client.delete_collection(COLL_MINERU)
    except Exception:
        pass
    col = client.create_collection(COLL_MINERU, metadata={"hnsw:space": "cosine"})

    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    col.add(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[{"page": c["page"], "section": c["section"], "doc": SOURCE} for c in chunks],
    )
    print(f"✅ mineru 集合入库: {col.count()} 块")


def evaluate(model, col, items: list[dict]) -> dict:
    """同一份评测逻辑（page 级判定，±1 容差）。"""
    hits, ranks = [], []
    for item in items:
        if item["should_refuse"]:
            continue
        qv = model.encode([item["question"]], normalize_embeddings=True)
        res = col.query(query_embeddings=qv.tolist(), n_results=5)
        pages = [m.get("page") for m in res["metadatas"][0]]
        rank = next((i + 1 for i, pg in enumerate(pages) if abs(pg - item["expected_page"]) <= 1), None)
        hits.append(rank is not None)
        ranks.append(rank if rank else 0)
    recall = sum(hits) / len(hits) if hits else 0
    mrr = sum(1 / r for r in ranks if r > 0) / len(ranks) if ranks else 0
    return recall, mrr


def main() -> None:
    content_list = load_content_list()
    print(f"content_list: {len(content_list)} 元素")

    chunks = build_chunks_by_page(content_list)
    print(f"按页合并: {len(chunks)} 块")
    ingest(chunks)

    items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    pos = [i for i in items if not i["should_refuse"]]

    os.environ["BGE_M3_MODEL"] = BGE
    from sentence_transformers import SentenceTransformer
    import chromadb

    model = SentenceTransformer(BGE)
    client = chromadb.PersistentClient(path=CHROMA)

    # mineru 版
    col_m = client.get_collection(COLL_MINERU)
    recall_m, mrr_m = evaluate(model, col_m, pos)

    # vision 版（现库）
    col_v = client.get_collection(COLL_VISION)
    recall_v, mrr_v = evaluate(model, col_v, pos)

    print("\n=== 对照结果 ===")
    print(f"{'指标':<12} {'vision版':<12} {'mineru版':<12}")
    print(f"{'recall@k':<12} {recall_v:<12.4f} {recall_m:<12.4f}")
    print(f"{'MRR':<12} {mrr_v:<12.4f} {mrr_m:<12.4f}")

    result = {
        "date": "2026-08-25",
        "vision": {"recall_at_k": recall_v, "mrr": mrr_v, "collection": COLL_VISION},
        "mineru": {"recall_at_k": recall_m, "mrr": mrr_m, "collection": COLL_MINERU},
        "note": "同一份评测集，仅解析方式不同",
    }
    out = Path(r"C:\Users\lizhihao\w1-day5\device-rag-44653\eval\compare_mineru_vs_vision.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 对比报告: {out}")


if __name__ == "__main__":
    main()
