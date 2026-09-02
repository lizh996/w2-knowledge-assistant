"""FR-004/FR-006: Qdrant 复合检索（FusionQuery RRF）+ top-k + 阈值 + 拒答判定。

检索底层：Qdrant embedded（device_knowledge_v8 + 上传集合），BGE-M3 dense+sparse 融合。
上层接口保持简洁：retrieve(query) -> list[dict]（含 score/section/page/content/source）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import threading

# 路径常量（Qdrant 数据 + BGE-M3 模型）
_BASE = Path(r"C:\Users\lizhihao\w2-knowledge-assistant")
_QDRANT_DIR = _BASE / "data" / "qdrant"
_BGE = os.environ.get("BGE_MODEL_PATH", r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181")  # 环境变量切换微调模型(默认原模型=可回滚)
_COLLECTION = "transformer_kb_v1"
_UPLOAD_COLLECTION_PREFIX = "transformer_upload_"
_RRF_K = 60
SEMANTIC_TYPES = ["概述", "术语定义", "参数查询", "流程步骤", "标准要求", "安全要求", "图语义"]
TOKEN_BUCKETS = ["0-100", "101-300", "301-512", "513+"]

# 拒答阈值（非 RRF 分）。
# dense：语义相关度。实测库内 min=0.6077、库外 max=0.5449，阈值 0.55 切开语义题。
# sparse：精确词命中度（标准号类）。标准号库内题 sparse 0.140~0.192，库外题 0.002~0.127，
#         阈值 0.13 切开精确词题。
# 判定：dense 高分 或 sparse 高分 任一满足即放行（互补——dense 善语义、sparse 善精确词）。
REJECT_DENSE_THRESHOLD = 0.445  # 微调模型校准(原0.55): 库内min0.485 vs 库外max0.406
REJECT_SPARSE_THRESHOLD = 0.20  # 微调模型校准(原0.13): 库外max0.173

# 懒加载单例（模型加载慢，进程内复用）
_model = None
_client = None
_init_lock = threading.Lock()


def _ensure_temp_dir() -> None:
    temp = _BASE / "data" / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMP", str(temp))
    os.environ.setdefault("TEMP", str(temp))
    os.environ.setdefault("TMPDIR", str(temp))


def _get_model():
    global _model
    if _model is None:
        with _init_lock:
            if _model is None:
                _ensure_temp_dir()
                from FlagEmbedding import BGEM3FlagModel
                _model = BGEM3FlagModel(_BGE, use_fp16=False)
    return _model


def _get_client():
    global _client
    if _client is None:
        with _init_lock:
            if _client is None:
                _repair_upload_collection_meta()
                from qdrant_client import QdrantClient
                _client = QdrantClient(path=str(_QDRANT_DIR))
    return _client


def _repair_upload_collection_meta() -> None:
    """补齐磁盘上已存在、但 meta.json 漏登记的上传集合。

    旧上传任务可能留下完整 collection 目录却没有写回 meta.json，Qdrant
    embedded 因此无法通过集合名访问。这里仅复制 v8 的向量配置登记集合名，
    不改写任何点数据。
    """
    meta_file = _QDRANT_DIR / "meta.json"
    collection_dir = _QDRANT_DIR / "collection"
    if not meta_file.exists() or not collection_dir.exists():
        return
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        collections = data.setdefault("collections", {})
        template = collections.get(_COLLECTION)
        if template is None:
            return
        missing = [
            path.name for path in collection_dir.iterdir()
            if path.is_dir()
            and path.name.startswith(_UPLOAD_COLLECTION_PREFIX)
            and path.name not in collections
        ]
        if not missing:
            return
        for name in missing:
            collections[name] = json.loads(json.dumps(template, ensure_ascii=False))
        meta_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except (OSError, json.JSONDecodeError, TypeError):
        return


def _to_sparse_vector(weights: dict):
    from qdrant_client.models import SparseVector
    items = sorted(weights.items(), key=lambda kv: int(kv[0]))
    return SparseVector(indices=[int(k) for k, _ in items],
                        values=[float(v) for _, v in items])


def search_collection_names(client=None) -> list[str]:
    """返回可参与检索的集合：内置 v8 + 所有上传集合。"""
    client = client or _get_client()
    names = [col.name for col in client.get_collections().collections]
    selected: list[str] = []
    if _COLLECTION in names:
        selected.append(_COLLECTION)
    selected.extend(
        name for name in sorted(names)
        if name.startswith(_UPLOAD_COLLECTION_PREFIX) and name not in selected
    )
    return selected


def _is_upload_collection(collection: str | None) -> bool:
    return bool(collection and collection.startswith(_UPLOAD_COLLECTION_PREFIX))


def _document_kind(collection: str | None) -> str:
    return "上传文档" if _is_upload_collection(collection) else "内置文档"


def _hit_key(hit: dict) -> str:
    content = " ".join((hit.get("content") or "").split())[:180]
    doc = hit.get("document_id") or hit.get("source") or hit.get("document") or ""
    if doc or content:
        return "|".join([
            str(doc),
            str(hit.get("page") or ""),
            str(hit.get("section") or ""),
            content,
        ])
    return f"{hit.get('collection', '')}:{hit.get('id', '')}"


def _merge_score_hits(ranked_lists: list[tuple[str, list[Any]]], limit: int) -> list[dict]:
    """合并 dense/sparse 原始通道分数，按最高原始分去重。"""
    best: dict[str, dict] = {}
    for collection, points in ranked_lists:
        for rank, hit in enumerate(points, start=1):
            item = _format_hit(hit, "score", collection=collection)
            item["rank"] = rank
            key = _hit_key(item)
            previous = best.get(key)
            if previous is None or float(item.get("score") or 0) > float(previous.get("score") or 0):
                best[key] = item
    return sorted(
        best.values(),
        key=lambda item: (-float(item.get("score") or 0), -float(item.get("source_score") or 0)),
    )[:limit]


def _fuse_ranked_hits(ranked_lists: list[tuple[str, list[Any]]], limit: int) -> list[dict]:
    """对多个集合返回的 RRF 排名再做一次全局 RRF 融合。"""
    fused: dict[str, dict] = {}
    for collection, points in ranked_lists:
        for rank, hit in enumerate(points, start=1):
            item = _format_hit(hit, "score", collection=collection)
            key = _hit_key(item)
            add_score = 1.0 / (_RRF_K + rank)
            if key not in fused:
                item["score"] = add_score
                item["rrf_score"] = add_score
                item["source_score"] = hit.score
                item["rank"] = rank
                fused[key] = item
            else:
                fused[key]["score"] += add_score
                fused[key]["rrf_score"] = fused[key]["score"]
                if float(hit.score or 0) > float(fused[key].get("source_score") or 0):
                    fused[key]["source_score"] = hit.score
                    fused[key]["rank"] = rank
                    for field in [
                        "id", "chunk_id", "page", "section", "content", "source",
                        "document", "semantic_type", "document_id", "extract_mode",
                        "image_path", "mermaid", "collection", "document_kind",
                    ]:
                        fused[key][field] = item.get(field)
    return sorted(
        fused.values(),
        key=lambda item: (-float(item.get("score") or 0), -float(item.get("source_score") or 0)),
    )[:limit]


def _query_rrf_collections(client, collections: list[str], dense_vec, sparse_vec, limit: int):
    from qdrant_client.models import Prefetch, FusionQuery, Fusion

    pool = max(limit * 4, 20)
    ranked: list[tuple[str, list[Any]]] = []
    for collection in collections:
        try:
            resp = client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(query=sparse_vec, using="sparse", limit=pool),
                    Prefetch(query=dense_vec, using="dense", limit=pool),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=pool,
            )
        except Exception:
            if collection == _COLLECTION:
                raise
            continue
        ranked.append((collection, list(resp.points)))
    return ranked


def _query_channel_collections(client, collections: list[str], query_vec, using: str, limit: int):
    pool = max(limit * 4, 20)
    ranked: list[tuple[str, list[Any]]] = []
    for collection in collections:
        try:
            resp = client.query_points(
                collection_name=collection,
                query=query_vec,
                using=using,
                limit=pool,
            )
        except Exception:
            if collection == _COLLECTION:
                raise
            continue
        ranked.append((collection, list(resp.points)))
    return ranked


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """复合检索：BGE-M3 编码 query → Qdrant FusionQuery(RRF) dense+sparse 融合。

    返回 list[dict]，每项含 score（RRF 分）/ section / page / content / source。
    """
    model = _get_model()
    client = _get_client()
    collections = search_collection_names(client)
    if not collections:
        return []

    # 逐条 encode query（避免 batch padding 影响 sparse）
    out = model.encode([query], return_dense=True, return_sparse=True,
                       return_colbert_vecs=False, batch_size=1)
    dense_vec = out["dense_vecs"][0].tolist()
    sparse_vec = _to_sparse_vector(out["lexical_weights"][0])

    ranked = _query_rrf_collections(client, collections, dense_vec, sparse_vec, top_k)
    return _fuse_ranked_hits(ranked, top_k)


_RERANKER_PATH = r"D:\models\bge-reranker-v2-m3"
_reranker = None


def _get_reranker():
    """懒加载 bge-reranker（受 RERANK_ENABLED 控制，默认关闭）。"""
    if os.environ.get("RERANK_ENABLED", "0") != "1":
        return None
    global _reranker
    if _reranker is None:
        from pathlib import Path as _P
        if _P(_RERANKER_PATH).exists():
            from FlagEmbedding import FlagReranker
            _reranker = FlagReranker(_RERANKER_PATH, use_fp16=False)
    return _reranker


def retrieve_debug(query: str, top_k: int = 5, enable_rerank: bool = False) -> dict[str, list[dict]]:
    """Return dense/sparse/RRF channels for the admin retrieval test console."""
    model = _get_model()
    client = _get_client()
    collections = search_collection_names(client)
    if not collections:
        return {"dense": [], "sparse": [], "rrf": [], "reranked": [], "similarity": []}

    out = model.encode([query], return_dense=True, return_sparse=True,
                       return_colbert_vecs=False, batch_size=1)
    dense_vec = out["dense_vecs"][0].tolist()
    sparse_vec = _to_sparse_vector(out["lexical_weights"][0])

    dense = _merge_score_hits(
        _query_channel_collections(client, collections, dense_vec, "dense", top_k),
        top_k,
    )
    sparse = _merge_score_hits(
        _query_channel_collections(client, collections, sparse_vec, "sparse", top_k),
        top_k,
    )
    rrf = _fuse_ranked_hits(
        _query_rrf_collections(client, collections, dense_vec, sparse_vec, top_k),
        top_k,
    )
    reranked = []
    reranker = _get_reranker() if enable_rerank else None
    if reranker is not None:
        pairs = [(query, c["content"]) for c in rrf]
        scores = reranker.compute_score(pairs, batch_size=8)
        order = sorted(range(len(rrf)), key=lambda i: -float(scores[i]))
        reranked = [
            {**rrf[i], "rerank_score": round(float(scores[i]), 4)}
            for i in order
        ]
    else:
        reranked = [{**hit, "rerank_score": None} for hit in rrf]
    return {
        "dense": dense,
        "sparse": sparse,
        "rrf": rrf,
        "reranked": reranked,
        "similarity": [
            {
                "page": hit["page"],
                "score": hit.get("rerank_score") if hit.get("rerank_score") is not None else hit["score"],
                "section": hit["section"],
            }
            for hit in reranked
        ],
    }


def _format_hit(hit, score_name: str, *, collection: str = "") -> dict:
    p = hit.payload or {}
    point_id = str(getattr(hit, "id", ""))
    source = p.get("source", "")
    return {
        "id": point_id,
        "chunk_id": point_id,
        "page": p.get("page"),
        score_name: hit.score,
        "section": p.get("section", ""),
        "content": p.get("text", ""),
        "source": source,
        "document": source,
        "semantic_type": p.get("semantic_type", ""),
        "document_id": p.get("document_id", ""),
        "extract_mode": p.get("extract_mode", ""),
        "image_path": p.get("image_path", ""),
        "mermaid": p.get("mermaid", ""),
        "collection": collection,
        "document_kind": _document_kind(collection),
    }


def _encode_query(query: str):
    """编码 query，返回 (dense_vec, sparse_vec)。"""
    model = _get_model()
    out = model.encode([query], return_dense=True, return_sparse=True,
                       return_colbert_vecs=False, batch_size=1)
    return out["dense_vecs"][0].tolist(), _to_sparse_vector(out["lexical_weights"][0])


def dense_similarity(query: str) -> float:
    """query 与所有可检索集合 top1 结果的 dense 余弦相似度（语义相关度）。"""
    client = _get_client()
    dense_vec, _ = _encode_query(query)
    scores = []
    for collection in search_collection_names(client):
        try:
            resp = client.query_points(
                collection_name=collection,
                query=dense_vec,
                using="dense",
                limit=1,
            )
        except Exception:
            if collection == _COLLECTION:
                raise
            continue
        if resp.points:
            scores.append(float(resp.points[0].score))
    return max(scores, default=0.0)


def sparse_similarity(query: str) -> float:
    """query 与所有可检索集合 top1 结果的 sparse 命中分（精确词相关度）。"""
    client = _get_client()
    _, sparse_vec = _encode_query(query)
    scores = []
    for collection in search_collection_names(client):
        try:
            resp = client.query_points(
                collection_name=collection,
                query=sparse_vec,
                using="sparse",
                limit=1,
            )
        except Exception:
            if collection == _COLLECTION:
                raise
            continue
        if resp.points:
            scores.append(float(resp.points[0].score))
    return max(scores, default=0.0)


def _estimate_tokens(text: str) -> int:
    total = 0.0
    for ch in text:
        if ord(ch) >= 0x2E80:
            total += 1
        elif ch.strip():
            total += 0.5
    return int(total)


def _format_scroll_point(point, *, collection: str) -> dict:
    payload = point.payload or {}
    content = payload.get("text", "")
    token_len = payload.get("token_len")
    char_len = payload.get("char_len")
    if token_len is None:
        token_len = _estimate_tokens(content)
    if char_len is None:
        char_len = len(content)
    point_id = str(getattr(point, "id", ""))
    source = payload.get("source", "")
    return {
        "id": point_id,
        "chunk_id": point_id,
        "content": content,
        "page": payload.get("page"),
        "document": source,
        "source": source,
        "section": payload.get("section", ""),
        "semantic_type": payload.get("semantic_type", ""),
        "document_id": payload.get("document_id", ""),
        "extract_mode": payload.get("extract_mode", ""),
        "image_path": payload.get("image_path", ""),
        "mermaid": payload.get("mermaid", ""),
        "token_len": int(token_len or 0),
        "char_len": int(char_len or 0),
        "collection": collection,
        "document_kind": _document_kind(collection),
    }


def _target_collections(
    client,
    *,
    document_id: str | None = None,
    collection: str | None = None,
) -> tuple[list[str], str | None]:
    collections = search_collection_names(client)
    if collection:
        return ([collection] if collection in collections else []), document_id
    if not document_id:
        return collections, None
    if document_id.startswith(_UPLOAD_COLLECTION_PREFIX):
        return ([document_id] if document_id in collections else []), None
    upload_collection = f"{_UPLOAD_COLLECTION_PREFIX}{document_id}"
    if upload_collection in collections:
        return [upload_collection], document_id
    if document_id.startswith("gb_t_"):
        return ([_COLLECTION] if _COLLECTION in collections else []), document_id
    return collections, document_id


def _chunk_filter(document_id: str | None, semantic_type: str | None):
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    must = []
    if document_id:
        must.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
    if semantic_type:
        must.append(FieldCondition(key="semantic_type", match=MatchValue(value=semantic_type)))
    return Filter(must=must) if must else None


def chunk_summary(items: list[dict]) -> dict[str, Any]:
    semantic = {label: 0 for label in SEMANTIC_TYPES}
    unknown_semantic_count = 0
    buckets = {label: 0 for label in TOKEN_BUCKETS}
    for item in items:
        semantic_type = item.get("semantic_type") or ""
        if semantic_type in semantic:
            semantic[semantic_type] += 1
        else:
            unknown_semantic_count += 1
        tokens = int(item.get("token_len") or 0)
        if tokens <= 100:
            buckets["0-100"] += 1
        elif tokens <= 300:
            buckets["101-300"] += 1
        elif tokens <= 512:
            buckets["301-512"] += 1
        else:
            buckets["513+"] += 1
    total_tokens = sum(int(item.get("token_len") or 0) for item in items)
    return {
        "chunk_count": len(items),
        "avg_token": round(total_tokens / len(items), 1) if items else 0,
        "fragment_count": sum(1 for item in items if int(item.get("char_len") or 0) < 100),
        "semantic_distribution": semantic,
        "unknown_semantic_count": unknown_semantic_count,
        "token_histogram": buckets,
    }


def list_chunks(
    *,
    document_id: str | None = None,
    semantic_type: str | None = None,
    collection: str | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """滚动读取所有目标集合的 chunk payload，供后台分块看板使用。"""
    client = _get_client()
    collections, payload_document_id = _target_collections(
        client,
        document_id=document_id,
        collection=collection,
    )
    scroll_filter = _chunk_filter(payload_document_id, semantic_type)
    items: list[dict] = []
    for name in collections:
        offset = None
        while len(items) < limit:
            points, offset = client.scroll(
                collection_name=name,
                scroll_filter=scroll_filter,
                limit=min(256, max(1, limit - len(items))),
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            items.extend(_format_scroll_point(point, collection=name) for point in points)
            if offset is None:
                break
        if len(items) >= limit:
            break
    return {
        "items": items,
        "summary": chunk_summary(items),
        "semantic_types": SEMANTIC_TYPES,
        "token_buckets": TOKEN_BUCKETS,
        "collections": collections,
    }


def _point_id(chunk_id: str) -> str | int:
    return int(chunk_id) if chunk_id.isdigit() else chunk_id


def get_chunk_by_id(chunk_id: str, *, collection: str | None = None) -> dict | None:
    client = _get_client()
    collections, _ = _target_collections(client, collection=collection)
    for name in collections:
        points = client.retrieve(
            collection_name=name,
            ids=[_point_id(chunk_id)],
            with_payload=True,
            with_vectors=False,
        )
        if points:
            return _format_scroll_point(points[0], collection=name)
    return None


def should_reject_by_dense(
    query: str,
    dense_threshold: float = REJECT_DENSE_THRESHOLD,
    sparse_threshold: float = REJECT_SPARSE_THRESHOLD,
) -> bool:
    """拒答判定：dense 或 sparse 任一命中即放行（互补）。

    dense 善语义（"回收率怎么算"）、sparse 善精确词（"DL/T 920"）。单独用 dense
    会误拒标准号类库内题（dense 低分但 sparse 高分）；单独用 sparse 会误放行库外题。
    故：dense < 阈值 且 sparse < 阈值 才拒答。
    """
    dense = dense_similarity(query)
    if dense >= dense_threshold:
        return False
    sparse = sparse_similarity(query)
    return sparse < sparse_threshold


def top_k(results: list[dict], k: int = 4) -> list[dict]:
    """取相似度最高的前 k 条。"""
    return results[:k]


def filter_by_threshold(results: list[dict], threshold: float = 0.4) -> list[dict]:
    """过滤低于相似度阈值的片段。"""
    return [r for r in results if r.get("score", 0) >= threshold]


def should_reject(results: list[dict]) -> bool:
    """检索结果为空 → 拒答。"""
    return len(results) == 0
