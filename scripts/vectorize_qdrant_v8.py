# -*- coding: utf-8 -*-
"""多文档向量化入库：device_knowledge_v8（3 文档合并）。

- 数据源：chunks_v7_final.json（GB/T 44653，22 块，成果保留）+ chunks_multidoc_new.json（12022 + 18867）
- BGE-M3 dense(1024) + sparse(lexical_weights)，编码文本 = content + important_kwd + question_kwd
- 集合 device_knowledge_v8，payload 带 document_id，支持按 document_id 过滤检索
- 幂等：先 delete_collection 再 create_collection（不动 device_knowledge_v7）

用法:
    D:/an/envs/langchain/python.exe scripts/vectorize_qdrant_v8.py
"""
from __future__ import annotations
import json
import re
import sys
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = r"C:\Users\lizhihao\w1-day5\device-rag-44653"
V7_CHUNKS = Path(BASE) / "data" / "mineru_out" / "chunks_v7_final.json"
NEW_CHUNKS = Path(BASE) / "data" / "mineru_out" / "chunks_multidoc_new.json"
QDRANT_DIR = Path(BASE) / "data" / "qdrant"
BGE = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
COLLECTION = "device_knowledge_v8"
NAMESPACE = uuid.NAMESPACE_URL


def build_embed_text(c: dict) -> str:
    parts = [c["content"]]
    ik = " ".join(c.get("important_kwd", []))
    qk = " ".join(c.get("question_kwd", []))
    if ik:
        parts.append(ik)
    if qk:
        parts.append(qk)
    return "\n".join(parts)


def to_sparse_vector(weights: dict):
    from qdrant_client.models import SparseVector
    items = sorted(weights.items(), key=lambda kv: int(kv[0]))
    indices = [int(k) for k, _ in items]
    values = [float(v) for _, v in items]
    return SparseVector(indices=indices, values=values)


def _load_44653_mermaid() -> str:
    """从 44653 的 content_list.json 提取流程图 mermaid（图块 chunk 里没有，运行时补）。

    不修改 44653 已有产物（chunks_v7_final.json 保持不动），只在入库时补 mermaid 供前端渲染。
    """
    cl_path = Path(BASE) / "data" / "mineru_out" / "content_list.json"
    if not cl_path.exists():
        return ""
    try:
        cl = json.load(open(cl_path, encoding="utf-8"))
    except Exception:
        return ""
    for e in cl:
        if e.get("type") == "image" and e.get("content"):
            m = re.search(r"```mermaid\s*(.*?)```", e["content"], re.S)
            if m:
                return m.group(1).strip()
    return ""


def main() -> None:
    v7 = json.load(open(V7_CHUNKS, encoding="utf-8"))
    new = json.load(open(NEW_CHUNKS, encoding="utf-8")) if NEW_CHUNKS.exists() else []
    # 44653 图块补 mermaid（chunks_v7_final.json 里图块无此字段，运行时从 content_list 补）
    mermaid_44653 = _load_44653_mermaid()
    for c in v7:
        if c.get("type") == "image" and not c.get("mermaid") and mermaid_44653:
            c["mermaid"] = mermaid_44653
    chunks = v7 + new
    print(f"读取 chunks: 44653={len(v7)} + 新文档={len(new)} = {len(chunks)}")

    from FlagEmbedding import BGEM3FlagModel
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, SparseVectorParams, Distance, Modifier, PointStruct

    print("加载 BGE-M3（BGEM3FlagModel，CPU）...")
    model = BGEM3FlagModel(BGE, use_fp16=False)

    embed_texts = [build_embed_text(c) for c in chunks]
    print(f"编码 {len(embed_texts)} 条（dense + sparse）...")
    out = model.encode(
        embed_texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        batch_size=16,
    )
    dense = out["dense_vecs"]
    sparse = out["lexical_weights"]

    empty = [i for i, w in enumerate(sparse) if len(w) == 0]
    if empty:
        print(f"⚠️ {len(empty)} 条 sparse 为空: {empty}")
    else:
        print(f"✅ sparse 全部非空（词数 {min(len(w) for w in sparse)}~{max(len(w) for w in sparse)}）")

    client = QdrantClient(path=str(QDRANT_DIR))
    # 幂等重建：delete_collection 在 Qdrant local 某些版本不彻底（磁盘残留导致重复点）。
    # 先 close 释放 storage.sqlite 文件锁，gc + 重试删磁盘目录，确保重建干净（不动 v7 等其他集合）。
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"已删除旧集合 {COLLECTION}")
    client.close()

    import gc
    import shutil
    import time
    gc.collect()
    col_dir = QDRANT_DIR / "collection" / COLLECTION
    for attempt in range(5):
        try:
            if col_dir.exists():
                shutil.rmtree(col_dir)
                print(f"已清理磁盘残留 {col_dir}")
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(1.0 * (attempt + 1))
            gc.collect()

    client = QdrantClient(path=str(QDRANT_DIR))
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    print(f"创建集合 {COLLECTION}（named vectors: dense + sparse）")

    points = []
    for i, c in enumerate(chunks):
        doc_id = c.get("document_id", "gb_t_44653_2024")
        pid = str(uuid.uuid5(NAMESPACE, f"{doc_id}-{c['section']}-{i}"))
        points.append(PointStruct(
            id=pid,
            vector={
                "dense": dense[i].tolist(),
                "sparse": to_sparse_vector(sparse[i]),
            },
            payload={
                "text": c["content"],
                "source": c["source"],
                "page": c["page"],
                "type": c["type"],
                "section": c["section"],
                "semantic_type": c["semantic_type"],
                "important_kwd": c["important_kwd"],
                "question_kwd": c["question_kwd"],
                "document_id": doc_id,
                "ingest_version": c.get("ingest_version", "20260825_v1"),
                "char_len": c.get("char_len"),
                "token_len": c.get("token_len"),
                "image_path": c.get("image_path", ""),
                "mermaid": c.get("mermaid", ""),
            },
        ))

    client.upsert(collection_name=COLLECTION, points=points)

    cnt = client.count(collection_name=COLLECTION).count
    from collections import Counter
    doc_dist = Counter(p.payload["document_id"] for p in points)
    print(f"[OK] 入库完成: {COLLECTION} = {cnt} 条")
    print(f"document_id 分布: {dict(doc_dist)}")
    print(f"Qdrant 数据目录: {QDRANT_DIR}")


if __name__ == "__main__":
    main()
