# -*- coding: utf-8 -*-
"""Day4：22 chunk 向量化（BGE-M3 dense+sparse）→ Qdrant embedded 入库。

对齐讲师《向量化方案》+《存储方案》：
- 编码文本 = content + important_kwd + question_kwd（Q2Q 参与向量化）
- BGEM3FlagModel 出 dense(1024) + sparse(lexical_weights)
- Qdrant embedded（path=data/qdrant），named vectors dense+sparse
- Point id 用确定性 UUID（embedded 不接受 string id）
- 幂等：先 delete_collection 再 create_collection

用法:
    D:/an/envs/langchain/python.exe scripts/vectorize_qdrant_v7.py
"""
from __future__ import annotations
import json
import os
import sys
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = r"C:\Users\lizhihao\w1-day5\device-rag-44653"
CHUNKS = Path(BASE) / "data" / "mineru_out" / "chunks_v7_final.json"
QDRANT_DIR = Path(BASE) / "data" / "qdrant"
BGE = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
COLLECTION = "device_knowledge_v7"
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
    """lexical_weights {token_id: weight} → Qdrant SparseVector（indices 升序）。"""
    from qdrant_client.models import SparseVector
    items = sorted(weights.items(), key=lambda kv: int(kv[0]))
    indices = [int(k) for k, _ in items]
    values = [float(v) for _, v in items]
    return SparseVector(indices=indices, values=values)


def main() -> None:
    chunks = json.load(open(CHUNKS, encoding="utf-8"))
    print(f"读取 chunks_v7_final.json: {len(chunks)} 块")

    from FlagEmbedding import BGEM3FlagModel
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, SparseVectorParams, Distance, Modifier, PointStruct

    print("加载 BGE-M3（BGEM3FlagModel，CPU）...")
    model = BGEM3FlagModel(BGE, use_fp16=False)

    embed_texts = [build_embed_text(c) for c in chunks]
    print(f"编码 {len(embed_texts)} 条（dense + sparse）...")
    # 注：模型目录已补 sparse_linear.pt/colbert_linear.pt 真实权重，批量 encode 的 sparse
    # 输出稳定且跨进程一致（此前的"空/不一致"根因是缺这两个权重文件导致随机初始化）。
    out = model.encode(
        embed_texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        batch_size=16,
    )
    dense = out["dense_vecs"]
    sparse = out["lexical_weights"]

    # sparse 词数检查
    empty = [i for i, w in enumerate(sparse) if len(w) == 0]
    if empty:
        print(f"⚠️ {len(empty)} 条 sparse 为空: {empty}")
    else:
        print(f"✅ sparse 全部非空（词数 {min(len(w) for w in sparse)}~{max(len(w) for w in sparse)}）")

    client = QdrantClient(path=str(QDRANT_DIR))
    # 幂等重建
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"已删除旧集合 {COLLECTION}")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    print(f"创建集合 {COLLECTION}（named vectors: dense + sparse）")

    points = []
    for i, c in enumerate(chunks):
        pid = str(uuid.uuid5(NAMESPACE, f"gb44653-{c['section']}-{i}"))
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
                "document_id": c.get("document_id", "gb_t_44653_2024"),
                "ingest_version": c.get("ingest_version", "20260825_v1"),
                "char_len": c.get("char_len"),
                "token_len": c.get("token_len"),
            },
        ))

    client.upsert(collection_name=COLLECTION, points=points)

    cnt = client.count(collection_name=COLLECTION).count
    print(f"[OK] 入库完成: {COLLECTION} = {cnt} 条")
    print(f"Qdrant 数据目录: {QDRANT_DIR}")


if __name__ == "__main__":
    main()
