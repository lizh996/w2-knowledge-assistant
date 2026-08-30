"""P2: 流程图语义入库（citation 契约）。

3 个 chunk：
- flow-01: 第一类流程（现场回收→检测→合格回充/不合格过滤干燥循环）
- flow-02: 第二类流程（现场净化 或 基地净化→检测→回充/暂存）
- flow-03: 流程总览（起点/分支/3条反馈回路）

用法：PYTHONPATH=backend D:/an/envs/langchain/python.exe scripts/ingest_flow.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

SOURCE = "GB/T 44653-2024 六氟化硫气体现场循环再利用导则"
PAGE = 10

FLOW_CHUNKS = [
    {
        "id": "gb44653-flow-01",
        "text": (
            "【SF6循环再利用 · 第一类气体流程】"
            "第一类气体（检测结果满足表1要求或仅湿度不满足表1要求的气体）："
            "现场回收 → 现场检测 → 合格则回充设备；"
            "不合格则过滤干燥 → 回到现场检测循环，直至合格。"
            "处理方式：采用带水分和颗粒物滤除模块的SF6气体回收装置，"
            "必要时串联SF6预处理装置。依据 GB/T 44653-2024 图1 第10页。"
        ),
    },
    {
        "id": "gb44653-flow-02",
        "text": (
            "【SF6循环再利用 · 第二类气体流程】"
            "第二类气体（除湿度外仍有指标不合格的气体）："
            "现场回收后分两条支路："
            "支路B1 现场净化 → 现场检测 → 合格回充设备，不合格回到现场净化循环；"
            "支路B2 基地净化 → 基地检测 → 合格暂存待用，不合格回到基地净化循环。"
            "现场具备条件时用具备过滤、干燥、吸附、低温冷却、精馏功能的净化装置；"
            "现场不具备条件时运输至回收处理基地净化。依据 GB/T 44653-2024 图1 第10页。"
        ),
    },
    {
        "id": "gb44653-flow-03",
        "text": (
            "【SF6循环再利用 · 流程总览】"
            "起点：设备内待回收SF6气体 → 现场检测（用于分类，不判合格）→ 按气体类别分支："
            "第一类（合格或仅湿度不合格）走A路径；第二类（其他指标不合格）走B路径。"
            "流程共4处判断：初始现场检测（分类分流）、第一类现场检测、第二类现场检测、基地检测。"
            "共3条反馈回路：①过滤干燥→现场检测循环 ②现场净化→现场检测循环 ③基地净化→基地检测循环。"
            "终点：回充设备 或 暂存待用。依据 GB/T 44653-2024 图1 第10页。"
        ),
    },
]


def main() -> None:
    import os as _os
    _os.environ["BGE_M3_MODEL"] = r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"

    from sentence_transformers import SentenceTransformer
    import chromadb

    model = SentenceTransformer(_os.environ["BGE_M3_MODEL"])
    client = chromadb.PersistentClient(path=r"C:\Users\lizhihao\w1-day5\device-rag-44653\data\chroma")
    col = client.get_or_create_collection("sf6_kb", metadata={"hnsw:space": "cosine"})

    texts = [c["text"] for c in FLOW_CHUNKS]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    metadatas = [
        {"page": PAGE, "section": "flow", "doc": SOURCE, "ingest_version": "day2_p10"}
        for _ in FLOW_CHUNKS
    ]

    col.add(
        ids=[c["id"] for c in FLOW_CHUNKS],
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"✅ 入库完成: {len(FLOW_CHUNKS)} 个流程图 chunk → 集合现有 {col.count()} 块")


if __name__ == "__main__":
    main()
