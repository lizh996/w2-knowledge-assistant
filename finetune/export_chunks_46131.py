# -*- coding: utf-8 -*-
"""从 qdrant 上传集合 transformer_upload_6d10895f0f2448c4ab374454b57631d1
导出 GB/T 46131 的 61 块 chunk 文本 → chunks_46131.json（Unseen Knowledge 子集素材）。

该集合未参与嵌入模型微调训练（训练数据 train_clean.jsonl 仅来自 chunks_all.json 的 88 块）。
点以 pickle(PointStruct) 形式存于 collection 的 storage.sqlite，需 qdrant_client 反序列化。

用法（langchain env 有 qdrant_client）: D:\\an\\envs\\langchain\\python.exe export_chunks_46131.py
输出: finetune/chunks_46131.json
"""
import os
import sys
import json
import sqlite3
import pickle
import ast

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
SQLITE = os.path.join(BASE, "..", "data", "qdrant", "collection",
                      "transformer_upload_6d10895f0f2448c4ab374454b57631d1", "storage.sqlite")
OUT = os.path.join(BASE, "chunks_46131.json")


def parse_kwd(s):
    """question_kwd / important_kwd 存的是字符串化的 list，这里还原为 list[str]。"""
    if not s:
        return []
    if isinstance(s, list):
        return s
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError):
        return []


if __name__ == "__main__":
    conn = sqlite3.connect(os.path.abspath(SQLITE))
    cur = conn.cursor()
    cur.execute("SELECT id, point FROM points ORDER BY id")
    rows = cur.fetchall()
    conn.close()

    chunks = []
    for pid, blob in rows:
        point = pickle.loads(blob)
        payload = point.payload or {}
        chunks.append({
            "chunk_id": str(point.id),
            "source": payload.get("source", ""),
            "page": payload.get("page", ""),
            "section": payload.get("section", ""),
            "semantic_type": payload.get("semantic_type", ""),
            "text": payload.get("text", ""),
            "question_kwd": parse_kwd(payload.get("question_kwd", "")),
            "important_kwd": parse_kwd(payload.get("important_kwd", "")),
        })

    chunks.sort(key=lambda c: (c["page"], c["chunk_id"]))
    json.dump(chunks, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[OK] 导出 {len(chunks)} 块 → {OUT}")
    print(f"     source: {sorted({c['source'] for c in chunks})}")
    print(f"     semantic_type 分布: { {t: sum(1 for c in chunks if c['semantic_type'] == t) for t in sorted({c['semantic_type'] for c in chunks})} }")
    # 打印首块摘要，确认文本正常
    if chunks:
        print("     首块示例 text[:80]:", chunks[0]["text"][:80])
