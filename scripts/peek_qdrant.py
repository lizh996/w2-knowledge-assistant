# -*- coding: utf-8 -*-
'''查看 Qdrant 集合内容（方式 1：Qdrant API）——只读，不改数据。'''
import sys
sys.stdout.reconfigure(encoding="utf-8")
from qdrant_client import QdrantClient

client = QdrantClient(path=r"C:\Users\lizhihao\w1-day5\device-rag-44653\data\qdrant")
col = "device_knowledge_v7"
print(f"集合: {col}, 总点数: {client.count(col).count}")

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
points, _ = client.scroll(collection_name=col, limit=limit, with_vectors=True)
for p in points:
    print(f"\n📌 {p.payload.get('section')}  (p{p.payload.get('page')})")
    print(f"   semantic_type: {p.payload.get('semantic_type')}")
    print(f"   正文: {p.payload.get('text', '')[:80]}...")
    print(f"   dense: {len(p.vector['dense'])}维 | sparse: {len(p.vector['sparse'].indices)}词")

client.close()  # 显式关闭，避免退出时 __del__ 警告
