# -*- coding: utf-8 -*-
"""清理幽灵上传集合(重复 GB/T 46131 分接开关文档)。

背景:Qdrant 里有 3 个上传集合,全部是同一份 GB/T 46131(各 61 块):
  - transformer_upload_6d10895f0f2448c4ab374454b57631d1  -> 合法(有 pipeline task 记录,保留)
  - transformer_upload_edbdac2e0e38497e8b076596716bb05c  -> 幽灵(无 task 记录,删除)
  - transformer_upload_ba514a27710f49d1b247c85d344b99d5  -> 幽灵(无 task 记录,删除)

用法(必须先停服务,释放 Qdrant 文件锁):
    D:/an/envs/langchain/python.exe scripts/cleanup_ghost_collections.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from qdrant_client import QdrantClient

BASE = Path(r"C:\Users\lizhihao\w2-knowledge-assistant")
QDRANT_DIR = BASE / "data" / "qdrant"

# 保留:内置集合 + 唯一合法上传集合
KEEP = {
    "transformer_kb_v1",
    "transformer_upload_6d10895f0f2448c4ab374454b57631d1",
}

# 待删幽灵集合
GHOSTS = [
    "transformer_upload_edbdac2e0e38497e8b076596716bb05c",
    "transformer_upload_ba514a27710f49d1b247c85d344b99d5",
]


def main() -> None:
    client = QdrantClient(path=str(QDRANT_DIR))
    existing = {c.name for c in client.get_collections().collections}
    print(f"清理前集合: {sorted(existing)}")

    for col in GHOSTS:
        if col in existing:
            try:
                client.delete_collection(col)
                print(f"✅ 已通过 Qdrant 删除集合 {col}")
            except Exception as exc:
                print(f"⚠️ Qdrant 删除失败 {col}: {exc}")
        else:
            print(f"ℹ️ 集合 {col} 不在 Qdrant 注册中(可能已被 repair 之外的路径移除)")
    client.close()

    # 清理 meta.json 里的幽灵登记(双保险:即使 Qdrant delete 不彻底)
    meta_file = QDRANT_DIR / "meta.json"
    if meta_file.exists():
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        collections = data.get("collections", {})
        removed = [k for k in GHOSTS if k in collections]
        for k in removed:
            del collections[k]
        meta_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"✅ meta.json 移除登记: {removed}")

    # 清理磁盘残留 collection 目录
    import gc
    import time
    gc.collect()
    for col in GHOSTS:
        col_dir = QDRANT_DIR / "collection" / col
        if col_dir.exists():
            for _ in range(5):
                try:
                    shutil.rmtree(col_dir)
                    print(f"✅ 已删除磁盘目录 {col_dir}")
                    break
                except PermissionError:
                    time.sleep(1.0)
                    gc.collect()

    # 复核
    client = QdrantClient(path=str(QDRANT_DIR))
    final = {c.name for c in client.get_collections().collections}
    print(f"\n清理后集合: {sorted(final)}")
    print(f"预期保留: {sorted(KEEP)}")
    assert KEEP <= final, "合法集合被误删,请立即检查!"
    assert not (set(GHOSTS) & final), "幽灵集合仍存在,清理不彻底!"
    print("\n✅ 清理完成,未误删合法集合。")
    client.close()


if __name__ == "__main__":
    main()
