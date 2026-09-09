# -*- coding: utf-8 -*-
"""MinerU 图语义块增量入库：3 份国标 image item → 图块 → transformer_kb_v1。

硬约束：
  - 原 88 文本块（fitz）不删不改不动（payload/chunk_id 零改动）
  - dense 用 bge-m3-ft-v1（与线上查询同向量空间）；sparse 用同模型 lexical_weights
  - 图块 point id = img_{doc}_{page}_{n}；type=image；extract_mode=mineru
  - 幂等：先删已有 img_ 前缀点，再 upsert

分两步：
  1. --dry-run：打印每份所有 image item 详情（img_path/page/caption/OCR/面积），
     用于确认过滤规则，不写库、不拷贝。
  2. 实际执行：过滤 → 拷贝图片 → 构建图块 → 编码 → 入库。

前置：先停 8011（释放 Qdrant 文件锁）。用法：
    PYTHONPATH=backend D:/an/envs/langchain/python.exe scripts/ingest_gb_images.py --dry-run
    PYTHONPATH=backend D:/an/envs/langchain/python.exe scripts/ingest_gb_images.py
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(r"C:\Users\lizhihao\w2-knowledge-assistant")
sys.path.insert(0, str(BASE / "backend" / "src"))

from sf6_rag import pipeline  # noqa: E402  复用 _make_q2q / estimate_tokens

OUT_ROOT = BASE / "data" / "mineru_out_gb"
COLLECTION = "transformer_kb_v1"
FT_MODEL = str(BASE / "data" / "models" / "bge-m3-ft-v1")
# 图片静态目录（对齐 pipeline._STATIC_UPLOAD_IMAGES，api.py 挂载 /images/upload）
STATIC_IMAGES = BASE / "data" / "mineru_out_upload" / "images"
# MinerU 输出根（图片源位于 output/{task_id}/{pdf}/vlm/images/）
MINERU_OUTPUT = BASE / "output"

# 图块命名 id 前缀（验收：/chunks 能按 document_id 查到 img_ 图块）
_IMG_CACHE: dict[str, str] = {}


def _locate_src(fname: str) -> str:
    """在 output/** 下按文件名定位图片源，命中即缓存。"""
    if fname in _IMG_CACHE:
        return _IMG_CACHE[fname]
    if MINERU_OUTPUT.exists():
        for cand in MINERU_OUTPUT.glob(f"**/{fname}"):
            _IMG_CACHE[fname] = str(cand)
            return str(cand)
    return ""


def _copy_image(img_path: str) -> str:
    """拷贝图片到静态目录，返回 web 路径 /images/upload/{fname}；失败返回空串。"""
    fname = Path(img_path.replace("\\", "/")).name
    src = _locate_src(fname)
    if not src:
        return ""
    try:
        STATIC_IMAGES.mkdir(parents=True, exist_ok=True)
        dest = STATIC_IMAGES / fname
        if not dest.exists():
            shutil.copy2(src, str(dest))
        return f"/images/upload/{fname}"
    except Exception:
        return ""

DOCS = [
    {"id": "gb_t_17623_2026", "source": "GB_T_17623-2026.pdf"},
    {"id": "gb_t_25438_2026", "source": "GB_T_25438-2026.pdf"},
    {"id": "gb_t_27743_2025", "source": "GB_T_27743-2025.pdf"},
]


def _bbox_area(img: dict) -> float:
    """从 image item 自带 bbox 算图面积（相对版面比例），失败返回 0。"""
    bbox = img.get("bbox")
    if bbox and len(bbox) == 4:
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox)
            # 面积（像素²）作为相对参考；过滤只看是否过小
            return max(0.0, x1 - x0) * max(0.0, y1 - y0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _is_keep(img: dict) -> tuple[bool, str]:
    """过滤封面 logo / 水印 / 无信息图。返回 (是否保留, 理由)。"""
    page = img.get("page") or 0
    caption = (img.get("caption") or "").strip()
    content = (img.get("content") or "").strip()
    context = (img.get("context") or "").strip()
    # 水印/无信息图：无 caption、无 OCR 文本、无相邻图题/标引说明
    if not caption and not content and not context:
        return False, "无caption且无OCR/上下文(疑似水印)"
    # 封面/前页 logo：无 caption、无上下文、OCR 文本极短
    if page <= 2 and not caption and not context and len(content) <= 8:
        return False, f"封面/前页logo(p{page})"
    return True, ""


def load_images(doc: dict) -> tuple[dict, list[dict]]:
    """读 result.json → (result_images, image_items)。

    图题与标引序号说明通常不在 image 元素的 image_caption 字段，
    而是作为独立 text 元素紧跟 image 之后。这里向后扫描同页相邻元素，
    把「图 X.X …」标题 + 「标引序号说明…」并入 caption/context，
    确保「取气装置结构图」这类查询能命中图块。
    """
    result_file = OUT_ROOT / doc["id"] / "result.json"
    if not result_file.exists():
        raise FileNotFoundError(f"缺失 {result_file}")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    result_images: dict = {}
    if isinstance(result.get("images"), dict):
        result_images = result["images"]
    cl_raw = result.get("content_list", "[]")
    cl = json.loads(cl_raw) if isinstance(cl_raw, str) else cl_raw

    items = []
    for i, item in enumerate(cl):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "image":
            continue
        idx = item.get("page_idx")
        if not isinstance(idx, (int, float)):
            continue
        img_path = (item.get("img_path") or "").strip()
        if not img_path:
            continue

        caption = "".join(item.get("image_caption") or []).strip()
        # 向后扫描同页相邻 text/list：找图题（图 X.X）与标引序号说明
        trailing: list[str] = []
        for j in range(i + 1, len(cl)):
            nxt = cl[j]
            if not isinstance(nxt, dict):
                continue
            if nxt.get("page_idx") != idx:
                break  # 跨页停止
            ntype = nxt.get("type")
            if ntype in ("header", "footer", "page_number", "image"):
                break
            # list 元素的正文在 list_items（text 字段常为空）
            if ntype == "list":
                txt = "".join(nxt.get("list_items") or []).strip()
            else:
                txt = (nxt.get("text") or "").strip()
            if not txt:
                continue
            trailing.append(txt)
            # 收集到图题就停（图题通常是最后一段，标引说明在其前）
            if txt.startswith("图") and len(txt) < 60:
                break
        context_text = "；".join(trailing[:8])
        # 若无 image_caption，用 trailing 里的「图 X.X」作 caption
        if not caption:
            for t in trailing:
                if t.startswith("图"):
                    caption = t
                    break

        items.append({
            "img_path": img_path,
            "page": int(idx) + 1,
            "content": item.get("content") or "",
            "caption": caption,
            "context": context_text,
        })
    return result_images, items


def dry_run() -> None:
    for doc in DOCS:
        try:
            result_images, items = load_images(doc)
        except FileNotFoundError as exc:
            print(f"[skip] {doc['id']}: {exc}")
            continue
        print(f"\n=== {doc['id']} ({doc['source']}) image_items={len(items)} ===")
        for i, img in enumerate(items):
            area = _bbox_area(img)
            keep, reason = _is_keep(img)
            fname = Path(img["img_path"].replace("\\", "/")).name
            print(f"  [{i}] {'KEEP' if keep else 'DROP'} p{img['page']} area={area:.0f} "
                  f"caption={img['caption'][:20]!r} ocr={img['content'][:30]!r} "
                  f"ctx={img['context'][:40]!r} file={fname}")
            if not keep:
                print(f"       -> {reason}")


def build_chunks(doc: dict, result_images: dict, items: list[dict]) -> list[dict]:
    chunks = []
    for n, img in enumerate(items, start=1):
        keep, _ = _is_keep(img)
        if not keep:
            continue
        fname = Path(img["img_path"].replace("\\", "/")).name
        web_path = _copy_image(img["img_path"])
        if not web_path:
            print(f"  [warn] {doc['id']} p{img['page']} 图片拷贝失败，跳过: {fname}")
            continue
        # content = 图题 + 图内OCR + 相邻标引序号说明（可检索即可）
        content = _image_semantic_content_enhanced(img)
        mermaid = _extract_mermaid(img.get("content") or "")
        ik, qk = pipeline._make_q2q(content, img.get("caption") or "图", doc["source"])
        # Qdrant local mode 强制点 id 为 UUID；用 uuid5 确定性映射
        # img_{doc}_{page}_{n} → 固定 UUID（幂等：重跑同一图块 id 不变）
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"img_{doc['id']}_{img['page']}_{n}"))
        chunks.append({
            "chunk_id": f"img_{doc['id']}_{img['page']}_{n}",
            "point_id": pid,
            "content": content,
            "source": doc["source"],
            "page": img["page"],
            "type": "image",
            "section": img.get("caption") or "图",
            "semantic_type": "图语义",
            "important_kwd": ik,
            "question_kwd": qk,
            "document_id": doc["id"],
            "ingest_version": time.strftime("%Y%m%d_mineru"),
            "extract_mode": "mineru",
            "image_path": web_path,
            "mermaid": mermaid,
            "char_len": len(content),
            "token_len": pipeline.estimate_tokens(content),
        })
    return chunks


def _extract_mermaid(text: str) -> str:
    """从图内 OCR 文本提取 mermaid 流程图（无则空串），对齐 v8 脚本逻辑。"""
    m = re.search(r"```mermaid\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else ""


def _image_semantic_content_enhanced(img: dict) -> str:
    """图语义块 content：图题 + 图内 OCR + 相邻标引序号说明。

    对齐 pipeline._image_semantic_content，额外并入 load_images 扫到的
    相邻 text（标引序号说明 + 图题），确保「取气装置结构图」能检索命中。
    """
    caption = (img.get("caption") or "").strip()
    body = (img.get("content") or "").strip()
    ctx = (img.get("context") or "").strip()
    parts = []
    if caption:
        parts.append(caption)
    if body:
        # 去掉 mermaid 代码块，正文只保留可读描述
        body_clean = re.sub(r"```mermaid.*?```", "", body, flags=re.S).strip()
        if body_clean:
            parts.append(body_clean.replace("\n", "；"))
    # context 里已含图题与标引说明，去重后追加
    for seg in ctx.split("；"):
        seg = seg.strip()
        if seg and seg not in parts:
            parts.append(seg)
    return "；".join(parts) or "图（无文字描述）"


def build_embed_text(c: dict) -> str:
    parts = [c["content"]]
    parts.extend(c.get("important_kwd") or [])
    parts.extend(c.get("question_kwd") or [])
    return "\n".join(parts)


def ingest() -> None:
    from FlagEmbedding import BGEM3FlagModel
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, SparseVector

    print(f"加载 bge-m3-ft-v1 ... {FT_MODEL}")
    model = BGEM3FlagModel(FT_MODEL, use_fp16=False)

    all_chunks = []
    for doc in DOCS:
        result_images, items = load_images(doc)
        chunks = build_chunks(doc, result_images, items)
        print(f"[{doc['id']}] 有效图块 {len(chunks)} / image_items {len(items)}")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("无有效图块，终止。")
        return
    print(f"总图块 {len(all_chunks)} 条，编码中 ...")
    texts = [build_embed_text(c) for c in all_chunks]
    out = model.encode(texts, return_dense=True, return_sparse=True,
                       return_colbert_vecs=False, batch_size=8)
    dense = out["dense_vecs"]
    sparse = out["lexical_weights"]
    print(f"dense {dense.shape}, sparse 词数 "
          f"{min(len(w) for w in sparse)}~{max(len(w) for w in sparse)}")

    client = QdrantClient(path=str(BASE / "data" / "qdrant"))

    # 幂等：删除已有 mineru 图块（extract_mode=mineru），不动 88 个 fitz 文本块
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    existing = client.count(collection_name=COLLECTION).count
    print(f"入库前 {COLLECTION} 点数 = {existing}")
    mineru_filter = Filter(must=[FieldCondition(key="extract_mode", match=MatchValue(value="mineru"))])
    # 统计并删除已有 mineru 点
    existing_mineru = client.count(collection_name=COLLECTION, count_filter=mineru_filter).count
    if existing_mineru:
        print(f"删除已有 mineru 图块 {existing_mineru} 个 ...")
        client.delete(collection_name=COLLECTION, points_selector=mineru_filter)

    points = []
    for i, c in enumerate(all_chunks):
        sw = sparse[i]
        items_sorted = sorted(sw.items(), key=lambda kv: int(kv[0]))
        points.append(PointStruct(
            id=c["point_id"],
            vector={
                "dense": dense[i].tolist(),
                "sparse": SparseVector(
                    indices=[int(k) for k, _ in items_sorted],
                    values=[float(v) for _, v in items_sorted],
                ),
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
                "document_id": c["document_id"],
                "ingest_version": c["ingest_version"],
                "extract_mode": c["extract_mode"],
                "image_path": c["image_path"],
                "mermaid": c["mermaid"],
            },
        ))
    client.upsert(collection_name=COLLECTION, points=points)
    after = client.count(collection_name=COLLECTION).count
    print(f"[OK] 入库完成: {COLLECTION} = {after} 条（新增图块 {len(points)}，原 88 文本块保留）")

    # 打印图块清单（供验收 a）
    for c in all_chunks:
        print(f"  {c['chunk_id']} ({c['point_id'][:8]}…) | p{c['page']} | {c['section'][:20]!r} | {c['image_path']}")
    client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只打印 image item 详情，不写库")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        ingest()


if __name__ == "__main__":
    main()
