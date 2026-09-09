# -*- coding: utf-8 -*-
"""补表 A.1 图块（17623 p16 色谱仪气路流程，含 6 张内嵌流程图）。"""
from __future__ import annotations
import json, shutil, sys, uuid, re
from pathlib import Path
BASE = Path(r"C:\Users\lizhihao\w2-knowledge-assistant")
sys.path.insert(0, str(BASE / "backend" / "src"))
from sf6_rag import pipeline  # noqa: E402

COLLECTION = "transformer_kb_v1"
FT_MODEL = str(BASE / "data" / "models" / "bge-m3-ft-v1")
STATIC_IMAGES = BASE / "data" / "mineru_out_upload" / "images"
IMG_SRC = BASE / "output" / "412e9721-b6ef-45fe-ba82-903138ddb357" / "GB_T_17623-2026" / "vlm" / "images" / "2876d1eecdc198322360eb0313e93665e889392f95799876b9368602610474d4.jpg"
FNAME = IMG_SRC.name

STATIC_IMAGES.mkdir(parents=True, exist_ok=True)
dest = STATIC_IMAGES / FNAME
if not dest.exists():
    shutil.copy2(str(IMG_SRC), str(dest))
print("img copied:", dest.exists(), dest)

res = json.load(open(BASE / "data" / "mineru_out_gb" / "gb_t_17623_2026" / "result.json", encoding="utf-8"))
if isinstance(res.get("results"), dict):
    for _v in res["results"].values():
        if isinstance(_v, dict) and "content_list" in _v:
            res = _v; break
cl_raw = res.get("content_list", "[]")
cl = json.loads(cl_raw) if isinstance(cl_raw, str) else cl_raw
table_item = next(it for it in cl if isinstance(it, dict) and it.get("type") == "table" and it.get("page_idx") == 15)
caption = "".join(table_item.get("table_caption") or [])
body = str(table_item.get("table_body") or "")
rows = re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S)
text_rows = []
for row in rows:
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
    parts = []
    for c in cells:
        c2 = re.sub(r"<[^>]+>", "", c).strip()
        if "<img" in c:
            parts.append("[流程图]")
        elif c2:
            parts.append(c2)
    if parts:
        text_rows.append(" | ".join(parts))
content = (caption + "\n" + "\n".join(text_rows)).strip()
print("content len:", len(content), "| rows:", len(text_rows))

ik = ["色谱仪气路流程", "流程图", "气路", "FID", "TCD", "镍触媒转化器", "分两次进样", "一次进样", "自动阀切换"]
qk = ["气路流程示意图", "色谱仪气路流程图", "气路流程是怎样的", "FID TCD 检测器如何连接", "几种气路流程", "表A.1 气路流程"]
item = {
    "text": content, "source": "GB_T_17623-2026.pdf", "page": 16,
    "section": "附录A", "semantic_type": "图语义",
    "important_kwd": ik, "question_kwd": qk,
    "document_id": "gb_t_17623_2026", "ingest_version": "20260908_tbl",
    "extract_mode": "mineru", "image_path": "/images/upload/" + FNAME,
    "mermaid": "", "type": "image",
    "point_id": uuid.uuid5(uuid.NAMESPACE_URL, "tbl_gb_t_17623_2026_p16_A1"),
}

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
model = BGEM3FlagModel(FT_MODEL, use_fp16=False)
out = model.encode([content], return_dense=True, return_sparse=True, return_colbert_vecs=False)
dense = out["dense_vecs"][0]
sw = out["lexical_weights"][0]
items_sorted = sorted(sw.items(), key=lambda kv: int(kv[0]))
client = QdrantClient(path=str(BASE / "data" / "qdrant"))
before = client.count(collection_name=COLLECTION).count
point = PointStruct(
    id=item["point_id"],
    vector={"dense": dense.tolist(),
            "sparse": SparseVector(indices=[int(k) for k, _ in items_sorted],
                                   values=[float(v) for _, v in items_sorted])},
    payload={k: v for k, v in item.items() if k != "point_id"},
)
client.upsert(collection_name=COLLECTION, points=[point])
after = client.count(collection_name=COLLECTION).count
print("upsert OK:", COLLECTION, before, "->", after)
client.close()
