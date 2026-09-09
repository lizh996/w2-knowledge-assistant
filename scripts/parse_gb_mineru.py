# -*- coding: utf-8 -*-
"""MinerU 串行解析 3 份国标，保存完整 result（含 content_list/middle_json/images）。

对齐 scripts/mineru_parse_44653.py 协议 + pipeline._mineru_parse：
  - POST /tasks（lang_list=ch, backend=vlm-auto-engine,
    return_content_list=true, return_middle_json=true）
  - 轮询 status_url 顶层 status 到 completed
  - GET result_url 拿完整 result（3.4.5 新格式兼容 results[key]）
  - 保存 result.json / content_list.json / middle.json 到 data/mineru_out_gb/{doc}/

串行跑（GPU 单任务并发会 tensor 崩溃）；幂等：已有 result.json 的文档跳过。

用法：D:/an/envs/mineru/python.exe scripts/parse_gb_mineru.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API = "http://127.0.0.1:58567"
BASE = Path(r"C:\Users\lizhihao\w2-knowledge-assistant")
OUT_ROOT = BASE / "data" / "mineru_out_gb"

DOCS = [
    {"id": "gb_t_17623_2026", "pdf": BASE / "data" / "GB_T_17623-2026.pdf"},
    {"id": "gb_t_25438_2026", "pdf": BASE / "data" / "GB_T_25438-2026.pdf"},
    {"id": "gb_t_27743_2025", "pdf": BASE / "data" / "GB_T_27743-2025.pdf"},
]

PENDING, PROCESSING, COMPLETED = "pending", "processing", "completed"
TIMEOUT = 7200  # 每份最多 2h（GPU 预计 20-60 分钟）


def _warn_gpu_backend() -> None:
    """提交前看一眼当前 VLM 引擎日志（供排障，不阻塞）。"""
    try:
        r = requests.get(f"{API}/health", timeout=10)
        h = r.json()
        print(f"[health] processing_tasks={h.get('processing_tasks')} "
              f"queued_tasks={h.get('queued_tasks')}")
    except Exception as exc:
        print(f"[health] 不可达: {exc}")


def parse_one(doc: dict) -> dict:
    out_dir = OUT_ROOT / doc["id"]
    result_file = out_dir / "result.json"
    if result_file.exists():
        print(f"[skip] {doc['id']} 已有 result.json，跳过")
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = doc["pdf"]
    if not pdf.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf}")

    print(f"\n=== [{doc['id']}] 提交 {pdf.name} ===")
    with open(pdf, "rb") as f:
        r = requests.post(
            f"{API}/tasks",
            files={"files": f},
            data={
                "lang_list": "ch",
                "backend": "vlm-auto-engine",
                "return_content_list": "true",
                "return_middle_json": "true",
            },
            timeout=120,
        )
    r.raise_for_status()
    payload = r.json()
    task_id = payload.get("task_id")
    status_url = payload.get("status_url")
    result_url = payload.get("result_url")
    if not (isinstance(task_id, str) and isinstance(status_url, str)):
        raise RuntimeError(f"提交响应缺字段: {json.dumps(payload, ensure_ascii=False)[:300]}")
    print(f"   task_id={task_id}")

    # 轮询
    start = time.time()
    last_status = None
    while time.time() - start < TIMEOUT:
        r2 = requests.get(status_url, timeout=30)
        r2.raise_for_status()
        p2 = r2.json()
        status = p2.get("status")
        elapsed = int(time.time() - start)
        if status != last_status:
            print(f"   [{time.strftime('%H:%M:%S')}] status={status} ({elapsed}s)")
            last_status = status
        if status == COMPLETED:
            break
        if status not in (PENDING, PROCESSING):
            raise RuntimeError(f"任务失败: {json.dumps(p2, ensure_ascii=False)[:400]}")
        time.sleep(30)
    else:
        raise TimeoutError(f"{doc['id']} 解析超时 >{TIMEOUT}s")

    # 取结果
    r3 = requests.get(result_url, timeout=180)
    r3.raise_for_status()
    result = r3.json()

    # 3.4.5 新格式：result["results"][key] 下才有内容
    if "results" in result and isinstance(result["results"], dict):
        for _v in result["results"].values():
            if isinstance(_v, dict) and "content_list" in _v:
                result = _v
                break

    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    cl_raw = result.get("content_list", "[]")
    cl = json.loads(cl_raw) if isinstance(cl_raw, str) else cl_raw
    mj_raw = result.get("middle_json", "{}")
    mj = json.loads(mj_raw) if isinstance(mj_raw, str) else mj_raw
    (out_dir / "content_list.json").write_text(
        json.dumps(cl, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "middle.json").write_text(
        json.dumps(mj, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    types = Counter(el.get("type") for el in cl if isinstance(el, dict))
    images = sum(1 for el in cl if isinstance(el, dict) and el.get("type") == "image")
    print(f"   [done] content_list={len(cl)} 元素, 类型分布={dict(types)}, image={images}")
    return {"id": doc["id"], "images": images}


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _warn_gpu_backend()
    t0 = time.time()
    summary = []
    for doc in DOCS:
        try:
            summary.append(parse_one(doc))
        except Exception as exc:
            print(f"[FAIL] {doc['id']}: {type(exc).__name__}: {exc}")
            raise
    total = int(time.time() - t0)
    print(f"\n=== 全部完成，总耗时 {total}s ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
