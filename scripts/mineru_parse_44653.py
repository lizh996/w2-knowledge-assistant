"""mineru 解析 GB_T_44653（带 page_idx）——修复版 v2。

修复（对照 mineru/cli/api_client.py 官方协议）：
1. 状态字段：轮询读顶层 status（非 state.status）；完成值 completed（非 done）
2. 轮询 URL：用提交返回的 status_url（非硬拼 /tasks/{id}）
3. 超时：默认 7200s（2h，CPU VLM 15页约 82 分钟），可 --timeout 覆盖
4. 文件名：content_list.json / middle.json（去掉错误的 "..." 字面量）

用法：D:/an/envs/mineru/python.exe scripts/mineru_parse_44653.py [--timeout 7200]
前置：mineru-api 已启动（--host 127.0.0.1 --port 58567）
"""
from __future__ import annotations

import argparse
import json
import time
import requests
from pathlib import Path

API = "http://127.0.0.1:58567"
PDF = r"C:\\Users\\lizhihao\\w1-day5\\device-rag-44653\\data\\source_docs\\GB_T_44653-2024_六氟化硫气体现场循环再利用导则.pdf"
OUT_DIR = Path(r"C:\\Users\\lizhihao\\w1-day5\\device-rag-44653\\data\\mineru_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MinerU 官方状态值（mineru/cli/api_client.py L966/L981）
PENDING = "pending"
PROCESSING = "processing"
COMPLETED = "completed"


def submit_task() -> dict:
    """提交解析任务，返回 {task_id, status_url, result_url}。"""
    with open(PDF, "rb") as f:
        files = {"files": f}
        data = {
            "lang_list": "ch",
            "backend": "vlm-auto-engine",          # 无 GPU → VLM 引擎
            "return_content_list": "true",          # ⭐ 必须！否则无 page_idx
            "return_middle_json": "true",           # ⭐ 必须！否则无表格/页码
        }
        r = requests.post(f"{API}/tasks", files=files, data=data, timeout=60)
    r.raise_for_status()
    payload = r.json()
    task_id = payload.get("task_id")
    status_url = payload.get("status_url")
    result_url = payload.get("result_url")
    if not (isinstance(task_id, str) and isinstance(status_url, str)):
        raise RuntimeError(f"提交响应缺字段: {json.dumps(payload, ensure_ascii=False)[:200]}")
    print(f"✅ 任务已提交: {task_id}")
    print(f"   状态查询: {status_url}")
    return {"task_id": task_id, "status_url": status_url, "result_url": result_url}


def poll_task(submit: dict, timeout: int = 7200) -> dict:
    """轮询 status_url 直到 completed（修复：顶层 status 字段）。"""
    start = time.time()
    last_status = None
    while time.time() - start < timeout:
        r = requests.get(submit["status_url"], timeout=30)
        r.raise_for_status()
        payload = r.json()
        status = payload.get("status")            # ⭐ 顶层字段（官方协议）
        elapsed = int(time.time() - start)
        if status != last_status:
            print(f"  [{time.strftime('%H:%M:%S')}] 状态: {status} ({elapsed}s)")
            last_status = status
        if status == COMPLETED:
            return payload
        if status not in (PENDING, PROCESSING):
            raise RuntimeError(f"任务失败: {json.dumps(payload, ensure_ascii=False)[:300]}")
        time.sleep(30)
    raise TimeoutError(f"任务超时（>{timeout}s）")


def save_result(result: dict) -> None:
    """保存 result.json + content_list.json + middle.json（标准文件名）。"""
    (OUT_DIR / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md = result.get("md_content", "")
    (OUT_DIR / "GB_T_44653-2024_六氟化硫气体现场循环再利用导则.md").write_text(md, encoding="utf-8")

    # content_list / middle_json 是 JSON 字符串 → 二次解析
    cl = json.loads(result.get("content_list", "[]"))
    mj = json.loads(result.get("middle_json", "{}"))
    (OUT_DIR / "content_list.json").write_text(
        json.dumps(cl, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "middle.json").write_text(
        json.dumps(mj, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    types = Counter(el.get("type") for el in cl)
    with_page = sum(1 for el in cl if "page_idx" in el)
    print(f"✅ 产物已保存: {OUT_DIR}")
    print(f"   content_list: {len(cl)} 元素（{with_page} 带 page_idx）")
    print(f"   类型分布: {dict(types)}")
    print(f"   middle_json: pdf_info {len(mj.get('pdf_info', []))} 页")
    print(f"   markdown: {len(md)} 字符")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=7200,
                        help="轮询超时秒数（CPU VLM 15页约 82 分钟，默认 7200）")
    args = parser.parse_args()

    submit = submit_task()
    result = poll_task(submit, timeout=args.timeout)
    save_result(result)
    print("PARSED_OK")


if __name__ == "__main__":
    main()
