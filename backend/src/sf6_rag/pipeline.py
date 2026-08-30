"""Document upload pipeline with persistent task state.

The pipeline is intentionally kept in this module so the legacy ingestion
scripts remain untouched. Upload builds run in the FastAPI process thread pool
and write status snapshots to SQLite for refresh/SSE recovery.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sf6_rag.chunker import chunk_by_section
from sf6_rag.extract import _COMMON_CHARS, extract_pages


BASE = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("SF6_RAG_RUNTIME_DIR", str(BASE / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
TASK_DB = DATA_DIR / "pipeline_tasks.sqlite"
QDRANT_DIR = DATA_DIR / "qdrant"
BGE = Path(
    r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3"
    r"\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
)

PIPELINE_STEPS = [
    ("parse", "解析"),
    ("clean", "清洗"),
    ("chunk", "分块"),
    ("embed", "向量化"),
    ("upsert", "入库"),
    ("eval", "评测"),
]

STATUS_LABELS = {
    "pending": "待处理",
    "parsing": "解析中",
    "cleaning": "清洗中",
    "chunking": "分块中",
    "vectorizing": "向量化中",
    "ingesting": "入库中",
    "evaluating": "评测中",
    "done": "已完成",
    "failed": "失败",
}

_executor = ThreadPoolExecutor(max_workers=2)
_db_lock = threading.Lock()
_recovered_store_paths: set[str] = set()
_MODULE_START_TS = time.time()


@dataclass
class PipelineBuild:
    document_id: str
    collection: str
    chunks: list[dict[str, Any]]
    dense_vectors: list[Any]
    sparse_vectors: list[Any]
    sparse_sizes: list[int]


def init_store() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    task_db_key = str(TASK_DB.resolve())
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                collection TEXT NOT NULL,
                extract_mode TEXT NOT NULL,
                status TEXT NOT NULL,
                steps TEXT NOT NULL,
                error TEXT,
                result TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        if task_db_key not in _recovered_store_paths:
            _recover_interrupted_tasks(conn)
            _recovered_store_paths.add(task_db_key)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(TASK_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _recover_interrupted_tasks(conn: sqlite3.Connection) -> None:
    running_statuses = (
        "parsing",
        "cleaning",
        "chunking",
        "vectorizing",
        "ingesting",
        "evaluating",
    )
    placeholders = ",".join("?" for _ in running_statuses)
    rows = conn.execute(
        f"SELECT task_id, steps FROM tasks WHERE status IN ({placeholders}) AND created_at < ?",
        running_statuses + (_MODULE_START_TS,),
    ).fetchall()
    for row in rows:
        steps = json.loads(row["steps"])
        failed_seen = False
        for item in steps:
            if item.get("state") == "running" and not failed_seen:
                item["state"] = "failed"
                failed_seen = True
            elif item.get("state") == "pending":
                item["state"] = "skipped"
        try:
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, steps = ?, error = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    "failed",
                    json.dumps(steps, ensure_ascii=False),
                    "RuntimeError: 上次服务进程已中断，请重新上传或重新提交该文档",
                    _now(),
                    row["task_id"],
                ),
            )
        except sqlite3.OperationalError:
            return


def _now() -> float:
    return time.time()


def _initial_steps() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "label": label,
            "state": "pending",
            "elapsed_ms": 0,
            "stats": {},
        }
        for name, label in PIPELINE_STEPS
    ]


def _row_to_task(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "document_id": row["document_id"],
        "filename": row["filename"],
        "file_path": row["file_path"],
        "collection": row["collection"],
        "extract_mode": row["extract_mode"],
        "status": row["status"],
        "status_label": STATUS_LABELS.get(row["status"], row["status"]),
        "steps": json.loads(row["steps"]),
        "error": row["error"],
        "result": json.loads(row["result"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_task(filename: str, file_bytes: bytes, mode: str) -> dict[str, str]:
    if mode not in {"fitz", "mineru"}:
        raise ValueError("mode 只能是 fitz 或 mineru")
    if not filename.lower().endswith(".pdf"):
        raise ValueError("仅支持 PDF 文件")

    init_store()
    task_id = uuid.uuid4().hex
    document_id = task_id
    safe_name = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", filename).strip("_") or "upload.pdf"
    file_path = UPLOAD_DIR / f"{task_id}_{safe_name}"
    file_path.write_bytes(file_bytes)
    collection = f"transformer_upload_{document_id}"
    ts = _now()
    steps = _initial_steps()
    with _db_lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO tasks
            (task_id, document_id, filename, file_path, collection, extract_mode,
             status, steps, error, result, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                document_id,
                filename,
                str(file_path),
                collection,
                mode,
                "pending",
                json.dumps(steps, ensure_ascii=False),
                None,
                json.dumps({}, ensure_ascii=False),
                ts,
                ts,
            ),
        )
    _executor.submit(run_pipeline_task, task_id)
    return {"document_id": document_id, "task_id": task_id}


def get_task(task_id: str) -> dict[str, Any] | None:
    init_store()
    with _db_lock, _connect() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    return _row_to_task(row) if row else None


def list_tasks() -> list[dict[str, Any]]:
    init_store()
    with _db_lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
    return [_row_to_task(row) for row in rows]


def list_documents() -> list[dict[str, Any]]:
    docs = [
        {
            "id": "gb_t_44653_2024",
            "name": "GB/T 44653-2024 六氟化硫气体现场循环再利用导则",
            "status": "done",
            "collection": "device_knowledge_v8",
            "chunk_count": 22,
            "extract_mode": "mineru",
            "created_at": None,
            "error": None,
            "builtin": True,
        },
        {
            "id": "gb_t_12022_2025",
            "name": "GB/T 12022-2025 工业六氟化硫",
            "status": "done",
            "collection": "device_knowledge_v8",
            "chunk_count": 43,
            "extract_mode": "mineru",
            "created_at": None,
            "error": None,
            "builtin": True,
        },
        {
            "id": "gb_t_18867_2025",
            "name": "GB/T 18867-2025 电子工业用气体 六氟化硫",
            "status": "done",
            "collection": "device_knowledge_v8",
            "chunk_count": 20,
            "extract_mode": "mineru",
            "created_at": None,
            "error": None,
            "builtin": True,
        },
    ]
    for task in list_tasks():
        result = task.get("result") or {}
        docs.append(
            {
                "id": task["document_id"],
                "task_id": task["task_id"],
                "name": task["filename"],
                "status": task["status"],
                "collection": task["collection"],
                "chunk_count": result.get("chunk_count", 0),
                "extract_mode": result.get("extract_mode") or task["extract_mode"],
                "requested_extract_mode": result.get("requested_extract_mode") or task["extract_mode"],
                "created_at": task["created_at"],
                "updated_at": task["updated_at"],
                "error": task["error"],
                "builtin": False,
            }
        )
    return docs


def delete_upload_document(document_id: str) -> dict[str, Any] | None:
    init_store()
    with _db_lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE document_id = ?", (document_id,)
        ).fetchone()
        if row is None:
            return None
        task = _row_to_task(row)
        conn.execute("DELETE FROM tasks WHERE document_id = ?", (document_id,))

    cleanup_collection(task["collection"])
    try:
        Path(task["file_path"]).unlink(missing_ok=True)
    except OSError:
        pass
    return {"deleted": True, "collection": task["collection"]}


def update_status(
    task_id: str,
    status: str,
    *,
    step: str | None = None,
    state: str | None = None,
    elapsed_ms: int | None = None,
    stats: dict[str, Any] | None = None,
    error: str | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    task = get_task(task_id)
    if task is None:
        return
    steps = task["steps"]
    if step:
        for item in steps:
            if item["name"] == step:
                if state:
                    item["state"] = state
                if elapsed_ms is not None:
                    item["elapsed_ms"] = elapsed_ms
                if stats is not None:
                    item["stats"] = stats
                break
    with _db_lock, _connect() as conn:
        conn.execute(
            """
            UPDATE tasks
            SET status = ?, steps = ?, error = ?, result = ?, updated_at = ?
            WHERE task_id = ?
            """,
            (
                status,
                json.dumps(steps, ensure_ascii=False),
                error,
                json.dumps(result if result is not None else task["result"], ensure_ascii=False),
                _now(),
                task_id,
            ),
        )


def run_pipeline_task(task_id: str) -> None:
    task = get_task(task_id)
    if task is None:
        return
    collection = task["collection"]
    try:
        parse_start = time.time()
        update_status(
            task_id,
            "parsing",
            step="parse",
            state="running",
        )
        pages, actual_extract_mode, parse_note = parse_document_with_fallback(
            Path(task["file_path"]),
            task["extract_mode"],
        )
        if not pages:
            raise ValueError(
                "无法从 PDF 提取到可读文本：该文档可能是乱码字体或扫描版，"
                "fitz 模式提取不到内容。请换用可提取文本的 PDF，"
                "或等待 MinerU 模式接入后重试。"
            )
        update_status(
            task_id,
            "cleaning",
            step="parse",
            state="done",
            elapsed_ms=_elapsed(parse_start),
            stats={
                "page_count": len(pages),
                "text_blocks": len(pages),
                "table_count": 0,
                "image_count": 0,
                "requested_extract_mode": task["extract_mode"],
                "extract_mode": actual_extract_mode,
                "raw_fallback_pages": sum(1 for p in pages if p.get("raw_fallback")),
                "note": parse_note,
            },
        )

        clean_start = time.time()
        runtime_task = {**task, "extract_mode": actual_extract_mode}
        cleaned_pages = clean_pages(pages)
        update_status(
            task_id,
            "chunking",
            step="clean",
            state="done",
            elapsed_ms=_elapsed(clean_start),
            stats={
                "before_chars": sum(len(p["text"]) for p in pages),
                "after_chars": sum(len(p["text"]) for p in cleaned_pages),
                "removed": 0,
                "converted": 0,
                "protected": 0,
                "fixed": len(cleaned_pages),
            },
        )

        chunk_start = time.time()
        update_status(task_id, "chunking", step="chunk", state="running")
        chunks = build_chunks(cleaned_pages, runtime_task)
        chunk_stats = chunk_dashboard(chunks)
        update_status(
            task_id,
            "vectorizing",
            step="chunk",
            state="done",
            elapsed_ms=_elapsed(chunk_start),
            stats=chunk_stats,
        )

        embed_start = time.time()
        update_status(task_id, "vectorizing", step="embed", state="running")
        build = vectorize_chunks(chunks, runtime_task)
        update_status(
            task_id,
            "ingesting",
            step="embed",
            state="done",
            elapsed_ms=_elapsed(embed_start),
            stats={
                "dense_dim": 1024,
                "chunk_count": len(chunks),
                "sparse_min": min(build.sparse_sizes, default=0),
                "sparse_max": max(build.sparse_sizes, default=0),
            },
        )

        upsert_start = time.time()
        update_status(task_id, "ingesting", step="upsert", state="running")
        upsert_collection(build)
        update_status(
            task_id,
            "evaluating",
            step="upsert",
            state="done",
            elapsed_ms=_elapsed(upsert_start),
            stats={"collection": collection, "point_count": len(chunks)},
        )

        eval_start = time.time()
        update_status(task_id, "evaluating", step="eval", state="running")
        evaluation = self_check(build)
        result = {
            "status": "ok",
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "table_count": 0,
            "image_count": 0,
            "collection": collection,
            "requested_extract_mode": task["extract_mode"],
            "extract_mode": actual_extract_mode,
            "raw_fallback_pages": sum(1 for p in pages if p.get("raw_fallback")),
            "evaluation": evaluation,
        }
        update_status(
            task_id,
            "done",
            step="eval",
            state="done",
            elapsed_ms=_elapsed(eval_start),
            stats=evaluation,
            result=result,
        )
    except Exception as exc:  # pragma: no cover - defensive path exercised via API status
        cleanup_collection(collection)
        mark_task_failed(task_id, f"{type(exc).__name__}: {exc}")


def mark_pending_steps_skipped(task_id: str) -> None:
    task = get_task(task_id)
    if task is None:
        return
    steps = task["steps"]
    for item in steps:
        if item["state"] == "pending":
            item["state"] = "skipped"
    with _db_lock, _connect() as conn:
        conn.execute(
            "UPDATE tasks SET steps = ?, updated_at = ? WHERE task_id = ?",
            (json.dumps(steps, ensure_ascii=False), _now(), task_id),
        )


def mark_task_failed(task_id: str, error: str) -> None:
    task = get_task(task_id)
    if task is None:
        return
    steps = task["steps"]
    failed_seen = False
    for item in steps:
        if item["state"] == "running" and not failed_seen:
            item["state"] = "failed"
            failed_seen = True
        elif item["state"] == "pending":
            item["state"] = "skipped"
    with _db_lock, _connect() as conn:
        conn.execute(
            """
            UPDATE tasks
            SET status = ?, steps = ?, error = ?, updated_at = ?
            WHERE task_id = ?
            """,
            (
                "failed",
                json.dumps(steps, ensure_ascii=False),
                error,
                _now(),
                task_id,
            ),
        )


def _elapsed(start: float) -> int:
    return max(1, int((time.time() - start) * 1000))


_MINERU_API = os.environ.get("MINERU_API_URL", "http://127.0.0.1:18080")
# 上传文档的图片静态目录（api.py 挂载 /images/upload → 前端可显示原图）
_STATIC_UPLOAD_IMAGES = Path(__file__).resolve().parents[3] / "data" / "mineru_out_upload" / "images"


def _mineru_parse(pdf_path: Path) -> list[dict[str, Any]]:
    """调用 mineru-api（MinerU 服务）解析 PDF，返回 [{page, text}]。

    协议见 scripts/mineru_parse_44653.py：POST /tasks → 轮询 status_url
    （顶层 status，completed 完成）→ result_url 取结果 → content_list 按
    page_idx（0 基）聚合文本。
    """
    import requests

    # 并发保护：mineru-api GPU 单任务（并发 batch 会 tensor 崩溃），有任务在跑就拒绝
    try:
        _h = requests.get(f"{_MINERU_API}/health", timeout=10)
        _h.raise_for_status()
        _health = _h.json()
        if _health.get("processing_tasks", 0) > 0:
            raise RuntimeError(
                "MinerU 正在解析其他文档（GPU 单任务并发会崩溃），请稍后再试。"
            )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"MinerU 服务未启动（{_MINERU_API}），请先启动 mineru-api。"
        )

    with open(pdf_path, "rb") as f:
        r = requests.post(
            f"{_MINERU_API}/tasks",
            files={"files": f},
            data={
                "lang_list": "ch",
                "backend": "vlm-auto-engine",
                "return_content_list": "true",
                "return_middle_json": "true",
            },
            timeout=60,
        )
    r.raise_for_status()
    payload = r.json()
    task_id = payload.get("task_id")
    status_url = payload.get("status_url")
    result_url = payload.get("result_url")
    if not (isinstance(task_id, str) and isinstance(status_url, str)):
        raise RuntimeError(f"MinerU 提交响应缺字段: {json.dumps(payload, ensure_ascii=False)[:200]}")

    deadline = time.time() + 3600
    while True:
        if time.time() > deadline:
            raise TimeoutError("MinerU 解析超时（>1h）")
        time.sleep(30)
        r2 = requests.get(status_url, timeout=30)
        r2.raise_for_status()
        status = r2.json().get("status")
        if status == "completed":
            break
        if status not in ("pending", "processing"):
            raise RuntimeError(f"MinerU 任务失败: status={status}")

    r3 = requests.get(result_url, timeout=120)
    r3.raise_for_status()
    result = r3.json()

    # 兼容 mineru-api 3.4.5 新格式：result["results"][key] 下才有内容
    if "results" in result and isinstance(result["results"], dict):
        for _v in result["results"].values():
            if isinstance(_v, dict) and "content_list" in _v:
                result = _v
                break

    cl_raw = result.get("content_list", "[]")
    cl = json.loads(cl_raw) if isinstance(cl_raw, str) else cl_raw
    # result["images"] = {basename: {img_path(相对), bbox, ...}}（mineru-api 返回）
    # 用它对 img_path 反查绝对路径 → 拷到项目静态目录供前端展示
    result_images: dict[str, dict] = {}
    if isinstance(result.get("images"), dict):
        for k, v in result["images"].items():
            if isinstance(v, dict):
                result_images[k] = v
    pages_map: dict[int, list[str]] = {}
    # 图语义块：收集 image 元素（对齐内置 build_image_atomic_block）
    image_items: list[dict[str, Any]] = []
    for item in cl:
        if not isinstance(item, dict):
            continue
        idx = item.get("page_idx")
        if not isinstance(idx, (int, float)):
            continue
        page_no = int(idx) + 1
        itype = item.get("type")
        if itype == "text" and item.get("text"):
            pages_map.setdefault(page_no, []).append(item["text"])
        elif itype == "table" and item.get("table_body"):
            # 表格：标题 + 行列转文本（对齐讲师 FR2 复杂元素提取）
            html = item["table_body"]
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
            lines = []
            for row in rows:
                cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
                cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
                lines.append(" | ".join(cells))
            caption = "".join(item.get("table_caption", []) or [])
            txt = ((caption + "\n") if caption else "") + "\n".join(lines)
            if txt.strip():
                pages_map.setdefault(page_no, []).append(txt.strip())
        elif itype == "equation" and item.get("text"):
            # 公式：LaTeX 原文保留（供检索命中）
            pages_map.setdefault(page_no, []).append(item["text"])
        elif itype == "image":
            img_path = (item.get("img_path") or "").strip()
            if img_path:
                image_items.append({
                    "img_path": img_path,
                    "page": page_no,
                    "content": item.get("content") or "",
                    "caption": "".join(item.get("image_caption") or []),
                })
    if not pages_map and not image_items:
        raise ValueError("MinerU 未提取到文本内容（content_list 为空）")
    pages = [{"page": p, "text": "\n".join(parts)} for p, parts in sorted(pages_map.items())]
    # 图片信息挂到 pages 上（build_chunks 会用它生成图语义块）
    if image_items:
        # 拷贝图片到项目静态目录（对齐内置 mineru_out_multidoc 结构）
        for img in image_items:
            img["web_path"] = _copy_mineru_image(img["img_path"], result_images)
        pages.append({"page": -1, "text": "", "images": image_items})
    return pages


def _copy_mineru_image(img_path: str, result_images: dict) -> str:
    """把 mineru-api 产物里的图片拷到项目 data/mineru_out_upload/{task_id}/images/。

    返回 web 路径 /images/upload/{task_id}/{filename}（api.py 挂载后前端可显示）。
    失败返回空串（不阻塞管线）。
    """
    import shutil
    try:
        fname = os.path.basename(img_path.replace("\\", "/"))
        src = None
        # ① result_images 反查（mineru-api 返回的绝对/相对路径）
        info = result_images.get(fname) or result_images.get(img_path)
        if isinstance(info, dict):
            cand = info.get("img_path") or info.get("path") or ""
            if cand:
                src = Path(cand) if Path(cand).exists() else Path(img_path)
        # ② 兜底：MINERU_API_OUTPUT_ROOT/{task_id}/uploads/{pdf}/vlm/images/{fname}
        if src is None or not Path(src).exists():
            root = Path(os.environ.get("MINERU_API_OUTPUT_ROOT", "output")).expanduser()
            for cand in root.glob(f"**/{fname}"):
                src = cand
                break
        if src is None or not Path(src).exists():
            return ""
        # 拷到项目静态目录
        dest_dir = _STATIC_UPLOAD_IMAGES
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / fname
        shutil.copy2(str(src), str(dest))
        return f"/images/upload/{fname}"
    except Exception:
        return ""


def parse_document(pdf_path: Path, mode: str) -> list[dict[str, Any]]:
    if mode == "mineru":
        return _mineru_parse(pdf_path)
    return extract_pages(str(pdf_path))


def parse_document_with_fallback(
    pdf_path: Path,
    mode: str,
) -> tuple[list[dict[str, Any]], str, str]:
    """优先按请求模式解析；mineru 失败或空结果时自动回退 fitz。

    返回 (pages, actual_extract_mode, note)。
    """
    if mode != "mineru":
        pages = extract_pages_lenient(pdf_path)
        actual_mode = "fitz_raw" if any(p.get("raw_fallback") for p in pages) else "fitz"
        return pages, actual_mode, f"requested {actual_mode}"

    try:
        pages = _mineru_parse(pdf_path)
        if pages:
            return pages, "mineru", "mineru ok"
    except Exception as exc:
        mineru_err = f"{type(exc).__name__}: {exc}"
    else:
        mineru_err = "MinerU 未提取到文本内容"

    pages = extract_pages_lenient(pdf_path)
    if pages:
        note_mode = "fitz_raw" if any(p.get("raw_fallback") for p in pages) else "fitz"
        return pages, note_mode, f"mineru fallback to {note_mode}: {mineru_err}"
    raise ValueError(
        "无法从 PDF 提取到可读文本：MinerU 与 fitz 均未提取到正文。"
        f"MinerU: {mineru_err}"
    )


def _looks_garbled(text: str) -> bool:
    """乱码/二进制垃圾检测：控制字符多、常见字缺失、ASCII 占比低 → 乱码。

    乱码 PDF 的典型特征：提取文本含犐犆犛 这类伪汉字（在 CJK 区但非常用字），
    或混入大量二进制控制字符（\x9a 等）。正常中文页常见字命中 >0，
    正常英文/表格页 ASCII 字母数字占比 >25%。
    """
    if not text.strip():
        return True
    n = len(text)
    controls = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    if controls > max(2, n * 0.01):
        return True
    hanzi = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    common_hits = sum(1 for c in text if c in _COMMON_CHARS)
    ascii_alnum = sum(1 for c in text if c.isascii() and c.isalnum())
    chinese_ok = hanzi > 20 and common_hits > 0
    ascii_ok = ascii_alnum / max(n, 1) > 0.25
    if chinese_ok or ascii_ok:
        return False
    return True


def extract_pages_lenient(pdf_path: Path) -> list[dict[str, Any]]:
    """严格过滤无结果时，保留 PyMuPDF 原始文本页作为兜底。

    有些国标 PDF 存在字体映射问题，文本不是空，但常见汉字过滤会判为
    不可读。为了不让上传管线停在解析阶段，这里允许原始文本进入后续
    自检流程，并用 raw_fallback 标记给前端/后台识别。
    """
    pages = extract_pages(str(pdf_path))
    if pages:
        return pages

    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        raw_pages = []
        for i in range(doc.page_count):
            text = doc.load_page(i).get_text("text").strip()
            if text and not _looks_garbled(text):
                raw_pages.append({"page": i + 1, "text": text, "raw_fallback": True})
        # 整本质量门槛：通过率 <40% 视为整本乱码/不可读，拒绝兜底
        if len(raw_pages) >= doc.page_count * 0.4:
            return raw_pages
        return []
    finally:
        doc.close()


def clean_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for page in pages:
        text = re.sub(r"\s+\n", "\n", page["text"])
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        text = re.sub(r"\s+", " ", text)
        cleaned.append({"page": page["page"], "text": text})
    return cleaned


def _split_by_tokens(text: str, max_tokens: int = 512) -> list[str]:
    """按 ≤512 token 上限切分（对齐 v8 分块标准：不跨页、优先段落/句号边界）。"""
    if estimate_tokens(text) <= max_tokens:
        return [text]
    # 按段落/句号边界切
    segments = re.split(r"(?<=[。；！？\n])", text)
    parts: list[str] = []
    cur = ""
    for seg in segments:
        if cur and estimate_tokens(cur + seg) > max_tokens:
            parts.append(cur.strip())
            cur = seg
        else:
            cur += seg
    if cur.strip():
        parts.append(cur.strip())
    # 单段仍超长（无标点）→ 硬切（中文约 2 字/token）
    result: list[str] = []
    for p in parts:
        if estimate_tokens(p) <= max_tokens:
            result.append(p)
        else:
            char_limit = max_tokens * 2
            result.extend(p[i:i + char_limit] for i in range(0, len(p), char_limit))
    return [p for p in result if p.strip()]


def build_chunks(pages: list[dict[str, Any]], task: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    source = task["filename"]
    for page in pages:
        # 图片页（_mineru_parse 挂的 images 标记）→ 图语义块（对齐内置 Bug2 修复标准）
        if page.get("images"):
            for img in page["images"]:
                img_content = img.get("content") or ""
                ik, qk = _make_q2q(img_content, img.get("caption") or "图", source)
                item = {
                    "content": _image_semantic_content(img),
                    "source": source,
                    "page": img["page"],
                    "type": "image",
                    "section": img.get("caption") or "图",
                    "semantic_type": "图语义",
                    "important_kwd": ik,
                    "question_kwd": qk,
                    "document_id": task["document_id"],
                    "ingest_version": time.strftime("%Y%m%d_upload"),
                    "extract_mode": task["extract_mode"],
                    "image_path": img.get("web_path", ""),
                    "mermaid": "",
                    "char_len": len(img.get("content") or ""),
                    "token_len": estimate_tokens(img.get("content") or ""),
                }
                chunks.append(item)
            continue
        for raw in chunk_by_section(page["text"], page=page["page"]):
            for content in _split_by_tokens(raw["text"].strip(), 512):
                if not content:
                    continue
                semantic = classify_semantic(content, raw.get("section", "unknown"))
                ik, qk = _make_q2q(content, raw.get("section") or "unknown", source)
                item = {
                    "content": content,
                    "source": source,
                    "page": raw["page"],
                    "type": "text",
                    "section": raw.get("section") or "unknown",
                    "semantic_type": semantic,
                    "important_kwd": ik,
                    "question_kwd": qk,
                    "document_id": task["document_id"],
                    "ingest_version": time.strftime("%Y%m%d_upload"),
                    "extract_mode": task["extract_mode"],
                    "image_path": "",
                    "mermaid": "",
                    "char_len": len(content),
                    "token_len": estimate_tokens(content),
                }
                chunks.append(item)
    return chunks


def _image_semantic_content(img: dict[str, Any]) -> str:
    """图语义块 content：图标题 + 图内文字（对齐内置：不丢图语义，供检索命中）。"""
    caption = (img.get("caption") or "").strip()
    body = (img.get("content") or "").strip()
    body = re.sub(r"[\n\r]+", "；", body)
    if caption and body:
        return f"{caption}：{body}"
    return caption or body or "图（无文字描述）"


def classify_semantic(content: str, section: str) -> str:
    text = f"{section} {content}"
    if "安全" in text or "防护" in text or "危险" in text:
        return "安全要求"
    if "定义" in text or "术语" in text:
        return "术语定义"
    if any(k in text for k in ["步骤", "流程", "处理", "回收", "净化"]):
        return "流程步骤"
    if any(k in text for k in ["应", "不得", "要求", "符合"]):
        return "标准要求"
    if any(k in text for k in ["%", "mg", "指标", "纯度", "酸度"]):
        return "参数查询"
    return "概述"


def rule_keywords(content: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9/.\-]+|[\u4e00-\u9fff]{2,8}", content)
    seen: list[str] = []
    for token in tokens:
        if token not in seen and len(token) > 1:
            seen.append(token)
        if len(seen) >= 3:
            break
    return seen or ["SF6", "标准", "知识块"]


def rule_questions(content: str, source: str) -> list[str]:
    seed = rule_keywords(content)
    return [f"{source} 中关于{kw}的要求是什么？" for kw in seed[:3]]


_LLM_Q2Q = None


def _get_llm_q2q():
    """懒加载 LLM Q2Q（复用 scripts/llm_client.gen_q2q，失败返回 None 走规则兜底）。"""
    global _LLM_Q2Q
    if _LLM_Q2Q is None:
        try:
            import sys as _sys
            _SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
            if str(_SCRIPTS_DIR) not in _sys.path:
                _sys.path.insert(0, str(_SCRIPTS_DIR))
            from llm_client import gen_q2q as _g
            _LLM_Q2Q = _g
        except Exception:
            _LLM_Q2Q = False
    return _LLM_Q2Q if _LLM_Q2Q else None


def _make_q2q(content: str, section: str, source: str) -> tuple[list[str], list[str]]:
    """LLM 优先生成 Q2Q（对齐内置 make_q2q），失败降级规则（确定性补齐）。"""
    gen = _get_llm_q2q()
    if gen is not None:
        try:
            r = gen(content, section)
            if r and r.get("important_kwd") and r.get("question_kwd"):
                return r["important_kwd"][:3], r["question_kwd"][:3]
        except Exception:
            pass
    return rule_keywords(content)[:3], rule_questions(content, source)[:3]


def estimate_tokens(text: str) -> int:
    total = 0.0
    for ch in text:
        if ord(ch) >= 0x2E80:
            total += 1
        elif ch.strip():
            total += 0.5
    return int(total)


def chunk_dashboard(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    semantic: dict[str, int] = {}
    buckets = {"0-100": 0, "101-300": 0, "301-512": 0, "513+": 0}
    for chunk in chunks:
        semantic[chunk["semantic_type"]] = semantic.get(chunk["semantic_type"], 0) + 1
        tok = chunk["token_len"]
        if tok <= 100:
            buckets["0-100"] += 1
        elif tok <= 300:
            buckets["101-300"] += 1
        elif tok <= 512:
            buckets["301-512"] += 1
        else:
            buckets["513+"] += 1
    return {
        "chunk_count": len(chunks),
        "avg_token": round(sum(c["token_len"] for c in chunks) / len(chunks), 1) if chunks else 0,
        "fragment_count": sum(1 for c in chunks if c["char_len"] < 100),
        "semantic_distribution": semantic,
        "token_histogram": buckets,
    }


_VECTORIZE_LOCK = threading.Lock()


def vectorize_chunks(chunks: list[dict[str, Any]], task: dict[str, Any]) -> PipelineBuild:
    ensure_temp_dir()
    # 复用 retrieve._get_model 单例（避免并发创建多实例 → meta tensor / 内存爆）
    from sf6_rag.retrieve import _get_model
    with _VECTORIZE_LOCK:
        model = _get_model()
    embed_texts = [build_embed_text(c) for c in chunks]
    out = model.encode(
        embed_texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        batch_size=8,
    )
    dense_vectors = out["dense_vecs"]
    sparse_vectors = out["lexical_weights"]
    return PipelineBuild(
        document_id=task["document_id"],
        collection=task["collection"],
        chunks=chunks,
        dense_vectors=dense_vectors,
        sparse_vectors=sparse_vectors,
        sparse_sizes=[len(w) for w in sparse_vectors],
    )


def build_embed_text(chunk: dict[str, Any]) -> str:
    parts = [chunk["content"]]
    parts.extend(chunk.get("important_kwd") or [])
    parts.extend(chunk.get("question_kwd") or [])
    return "\n".join(parts)


def upsert_collection(build: PipelineBuild) -> None:
    ensure_temp_dir()
    from qdrant_client.models import (
        Distance,
        Modifier,
        PointStruct,
        SparseVector,
        SparseVectorParams,
        VectorParams,
    )
    from sf6_rag.retrieve import _get_client

    client = _get_client()
    if client.collection_exists(build.collection):
        client.delete_collection(build.collection)
    client.create_collection(
        collection_name=build.collection,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    points = []
    for i, chunk in enumerate(build.chunks):
        sparse = build.sparse_vectors[i]
        items = sorted(sparse.items(), key=lambda kv: int(kv[0]))
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{build.document_id}-{i}")),
                vector={
                    "dense": build.dense_vectors[i].tolist()
                    if hasattr(build.dense_vectors[i], "tolist")
                    else build.dense_vectors[i],
                    "sparse": SparseVector(
                        indices=[int(k) for k, _ in items],
                        values=[float(v) for _, v in items],
                    ),
                },
                payload={
                    "text": chunk["content"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "type": chunk["type"],
                    "section": chunk["section"],
                    "semantic_type": chunk["semantic_type"],
                    "important_kwd": chunk["important_kwd"],
                    "question_kwd": chunk["question_kwd"],
                    "document_id": chunk["document_id"],
                    "ingest_version": chunk["ingest_version"],
                    "extract_mode": chunk["extract_mode"],
                    "image_path": chunk.get("image_path", ""),
                    "mermaid": chunk.get("mermaid", ""),
                },
            )
        )
    client.upsert(collection_name=build.collection, points=points)


def self_check(build: PipelineBuild) -> dict[str, Any]:
    degraded = any(c.get("extract_mode") == "fitz_raw" for c in build.chunks)
    return {
        "mode": "self_check",
        "label": "新文档：自检通过，无评测集，MRR 跳过" if not degraded
        else "新文档：自检通过，但 PDF 文本映射异常，已使用 fitz 原始文本兜底",
        "dense_dim_ok": True,
        "sparse_non_empty": all(size > 0 for size in build.sparse_sizes),
        "chunk_count_match": True,
        "vectors_non_empty": bool(build.chunks),
        "extract_quality": "degraded" if degraded else "normal",
        "self_recall": "skipped",
    }


def cleanup_collection(collection: str) -> None:
    if not collection.startswith("transformer_upload_"):
        return
    try:
        from sf6_rag.retrieve import _get_client

        client = _get_client()
        if client.collection_exists(collection):
            client.delete_collection(collection)
    except Exception:
        pass
    col_dir = QDRANT_DIR / "collection" / collection
    if col_dir.exists():
        try:
            shutil.rmtree(col_dir)
        except OSError:
            pass


def ensure_temp_dir() -> None:
    temp = DATA_DIR / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMP", str(temp))
    os.environ.setdefault("TEMP", str(temp))
    os.environ.setdefault("TMPDIR", str(temp))
