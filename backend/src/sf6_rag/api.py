"""FastAPI 服务：/health + /ask。

/ask 流程：question → retrieve（Qdrant 复合检索）→ 拒答判定（dense 阈值）
→ DeepSeek 生成（带引用）→ 返回 {answer, citations, source}。

用法:
    cd backend && PYTHONPATH=src D:/an/envs/langchain/python.exe -m uvicorn sf6_rag.api:app --port 8000
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 让 llm_client 可导入（在 scripts/ 下，密钥在 ~/.deepseek.env）
_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from sf6_rag.retrieve import (  # noqa: E402
    REJECT_DENSE_THRESHOLD,
    REJECT_SPARSE_THRESHOLD,
    dense_similarity,
    get_chunk_by_id,
    list_chunks as list_retrieval_chunks,
    retrieve,
    retrieve_debug,
    sparse_similarity,
)
from sf6_rag.generate import FALLBACK_TEXT, format_citation  # noqa: E402
from sf6_rag.auth import login, get_current_user, verify_token, User  # noqa: E402
from sf6_rag import pipeline  # noqa: E402

app = FastAPI(title="SF6 知识库问答", version="1.0.0")

# CORS：允许前端页面跨域调用 /ask
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态图片目录：挂载多文档解析产物，供前端显示原图（不复制图片文件）
_MULTIDOC_DIR = Path(__file__).resolve().parents[3] / "data" / "mineru_out_multidoc"
if _MULTIDOC_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(_MULTIDOC_DIR)), name="images")

# 上传文档的图片（pipeline 拷到 data/mineru_out_upload/images/）→ /images/upload/xxx
_UPLOAD_IMAGES_DIR = Path(__file__).resolve().parents[3] / "data" / "mineru_out_upload" / "images"
if _UPLOAD_IMAGES_DIR.exists():
    app.mount("/images/upload", StaticFiles(directory=str(_UPLOAD_IMAGES_DIR)), name="upload_images")

_FRONTEND = Path(__file__).resolve().parents[3] / "frontend"

SOURCE_NAME = "变压器检测维护标准知识库"

# 问答记录（内存，教学级；生产应持久化到 DB）
_qa_history: list[dict] = []
_START_TIME = time.time()
pipeline.init_store()


def _require_auth(authorization: str | None = Header(None)) -> User:
    """认证依赖：从 Authorization 头解析用户，未登录抛 401。"""
    user = get_current_user(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="未登录或 token 无效")
    return user


def json_dumps(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _require_admin(user: User = Depends(_require_auth)) -> User:
    """管理员权限依赖。"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


def _require_admin_token(token: str) -> User:
    """SSE 用 query token 鉴权；EventSource 无法设置 Authorization 头。"""
    user = verify_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="未登录或 token 无效")
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user

_RERANKER_PATH = r"D:\models\bge-reranker-v2-m3"
_reranker = None


def _get_reranker():
    """懒加载 bge-reranker（受环境变量 RERANK_ENABLED 控制，默认关闭）。

    评测证明 reranker 在当前 22 块小规模下 MRR 为负收益（0.9286 → 0.9167），
    故默认不启用；设 RERANK_ENABLED=1 时启用，trace 展示重排流程。
    """
    import os as _os
    if _os.environ.get("RERANK_ENABLED", "0") != "1":
        return None
    global _reranker
    if _reranker is None:
        from pathlib import Path as _P
        if _P(_RERANKER_PATH).exists():
            from FlagEmbedding import FlagReranker
            _reranker = FlagReranker(_RERANKER_PATH, use_fp16=False)
    return _reranker


class AskRequest(BaseModel):
    question: str
    history: list[dict] | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    source: str
    refused: bool
    trace: dict | None = None
    images: list[str] = []
    mermaid: list[str] = []


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    role: str


class RetrieveTestRequest(BaseModel):
    question: str
    top_k: int = 5
    enable_rerank: bool = False


# 上下文组织（生成前）参数（讲师设计：score阈值过滤→去重→token预算裁剪）
_CONTEXT_MIN_SCORE = 0.0        # rerank_score < 0 = 负相关噪音（实测：相关>1.9，噪音<-0.8）
_CONTEXT_SIM_THRESHOLD = 0.90   # embedding 余弦相似度 > 0.90 视为重复 chunk，合并
_CONTEXT_TOKEN_BUDGET = 2000    # 进 prompt 的总 token 预算（对齐生成窗口）
_CONTEXT_MIN_KEEP = 2           # 兜底：再少也保留至少 2 条（防过度裁剪丢关键 chunk）


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """BGE-M3 dense 编码（去重用）。"""
    from sf6_rag.retrieve import _get_model
    out = _get_model().encode(texts, return_dense=True, return_sparse=False,
                              return_colbert_vecs=False, batch_size=8)
    return out["dense_vecs"].tolist()


def _organize_context(question: str, ranked: list[dict]) -> tuple[list[dict], dict]:
    """上下文组织（生成前）：score阈值过滤 → embedding去重 → token预算裁剪。

    讲师设计：检索5条 → 低于阈值去掉(防噪音) → 相似chunk合并 → 超窗口截断 → 进prompt。
    红线：不能过度裁剪导致答案缺关键 chunk（_CONTEXT_MIN_KEEP 兜底）。
    """
    stats = {"before": len(ranked)}
    # ① score 阈值过滤（防噪音；rerank 分有判别力）
    kept = [h for h in ranked if (h.get("rerank_score") or h.get("score") or 0) >= _CONTEXT_MIN_SCORE]
    if not kept:
        kept = ranked[:_CONTEXT_MIN_KEEP]
    stats["filtered"] = len(ranked) - len(kept)
    # ② embedding 去重（相似 chunk 合并，保留分数高的）
    if len(kept) > _CONTEXT_MIN_KEEP:
        try:
            vecs = _embed_texts([(h.get("content") or "")[:400] for h in kept])
            unique: list[dict] = []
            for i, h in enumerate(kept):
                dup = any(_cosine(vecs[i], vecs[j]) >= _CONTEXT_SIM_THRESHOLD for j in range(i))
                if not dup:
                    unique.append(h)
            kept = unique or kept[:_CONTEXT_MIN_KEEP]
        except Exception:
            pass  # 编码失败不阻塞主链路
    stats["deduped"] = len(ranked) - len(kept) - stats["filtered"]
    # ③ token 预算裁剪（按相关性从尾部截，保留 MIN_KEEP 兜底）
    from sf6_rag.pipeline import estimate_tokens
    total = sum(estimate_tokens(h.get("content") or "") for h in kept)
    trimmed = 0
    while total > _CONTEXT_TOKEN_BUDGET and len(kept) > _CONTEXT_MIN_KEEP:
        tail = kept.pop()
        total -= estimate_tokens(tail.get("content") or "")
        trimmed += 1
    stats["trimmed"] = trimmed
    stats["after"] = len(kept)
    stats["tokens"] = total
    return kept, stats


_rewrite_cache: dict[str, str] = {}


def _rewrite_query(question: str) -> str:
    """查询改写（召回前，可选）：口语化/简短 query → 规范术语 query。

    讲师设计：用户 query → DeepSeek 改写为规范术语 → 与 Q2Q 索引侧 question_kwd 成对呼应。
    开关 QUERY_REWRITE_ENABLED=1 开启；结果可缓存；失败回退原 query（不阻塞）。
    """
    if os.environ.get("QUERY_REWRITE_ENABLED", "0") != "1":
        return question
    cached = _rewrite_cache.get(question)
    if cached:
        return cached
    try:
        from llm_client import chat
        sys_prompt = (
            "你是电力行业标准检索的查询改写器。把用户口语化/简短的问题改写成规范术语查询，"
            "保留标准号（如 GB/T、DL/T），不要添加检索不存在的实体，"
            "只输出改写后的查询本身，不要任何解释或前缀。"
        )
        text = chat([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ], json_mode=False)
        rewritten = (text or "").strip().strip('"')
        rewritten = rewritten.splitlines()[0].strip() if rewritten else ""
        if len(rewritten) < 2 or rewritten == question:
            return question
        _rewrite_cache[question] = rewritten
        return rewritten
    except Exception:
        return question  # 改写失败 → 原 query 兜底


def _truncate_context(text: str, limit: int = 2000) -> str:
    """证据块智能截断：保留头部+尾部（表/公式常在块尾），中间省略。"""
    text = text or ""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n…[内容省略]…\n" + text[-half:]


def _build_generation_messages(question: str, results: list[dict], history: list[dict] | None = None) -> list[dict]:
    context = "\n\n".join(
        (
            f"[证据 {idx + 1}] 来源：{r.get('source') or '标准文档'}；"
            f"页码：{r.get('page') or '—'}；章节：{r.get('section') or '章节未提供'}\n"
            f"{_truncate_context(r.get('content'))}"
        )
        for idx, r in enumerate(results)
    )
    sys_prompt = (
        "你是电力行业标准问答助手。只依据给定检索块回答，不要编造。"
        "回答要简洁，并用 [第 X 页] 标注每条事实的出处。"
        "如果用户追问（如“那它的试验压力呢”），结合上一轮对话理解指代。"
    )
    messages: list[dict] = [{"role": "system", "content": sys_prompt}]
    # 历史对话（最近 6 条，作为上下文支持多轮追问）
    if history:
        for h in history[-6:]:
            role = h.get("role")
            hc = h.get("content")
            if role in ("user", "assistant") and hc:
                messages.append({"role": role, "content": str(hc)[:800]})
    user_prompt = f"检索块：\n{context}\n\n问题：{question}"
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _format_prompt(messages: list[dict]) -> str:
    return "\n\n".join(
        f"{str(item.get('role', '')).upper()}:\n{item.get('content', '')}"
        for item in messages
    )


def _llm_model_name() -> str:
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")


def _generate_answer(question: str, results: list[dict], history: list[dict] | None = None) -> str:
    """用 DeepSeek 基于检索块生成回答（复用 llm_client.chat）。"""
    try:
        from llm_client import chat
    except ImportError:
        return ""

    messages = _build_generation_messages(question, results, history)
    text = chat(
        messages,
        json_mode=False,
    )
    return text or ""


def _step(
    name: str,
    label: str,
    state: str,
    elapsed_ms: int = 0,
    detail: dict | None = None,
) -> dict:
    item = {"name": name, "label": label, "state": state, "elapsed_ms": elapsed_ms}
    if detail is not None:
        item["detail"] = detail
    return item


def _ms(start: float) -> int:
    return max(1, int((time.time() - start) * 1000))


def _skipped_steps(*names: str) -> list[dict]:
    labels = {
        "dense_retrieval": "Dense检索",
        "sparse_retrieval": "Sparse检索",
        "rrf_fusion": "RRF融合",
        "rerank": "Rerank",
        "generation": "生成回答",
        "citation": "引用输出",
    }
    return [_step(name, labels[name], "skipped", 0) for name in names]


def _trace_hit(hit: dict, *, score_key: str = "score", score: float | None = None) -> dict:
    value = hit.get(score_key) if score is None else score
    return {
        "id": hit.get("id", ""),
        "chunk_id": hit.get("chunk_id") or hit.get("id", ""),
        "page": hit.get("page"),
        "section": hit.get("section", ""),
        "score": round(float(value), 4) if value is not None else None,
        "source": hit.get("source", ""),
        "document_id": hit.get("document_id", ""),
        "collection": hit.get("collection", ""),
        "document_kind": hit.get("document_kind", ""),
        "semantic_type": hit.get("semantic_type", ""),
        "content": (hit.get("content") or "")[:260],
        "image_path": hit.get("image_path", ""),
        "mermaid": hit.get("mermaid", ""),
        "extract_mode": hit.get("extract_mode", ""),
    }


@app.get("/echarts.min.js")
def echarts_js():
    """本地 ECharts（避免浏览器 CDN 拦截）。"""
    f = _FRONTEND / "echarts.min.js"
    if f.exists():
        return Response(f.read_bytes(), media_type="application/javascript")
    return Response(status_code=404)


@app.get("/katex.min.js")
def katex_js():
    """本地 KaTeX（公式渲染）。"""
    f = _FRONTEND / "katex.min.js"
    if f.exists():
        return Response(f.read_bytes(), media_type="application/javascript")
    return Response(status_code=404)


@app.get("/katex.min.css")
def katex_css():
    f = _FRONTEND / "katex.min.css"
    if f.exists():
        return Response(f.read_bytes(), media_type="text/css")
    return Response(status_code=404)


@app.get("/favicon.ico")
def favicon():
    """网站图标（消除 404 噪音）。"""
    f = _FRONTEND / "favicon.svg"
    if f.exists():
        return Response(f.read_bytes(), media_type="image/svg+xml")
    return Response(status_code=404)


@app.get("/", response_class=HTMLResponse)
def index():
    """返回前端问答页（frontend/index.html）。"""
    html = _FRONTEND / "index.html"
    if html.exists():
        return html.read_text(encoding="utf-8")
    return "<h1>SF6 知识库问答</h1><p>前端页面未找到</p>"


@app.get("/login", response_class=HTMLResponse)
def login_page():
    """返回登录页（frontend/login.html）。"""
    html = _FRONTEND / "login.html"
    if html.exists():
        return html.read_text(encoding="utf-8")
    return "<h1>登录</h1>"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    """返回后台管理页（frontend/admin.html）。"""
    html = _FRONTEND / "admin.html"
    if html.exists():
        return html.read_text(encoding="utf-8")
    return "<h1>后台管理</h1>"


@app.get("/admin/stats")
def admin_stats(user: User = Depends(_require_admin)):
    """后台统计：问答记录 + 集合信息（admin 专用，非 admin 拒绝）。"""
    from sf6_rag.retrieve import _get_client, _COLLECTION, search_collection_names

    client = _get_client()
    count = sum(
        client.count(collection_name=name).count
        for name in search_collection_names(client)
    )
    return {
        "collection": _COLLECTION,
        "chunk_count": count,
        "qa_history": _qa_history[-20:],  # 最近 20 条
    }


@app.post("/login", response_model=LoginResponse)
def login_endpoint(req: LoginRequest):
    """登录：校验用户名密码，返回 token + role。"""
    result = login(req.username, req.password)
    if result is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return LoginResponse(token=result["token"], role=result["role"])


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    mode: str = Form("fitz"),
    user: User = Depends(_require_admin),
):
    """提交上传文档构建任务。"""
    if mode not in {"fitz", "mineru"}:
        raise HTTPException(status_code=422, detail="mode 只能是 fitz 或 mineru")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="仅支持 PDF 文件")
    content = await file.read()
    try:
        return pipeline.create_task(file.filename, content, mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/pipeline/{task_id}")
def pipeline_snapshot(task_id: str, user: User = Depends(_require_admin)):
    """返回管线任务快照，支持页面刷新后恢复进度。"""
    task = pipeline.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task_id 不存在")
    return task


@app.get("/pipeline/{task_id}/events")
def pipeline_events(task_id: str, token: str = Query(...)):
    """SSE 推送管线进度；query token 用于 EventSource 鉴权。"""
    _require_admin_token(token)
    if pipeline.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail="task_id 不存在")

    def stream():
        last_payload = ""
        deadline = time.time() + 3600
        while time.time() < deadline:
            task = pipeline.get_task(task_id)
            if task is None:
                yield "event: error\ndata: {\"detail\":\"task_id 不存在\"}\n\n"
                return
            payload = json_dumps({"event": "snapshot", "task": task})
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if task["status"] in {"done", "failed"}:
                return
            time.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/documents")
def documents(user: User = Depends(_require_admin)):
    """文档状态列表：内置三文档 + 上传任务。"""
    return pipeline.list_documents()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str, user: User = Depends(_require_admin)):
    """删除上传文档及其独立 Qdrant 集合；内置集合受红线保护。"""
    if not document_id:
        raise HTTPException(status_code=404, detail="文档不存在")
    if not document_id.startswith("transformer_upload_") and len(document_id) != 32:
        raise HTTPException(status_code=403, detail="内置文档不可删除")
    result = pipeline.delete_upload_document(document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return result


@app.get("/admin/health")
def admin_health(user: User = Depends(_require_auth)):
    """运维控制台健康状态。"""
    from sf6_rag.retrieve import _COLLECTION, _QDRANT_DIR, _get_client

    qdrant_status = "disconnected"
    chunk_count = 0
    upload_collections: dict[str, int] = {}
    try:
        client = _get_client()
        for col in client.get_collections().collections:
            name = col.name
            if name == _COLLECTION or name.startswith("transformer_upload_"):
                cnt = client.count(collection_name=name).count
                chunk_count += cnt
                if name.startswith("transformer_upload_"):
                    upload_collections[name] = cnt
        qdrant_status = "connected"
    except Exception:
        qdrant_status = "disconnected"

    docs = pipeline.list_documents()
    # 生成能力检查与 llm_client 一致：key 在 ~/.deepseek.env（dotenv 文件，非环境变量）
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.expanduser("~/.deepseek.env"), override=False)
    except Exception:
        pass
    return {
        "collection": _COLLECTION,
        "chunk_count": chunk_count,
        "upload_collections": upload_collections,
        "document_count": len(docs),
        "api_health": "online",
        "retrieval_ready": qdrant_status == "connected",
        "generation_ready": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "reranker_enabled": os.environ.get("RERANK_ENABLED", "0") == "1",
        "qdrant_status": qdrant_status,
        "qdrant_path": str(_QDRANT_DIR),
        "uptime_s": round(time.time() - _START_TIME, 1),
    }


@app.post("/retrieve-test")
def retrieve_test(req: RetrieveTestRequest, user: User = Depends(_require_admin)):
    """只检索不生成，用于后台检索测试台。"""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question 不能为空")
    if req.top_k < 1 or req.top_k > 20:
        raise HTTPException(status_code=422, detail="top_k 范围为 1-20")
    return retrieve_debug(question, top_k=req.top_k, enable_rerank=req.enable_rerank)


@app.get("/chunks")
def chunks(
    document_id: str | None = Query(None),
    semantic_type: str | None = Query(None),
    collection: str | None = Query(None),
    limit: int = Query(1000, ge=1, le=2000),
    user: User = Depends(_require_admin),
):
    """分块列表与统计看板数据，覆盖内置 v8 + 上传集合。"""
    return list_retrieval_chunks(
        document_id=document_id,
        semantic_type=semantic_type,
        collection=collection,
        limit=limit,
    )


@app.get("/chunks/{chunk_id}")
def chunk_detail(
    chunk_id: str,
    collection: str | None = Query(None),
    user: User = Depends(_require_admin),
):
    """按 Qdrant point id 查询块详情（P2 钻取接口）。"""
    chunk = get_chunk_by_id(chunk_id, collection=collection)
    if chunk is None:
        raise HTTPException(status_code=404, detail="chunk 不存在")
    return {
        "id": chunk["id"],
        "content": chunk["content"],
        "page": chunk["page"],
        "document": chunk["document"],
        "source": chunk["source"],
        "section": chunk["section"],
        "semantic_type": chunk["semantic_type"],
        "document_id": chunk["document_id"],
        "collection": chunk["collection"],
        "document_kind": chunk["document_kind"],
        "token_len": chunk["token_len"],
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, user: User = Depends(_require_auth)):
    total_start = time.time()
    question = req.question.strip()
    steps = [_step("question_input", "问题输入", "done", _ms(total_start))]
    if not question:
        steps.extend(_skipped_steps(
            "dense_retrieval", "sparse_retrieval", "rrf_fusion",
            "rerank", "generation", "citation",
        ))
        return AskResponse(
            answer=FALLBACK_TEXT,
            citations=[],
            source="",
            refused=True,
            trace={"steps": steps, "reason": "empty_question"},
        )

    # 拒答判定：dense 偏语义，sparse 偏精确标准号，二者都低才拒答。
    dense_start = time.time()
    dense_score = dense_similarity(question)
    dense_elapsed = _ms(dense_start)
    sparse_score = None
    sparse_elapsed = 0
    passed = dense_score >= REJECT_DENSE_THRESHOLD
    if not passed:
        sparse_start = time.time()
        sparse_score = sparse_similarity(question)
        sparse_elapsed = _ms(sparse_start)
        passed = sparse_score >= REJECT_SPARSE_THRESHOLD

    reject_elapsed = dense_elapsed + sparse_elapsed
    steps.append(_step(
        "reject_check",
        "拒答判断",
        "done",
        reject_elapsed,
        {
            "dense": round(float(dense_score), 4),
            "sparse": round(float(sparse_score), 4) if sparse_score is not None else None,
            "dense_threshold": REJECT_DENSE_THRESHOLD,
            "sparse_threshold": REJECT_SPARSE_THRESHOLD,
            "passed": passed,
        },
    ))
    steps.append(_step("dense_retrieval", "Dense检索", "done", dense_elapsed))
    steps.append(_step(
        "sparse_retrieval",
        "Sparse检索",
        "done" if sparse_score is not None else "skipped",
        sparse_elapsed,
    ))

    if not passed:
        steps.extend(_skipped_steps("rrf_fusion", "rerank", "generation", "citation"))
        _qa_history.append({
            "question": question,
            "answer_preview": FALLBACK_TEXT,
            "page": None,
            "user": user.username,
            "role": user.role,
            "refused": True,
        })
        if len(_qa_history) > 100:
            _qa_history.pop(0)
        return AskResponse(
            answer=FALLBACK_TEXT,
            citations=[],
            source="",
            refused=True,
            trace={
                "steps": steps,
                "rough_top15": [],
                "rough_top5": [],
                "reranked_order": [],
                "reranker_enabled": False,
                "gen_elapsed_s": 0,
                "context_blocks": [],
                "context_stats": {},
            },
        )

    # 查询改写（召回前，可选）：口语 query → 规范术语 query（对齐 Q2Q 索引侧 question_kwd）
    rewrite_start = time.time()
    search_query = _rewrite_query(question)
    steps.append(_step(
        "query_rewrite",
        "查询改写",
        "done" if search_query != question else "skipped",
        _ms(rewrite_start),
        {"original": question, "rewritten": search_query},
    ))

    # 粗排：RRF 复合检索取 top15 候选（讲师设计：粗排 15 → 精排 5，防 rank7/8 真答案被截断）
    retrieve_start = time.time()
    candidates = retrieve(search_query, top_k=15)
    if not candidates:
        steps.append(_step("rrf_fusion", "RRF融合", "done", _ms(retrieve_start), {"top5": []}))
        steps.extend(_skipped_steps("rerank", "generation", "citation"))
        _qa_history.append({
            "question": question,
            "answer_preview": FALLBACK_TEXT,
            "page": None,
            "user": user.username,
            "role": user.role,
            "refused": True,
        })
        if len(_qa_history) > 100:
            _qa_history.pop(0)
        return AskResponse(
            answer=FALLBACK_TEXT,
            citations=[],
            source="",
            refused=True,
            trace={
                "steps": steps,
                "rough_top15": [],
                "rough_top5": [],
                "reranked_order": [],
                "reranker_enabled": False,
                "gen_elapsed_s": 0,
                "context_blocks": [],
                "context_stats": {},
            },
        )
    rrf_elapsed = _ms(retrieve_start)
    rough_top15 = [_trace_hit(c) for c in candidates]
    steps.append(_step(
        "rrf_fusion",
        "RRF融合",
        "done",
        rrf_elapsed,
        {"top15": rough_top15},
    ))

    rerank_start = time.time()
    reranker = _get_reranker()
    reranked = candidates  # 默认无 rerank 时用粗排顺序
    rerank_scores = None
    if reranker is not None:
        pairs = [(question, c["content"]) for c in candidates]
        scores = reranker.compute_score(pairs, batch_size=8)
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
        reranked = [candidates[i] for i in order]
        rerank_scores = [float(scores[i]) for i in order]
    steps.append(_step(
        "rerank",
        "Rerank",
        "done" if reranker is not None else "skipped",
        _ms(rerank_start) if reranker is not None else 0,
        {"enabled": reranker is not None},
    ))

    # 生成回答（讲师：上下文组织 = 精排top5 → score过滤 → 去重 → token预算 → 进prompt）
    org_start = time.time()
    context_blocks, org_stats = _organize_context(question, reranked[:5])
    steps.append(_step("context_organize", "上下文组织", "done", _ms(org_start), org_stats))
    generation_messages = _build_generation_messages(question, context_blocks, req.history)
    prompt_text = _format_prompt(generation_messages)
    gen_start = time.time()
    answer = _generate_answer(question, context_blocks, req.history)
    gen_elapsed = round(time.time() - gen_start, 3)
    steps.append(_step("generation", "生成回答", "done", int(gen_elapsed * 1000)))
    if not answer:  # LLM 失败 → 用检索块原文兜底
        answer = context_blocks[0]["content"][:400] if context_blocks else reranked[0]["content"][:400]

    citation_start = time.time()
    citations = [
        format_citation(r.get("source") or SOURCE_NAME, r["page"])
        for r in context_blocks  # 诚实：引用=实际进 LLM 的块（去重/过滤后）
        if r.get("page")
    ]
    steps.append(_step("citation", "引用输出", "done", _ms(citation_start), {"count": len(citations)}))

    # 从 rerank 后结果收集图原图路径 + mermaid 流程图（供前端展示）
    images = []
    mermaid = []
    for r in context_blocks:
        ip = r.get("image_path") or ""
        if ip and ip not in images:
            images.append(ip)
        md = r.get("mermaid") or ""
        if md and md not in mermaid:
            mermaid.append(md)

    # trace：粗排 top15（RRF 分）+ rerank 后排名 + 最终入 prompt 证据 + 生成用时
    trace = {
        "steps": steps,
        "rewritten_query": search_query,
        "rough_top15": rough_top15,
        "rough_top5": rough_top15[:5],
        "context_blocks": [_trace_hit(c) for c in context_blocks],
        "context_stats": org_stats,
        "reranked_order": [
            {
                **_trace_hit(c, score=rerank_scores[i] if rerank_scores else c.get("score")),
                "rerank_score": round(rerank_scores[i], 4) if rerank_scores else None,
                "rrf_score": round(float(c.get("score") or 0), 4),
            }
            for i, c in enumerate(reranked)
        ],
        "reranker_enabled": reranker is not None,
        "llm_model": _llm_model_name(),
        "llm": {
            "model": _llm_model_name(),
            "gen_elapsed_s": gen_elapsed,
        },
        "gen_elapsed_s": gen_elapsed,
        "prompt": prompt_text,
        "extract_mode": next((r.get("extract_mode") for r in reranked if r.get("extract_mode")), ""),
        "total_elapsed_ms": _ms(total_start),
    }

    # 记录问答（内存，供后台管理页展示）
    _qa_history.append({
        "question": question,
        "answer_preview": answer[:100],
        "page": context_blocks[0]["page"] if context_blocks else reranked[0]["page"],
        "user": user.username,
        "role": user.role,
        "refused": False,
    })
    # 只保留最近 100 条
    if len(_qa_history) > 100:
        _qa_history.pop(0)

    return AskResponse(
        answer=answer,
        citations=citations,
        source=(context_blocks[0].get("source") if context_blocks else reranked[0].get("source")) or SOURCE_NAME,
        refused=False,
        trace=trace,
        images=images,
        mermaid=mermaid,
    )


# ============ 评测历史（趋势图数据源） ============
_EVAL_DB = Path(__file__).resolve().parents[3] / "data" / "eval_history.sqlite"


def _eval_connect():
    import sqlite3
    conn = sqlite3.connect(_EVAL_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS eval_history ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "created_at REAL NOT NULL,"
        "collection TEXT NOT NULL,"
        "mrr REAL, recall REAL, refusal_acc REAL,"
        "note TEXT DEFAULT '')"
    )
    return conn


def record_eval(collection: str, mrr=None, recall=None, refusal_acc=None, note: str = "") -> None:
    """写入一条评测历史（内置文档评测 / 上传自检均可）。"""
    try:
        conn = _eval_connect()
        conn.execute(
            "INSERT INTO eval_history (created_at, collection, mrr, recall, refusal_acc, note)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), collection, mrr, recall, refusal_acc, note),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


@app.get("/admin/eval-history")
def eval_history(user: User = Depends(_require_auth)):
    """评测趋势历史（MRR/recall 随时间）。"""
    try:
        conn = _eval_connect()
        rows = conn.execute(
            "SELECT id, created_at, collection, mrr, recall, refusal_acc, note"
            " FROM eval_history ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


@app.get("/admin/vector-libraries")
def vector_libraries(user: User = Depends(_require_auth)):
    """向量库概览：所有 Qdrant 集合 + 向量配置 + 示例 payload。"""
    from sf6_rag.retrieve import _get_client
    try:
        client = _get_client()
        result = []
        for col in client.get_collections().collections:
            name = col.name
            try:
                cnt = client.count(name, exact=True).count
            except Exception:
                cnt = 0
            info = {"collection": name, "chunk_count": cnt}
            # 向量配置
            try:
                c_info = client.get_collection(name)
                vecs = c_info.config.params.vectors
                if isinstance(vecs, dict):
                    # named vectors：{"dense": VectorParams(size=1024), "sparse": ...}
                    dense_cfg = vecs.get("dense")
                    info["dense_dim"] = dense_cfg.size if dense_cfg else None
                    info["dense_distance"] = str(dense_cfg.distance) if dense_cfg else None
                elif hasattr(vecs, "size"):
                    info["dense_dim"] = vecs.size
                    info["dense_distance"] = str(vecs.distance)
                else:
                    info["dense_dim"] = None
                sparse = getattr(c_info.config.params, "sparse_vectors", None)
                info["sparse_enabled"] = sparse is not None
            except Exception:
                info["dense_dim"] = None
            # 示例 payload（前 1 个点的元数据字段）
            try:
                pts, _ = client.scroll(name, limit=1, with_payload=True, with_vectors=False)
                if pts:
                    p = pts[0].payload or {}
                    info["payload_fields"] = list(p.keys())
                    info["sample"] = {
                        "text_preview": (p.get("text") or "")[:80],
                        "page": p.get("page"),
                        "semantic_type": p.get("semantic_type"),
                        "document_id": p.get("document_id"),
                        "source": p.get("source"),
                        "extract_mode": p.get("extract_mode"),
                    }
            except Exception:
                info["payload_fields"] = []
            result.append(info)
        return {"collections": result}
    except Exception as e:
        return {"error": str(e), "collections": []}
