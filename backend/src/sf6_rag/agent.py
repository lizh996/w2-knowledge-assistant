# -*- coding: utf-8 -*-
"""Agentic RAG 演示端点核心：DeepSeek function calling 自主决策。

LLM 拿到问题后自己决定：调工具检索 / 直接答 / 拒答，多轮循环后出中文答案。

协议（对齐 Desktop/fc_demo2/client.py，已实测通过）：
- OpenAI 兼容端点 https://api.deepseek.com/chat/completions
- model=deepseek-chat，temperature=0.3
- key 从 ~/.claude/settings.json 的 env.ANTHROPIC_AUTH_TOKEN 读（不硬编码）
- tools 用 type=function；tool_calls 回填 role=tool + tool_call_id

硬约束：
- 检索复用 retrieve()（不复制检索实现）；文档清单复用 pipeline.list_documents()
- 只读，不改现有 /ask / 检索 / pipeline 任何逻辑
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sf6_rag.retrieve import retrieve
from sf6_rag.pipeline import list_documents
from sf6_rag.generate import format_citation

_API_URL = "https://api.deepseek.com/chat/completions"
_MODEL = "deepseek-chat"
_TEMPERATURE = 0.3
_MAX_TOOL_ROUNDS = 3
_LLM_TIMEOUT = 60  # 秒
_SETTINGS = Path.home() / ".claude" / "settings.json"

_SYSTEM_PROMPT = (
    "你是电力标准知识助手。判断用户问题是否需要查知识库：\n"
    "1. 标准/参数/检测方法/结构图等电力标准知识类问题 → 调用 search_knowledge 检索"
    "（拿不准有哪些文档时，可先调 list_documents 看可选范围；"
    "要对比多份文档时，可对每份各调一次 search_knowledge）。\n"
    "2. 闲聊/常识/无需查库 → 直接回答。\n"
    "3. 与电力标准无关（如天气、股票）或危害类 → 直接以『[拒答]』开头输出拒绝理由，不要调用工具。\n"
    "回答用中文，引用证据时标注页码，不要编造检索结果里没有的内容。"
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "列出知识库当前可检索的文档清单（id/名称/块数），用于确认可选检索范围。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "在电力标准知识库中检索与问题相关的片段。"
                "标准/参数/检测方法/结构图类问题必须调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "检索查询句"},
                    "doc_id": {
                        "type": "string",
                        "description": "可选，限定文档 id（内置国标为 gb_t_*，上传文档为 32 位 hex）",
                    },
                    "top_k": {"type": "integer", "description": "返回片段数，默认 5，最大 10"},
                },
                "required": ["question"],
            },
        },
    },
]


def _load_key() -> str:
    """从 ~/.claude/settings.json 读 DeepSeek key（不硬编码）。"""
    try:
        data = json.loads(_SETTINGS.read_text(encoding="utf-8"))
        return data.get("env", {}).get("ANTHROPIC_AUTH_TOKEN") or ""
    except Exception:
        return ""


def _call_llm(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """发一次 DeepSeek chat/completions 请求，返回 choices[0].message。"""
    body = {"model": _MODEL, "messages": messages, "temperature": _TEMPERATURE}
    if tools:
        body["tools"] = tools
    key = _load_key()
    if not key:
        raise RuntimeError("DeepSeek key 缺失：~/.claude/settings.json 无 ANTHROPIC_AUTH_TOKEN")
    req = urllib.request.Request(
        _API_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=_LLM_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]


def _tool_list_documents(_args: dict) -> list[dict]:
    """list_documents 工具实现：复用 pipeline.list_documents（只读）。"""
    docs = list_documents()
    return [
        {"id": d.get("id"), "name": d.get("name"), "chunk_count": d.get("chunk_count")}
        for d in docs
    ]


def _tool_search_knowledge(args: dict) -> list[dict]:
    """search_knowledge 工具实现：复用 retrieve()，doc_id 过滤 + 300 字截断。"""
    question = str(args.get("question") or "").strip()
    doc_id = args.get("doc_id") or None
    top_k = int(args.get("top_k") or 5)
    top_k = max(1, min(top_k, 10))
    if not question:
        return []

    # 取更多候选，便于 doc_id 过滤后仍有足够结果
    hits = retrieve(question, top_k=max(top_k * 3, 15))
    if doc_id:
        hits = [h for h in hits if str(h.get("document_id") or "") == str(doc_id)]

    out = []
    for h in hits[:top_k]:
        out.append({
            "text": (h.get("content") or "")[:300],
            "page": h.get("page"),
            "source": h.get("source") or "",
            "document_id": h.get("document_id") or "",
            "image_path": h.get("image_path") or "",
        })
    return out


def _is_refusal(text: str) -> bool:
    """拒答判定：空答案或 LLM 以『[拒答]』开头。"""
    t = (text or "").strip()
    if not t:
        return True
    return t.startswith("[拒答]")


def run_agent(question: str) -> dict:
    """Agentic RAG 主流程：最多 3 轮工具循环后出中文答案。

    返回 {question, refused, answer, calls: [{tool, args, n_results}],
          images, citations}。images/citations 收集自所有轮次 search_knowledge
    返回块（图块 image_path 去重 + 页码引用），与 /ask 语义一致。
    """
    calls: list[dict] = []
    images: list[str] = []
    citations: list[str] = []
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    answer = ""

    for _round in range(_MAX_TOOL_ROUNDS):
        msg = _call_llm(messages, tools=_TOOLS)
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            answer = msg.get("content") or ""
            break
        # 逐个执行工具，结果以 role=tool 回填
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "list_documents":
                result = _tool_list_documents(args)
            elif name == "search_knowledge":
                result = _tool_search_knowledge(args)
                # 收集图块 image_path 与页码引用（跨所有轮次，去重）
                for h in result:
                    ip = (h.get("image_path") or "").strip()
                    if ip and ip not in images:
                        images.append(ip)
                    pg = h.get("page")
                    if pg is not None:
                        cit = format_citation(h.get("source") or "", pg)
                        if cit not in citations:
                            citations.append(cit)
            else:
                result = []
            calls.append({
                "tool": name,
                "args": args,
                "n_results": len(result) if isinstance(result, list) else 0,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id") or "",
                "content": json.dumps(result, ensure_ascii=False),
            })
    else:
        # 3 轮内每轮都在调工具 → 补一次不带工具声明，强制 LLM 收尾
        final_msg = _call_llm(messages, tools=None)
        answer = final_msg.get("content") or ""

    return {
        "question": question,
        "refused": _is_refusal(answer),
        "answer": answer,
        "calls": calls,
        "images": images,
        "citations": citations,
    }
