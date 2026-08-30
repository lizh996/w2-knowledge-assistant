# -*- coding: utf-8 -*-
"""DeepSeek LLM 封装（Q2Q 生成 + 语义兜底）。

对齐《分块方案》⑦/④：
- Q2Q 用 DeepSeek 生成 important_kwd(3) + question_kwd(3)
- 语义类型"规则为主 + LLM 兜底"

设计原则：失败不阻塞。所有调用 30s 超时 + 1 次重试，异常返回 None，
由调用方降级到规则模板（呼应方案⑦"失败不阻塞"）。

用法：
    from llm_client import gen_q2q, classify_semantic
"""
from __future__ import annotations

import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    from openai import OpenAI
except ImportError as e:  # 缺依赖时优雅降级（OpenAI/Dotenv 未装 → 全走规则）
    OpenAI = None
    load_dotenv = None

_load_dotenv_called = False

SEMANTIC_TYPES = ["参数查询", "流程步骤", "标准要求", "术语定义", "图语义", "安全要求", "概述"]


def _ensure_env():
    """加载 .env（幂等）。"""
    global _load_dotenv_called
    if not _load_dotenv_called and load_dotenv is not None:
        # 密钥放项目外（用户主目录），避免明文进入 git 历史
        load_dotenv(os.path.expanduser("~/.deepseek.env"))
        _load_dotenv_called = True


def _client():
    if OpenAI is None:
        return None
    _ensure_env()
    key = os.environ.get("DEEPSEEK_API_KEY")
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not key:
        return None
    return OpenAI(api_key=key, base_url=base)


def chat(messages: list[dict], json_mode: bool = False, timeout: float = 30.0):
    """发一次对话请求。失败/超时返回 None（不抛异常）。"""
    client = _client()
    if client is None:
        return None
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    kwargs = {"model": model, "messages": messages, "timeout": timeout, "max_tokens": 800}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(2):  # 1 次重试
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == 1:
                print(f"[llm_client] 调用失败(已重试): {type(e).__name__}: {str(e)[:120]}")
                return None
            time.sleep(1.0)


def gen_q2q(content: str, section: str):
    """LLM 生成 3 关键词 + 3 问题。失败返回 None（走规则兜底）。"""
    sys_prompt = (
        "你是电力行业标准检索知识库的标注员。阅读给定 chunk 的正文，生成检索增强用的关键词和问题句。"
        "只输出 JSON，格式：{\"important_kwd\": [3个关键词], \"question_kwd\": [3个问题句]}。"
        "关键词要具体（物质名/参数名/标准号），问题句要模拟用户真实问法，不要泛泛的『该部分讲了什么』。"
    )
    user_prompt = f"章节：{section}\n正文：\n{content[:1200]}"
    text = chat(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        json_mode=True,
    )
    if not text:
        return None
    try:
        data = json.loads(text)
        ik = data.get("important_kwd", [])
        qk = data.get("question_kwd", [])
        if isinstance(ik, list) and isinstance(qk, list) and ik and qk:
            return {
                "important_kwd": [str(x)[:40] for x in ik][:3],
                "question_kwd": [str(x)[:60] for x in qk][:3],
            }
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return None


def classify_semantic(content: str, section: str):
    """LLM 语义兜底：7 类里选一个。失败返回 None（走规则/概述）。"""
    sys_prompt = (
        "你是电力行业标准分类员。把给定 chunk 归入以下 7 类之一，只输出类名，不要解释：\n"
        + "、".join(SEMANTIC_TYPES)
    )
    user_prompt = f"章节：{section}\n正文：\n{content[:800]}"
    text = chat(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        json_mode=False,
    )
    if not text:
        return None
    t = text.strip()
    for s in SEMANTIC_TYPES:
        if s in t:
            return s
    return None
