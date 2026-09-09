# -*- coding: utf-8 -*-
"""任务2（LLM 部分）：用 DeepSeek 给 46131（未参与训练）生成 25-30 条评测问题。
每块 1 条 question（含 gold_chunk_id），覆盖不同 semantic_type，避开与 88 块训练集
主题重叠（46131 = 特高压变压器分接开关，属独立文档）。

调用方式对齐 gen_data.py：key 读 ~/.claude/settings.json 的 ANTHROPIC_AUTH_TOKEN，
model=deepseek-chat。若 402 余额不足会抛出并报告（eval_unseen.py 可回退规则生成）。

用法: D:\\an\\envs\\mineru\\python.exe gen_unseen_46131.py
输出: eval_unseen_46131.json
"""
import os
import sys
import json
import random
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = json.load(open(os.path.join(BASE, "chunks_46131.json"), encoding="utf-8"))
OUT = os.path.join(BASE, "eval_unseen_46131.json")
KEY = json.load(open(os.path.expanduser("~/.claude/settings.json"), encoding="utf-8"))["env"]["ANTHROPIC_AUTH_TOKEN"]

# 目标块分布（优先选语义信息量大的「标准要求/安全要求/参数查询」，各 1 条）
random.seed(42)
TARGET_TYPES = ["参数查询", "安全要求", "标准要求", "流程步骤", "概述"]
N_TARGET = 28  # 落在 25-30 区间


def call_llm(prompt, max_tokens=1600, retries=3):
    body = json.dumps({"model": "deepseek-chat",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.7}).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.deepseek.com/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode())["choices"][0]["message"]["content"]
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(3 * (i + 1))


PROMPT = """你是电力设备检测领域的数据标注专家。以下是国标文档《GB/T 46131 特高压变压器分接开关技术要求与试验方法》的一个知识块：
【知识块】(第{page}页, 语义类型 {semantic_type})
{text}

【任务】围绕这个知识块，写 1 个真实用户会问的问题。
【要求】
1. 问题必须能由这个知识块回答，且指向该块而非其他块；
2. 问题简短自然、像真实用户提问（口语化即可，不要求专业）；
3. 只输出问题本身（一句话），不要序号、引号、解释或 JSON。
"""


def pick_blocks():
    """按 semantic_type 分层抽取 N_TARGET 块（每类型至少 1 块，其余按占比补）。"""
    by_type = {}
    for c in CHUNKS:
        by_type.setdefault(c["semantic_type"], []).append(c)
    # 每类型先取 1 块
    picked = []
    for t in TARGET_TYPES:
        if by_type.get(t):
            b = random.choice(by_type[t])
            picked.append(b)
            by_type[t].remove(b)
    pool = [b for lst in by_type.values() for b in lst]
    random.shuffle(pool)
    while len(picked) < N_TARGET and pool:
        picked.append(pool.pop())
    # 去重 + 按 page 排序保证稳定
    seen, unique = set(), []
    for b in picked:
        if b["chunk_id"] not in seen:
            seen.add(b["chunk_id"])
            unique.append(b)
    unique.sort(key=lambda c: (c["page"], c["chunk_id"]))
    return unique


if __name__ == "__main__":
    blocks = pick_blocks()
    print(f"目标 {len(blocks)} 块（semantic_type 覆盖 {sorted({b['semantic_type'] for b in blocks})}）")

    eval_set = []
    done = 0
    for b in blocks:
        prompt = PROMPT.format(page=b["page"], semantic_type=b["semantic_type"],
                               text=b["text"][:512])
        try:
            q = call_llm(prompt).strip().strip('"').strip("'").strip()
            eval_set.append({"question": q, "gold_chunk_id": b["chunk_id"],
                             "page": b["page"], "source": b["source"],
                             "semantic_type": b["semantic_type"]})
            done += 1
        except Exception as e:
            print(f"  [FAIL] {b['chunk_id']} 生成失败: {str(e)[:120]}")
            # 402 余额不足直接中止，交由 eval_unseen.py 规则回退
            if "402" in str(e) or "Insufficient" in str(e):
                print("  [ABORT] DeepSeek 余额不足(402)，跳过 LLM 生成。")
                break
        if done % 5 == 0:
            print(f"  进度 {done}/{len(blocks)}")
        time.sleep(0.3)

    json.dump(eval_set, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[OK] 生成 {len(eval_set)} 条 → {OUT}")
    if not eval_set:
        print("[WARN] 生成 0 条，eval_unseen.py 将用规则生成回退。")
