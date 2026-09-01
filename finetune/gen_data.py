# -*- coding: utf-8 -*-
"""
数据生成脚本：从 88 块 chunk 生成
  1) 评测集 eval_retrieval_set.json  (300+ 条, question_kwd 种子 + LLM 补充)
  2) 训练数据 train.jsonl           (800+ 条, LLM 9 角度问法 + pos + hard_negative)
用法: D:\\an\\envs\\langchain\\python.exe gen_data.py
"""
import json, os, random, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(BASE, "chunks_all.json")
KEY = json.load(open(os.path.expanduser("~/.claude/settings.json"), encoding="utf-8"))["env"]["ANTHROPIC_AUTH_TOKEN"]
SEED = 42
random.seed(SEED)

def call_llm(prompt, max_tokens=1200, retries=3):
    body = json.dumps({"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.8}).encode()
    for i in range(retries):
        try:
            req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode())["choices"][0]["message"]["content"]
        except Exception as e:
            if i == retries - 1: raise
            time.sleep(3 * (i + 1))

# ---------- 1) 评测集：question_kwd 种子（0 API 调用） ----------
chunks = json.load(open(CHUNKS, encoding="utf-8"))
eval_set = []
for c in chunks:
    for q in c["question_kwd"]:
        eval_set.append({"question": q, "gold_chunk_id": c["chunk_id"],
                         "page": c["page"], "source": c["source"]})
print(f"评测集种子: {len(eval_set)} 条 (question_kwd)")

# ---------- 2) 训练数据：LLM 9 角度问法 ----------
PROMPT = """你是电力设备检测领域的数据标注专家。以下是国标文档的一个知识块：
【知识块】(第{page}页, 来源{source})
{text}

【任务】围绕这个知识块生成 9 个用户可能问的问题，角度要求：
1.直接问法 2.同义改写 3.口语化 4.专业术语 5.缩写/简称 6.故障场景 7.不完整表达 8.参数数值角度 9.操作流程角度
【要求】每个问题必须能用这个知识块回答；问题简短自然像真实用户提问；question_type 从 [method,parameter,safety,definition,operation,theory,comparison,standard] 选。
只输出 JSON 数组，格式: [{{"query":"...","question_type":"..."}}]，不要其他文字。"""

def gen_for_chunk(c):
    text = c["text"][:600]
    out = call_llm(PROMPT.format(text=text, page=c["page"], source=c["source"]))
    out = out.strip().strip("```json").strip("```").strip()
    qs = json.loads(out)
    # hard_negative: 同 source 其他块随机 2 个
    same_src = [x for x in chunks if x["source"] == c["source"] and x["chunk_id"] != c["chunk_id"]]
    negs = [x["text"][:300] for x in random.sample(same_src, min(2, len(same_src)))]
    rows = []
    for q in qs[:9]:
        rows.append({"query": q["query"], "pos": [text],
                     "neg": negs, "knowledge_id": c["chunk_id"],
                     "question_type": q.get("question_type", "standard")})
    return rows

if __name__ == "__main__":
    print(f"开始生成训练数据: {len(chunks)} 块 x 9 问法...")
    all_train = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(gen_for_chunk, c): c for c in chunks}
        done = 0
        for fut in futures:
            try:
                all_train.extend(fut.result())
            except Exception as e:
                print(f"  ❌ {futures[fut]['chunk_id']} 失败: {str(e)[:80]}")
            done += 1
            if done % 10 == 0:
                print(f"  进度 {done}/{len(chunks)}")
                json.dump(all_train, open(os.path.join(BASE, "train_partial.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    with open(os.path.join(BASE, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in all_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✅ 训练数据: {len(all_train)} 条 -> train.jsonl")

    # 补评测集到 300+（用训练数据里的 query 补）
    seen = {e["question"] for e in eval_set}
    for r in all_train:
        if r["query"] not in seen and len(eval_set) < 350:
            eval_set.append({"question": r["query"], "gold_chunk_id": r["knowledge_id"],
                             "page": 0, "source": "train-query"})
            seen.add(r["query"])
    json.dump(eval_set, open(os.path.join(BASE, "eval_retrieval_set.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"✅ 评测集: {len(eval_set)} 条 -> eval_retrieval_set.json")
