# -*- coding: utf-8 -*-
"""RAGAs 全链路评测（真实服务 8011 + DeepSeek 阅卷）
流程: 备卷(抽样) -> 收答卷(调 /ask 拿 answer+contexts) -> 阅卷(DeepSeek 4指标) -> 出成绩
用法: D:\\an\\envs\\langchain_ragas\\python.exe ragas_eval.py [条数, 默认20]
"""
import json, os, sys, random, time, urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
API = "http://127.0.0.1:8011"
BGE = "C:/Users/lizhihao/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
KEY = json.load(open(os.path.expanduser("~/.claude/settings.json"), encoding="utf-8"))["env"]["ANTHROPIC_AUTH_TOKEN"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

def login():
    req = urllib.request.Request(API + "/login", data=json.dumps({"username": "admin", "password": "admin123"}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=10).read().decode())["token"]

def ask(token, q):
    req = urllib.request.Request(API + "/ask", data=json.dumps({"question": q}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
    d = json.loads(urllib.request.urlopen(req, timeout=240).read().decode())
    ctxs = [b.get("content", "") for b in d.get("trace", {}).get("context_blocks", []) if b.get("content")]
    return d.get("answer", ""), ctxs, d.get("refused")

def main():
    evals = json.load(open(os.path.join(BASE, "eval_retrieval_set.json"), encoding="utf-8"))
    evals = [e for e in evals if e.get("source") != "train-query"]
    chunks = json.load(open(os.path.join(BASE, "chunks_all.json"), encoding="utf-8"))
    text_of = {c["chunk_id"]: c["text"] for c in chunks}

    # (1) 备卷: 固定种子、按知识块去重抽样
    random.seed(42)
    seen, picked = set(), []
    pool = list(evals)
    random.shuffle(pool)
    for e in pool:
        if e["gold_chunk_id"] not in seen:
            seen.add(e["gold_chunk_id"])
            picked.append(e)
        if len(picked) >= N:
            break
    print(f"(1) 备卷: 抽 {len(picked)} 条 (覆盖 {len(seen)} 个不同知识块)")

    # (2) 收答卷: 真实调 8011 服务
    token = login()
    qs, answers, ctxss, refs = [], [], [], []
    for i, e in enumerate(picked, 1):
        ans, ctxs, refused = ask(token, e["question"])
        gold = text_of.get(e["gold_chunk_id"], "")
        print(f"   [{i}/{len(picked)}] refused={refused} ctx={len(ctxs)} | {e['question'][:36]}")
        qs.append(e["question"]); answers.append(ans); ctxss.append(ctxs); refs.append(gold)
        time.sleep(0.3)
    json.dump(picked, open(os.path.join(BASE, "ragas_sample_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # (3) 阅卷: DeepSeek 打分
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings

    llm = LangchainLLMWrapper(ChatOpenAI(model="deepseek-chat", api_key=KEY,
                                         base_url="https://api.deepseek.com/v1", temperature=0))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=BGE, model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}))
    t0 = time.time()
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    samples = [SingleTurnSample(user_input=q, response=a, retrieved_contexts=c, reference=r)
               for q, a, c, r in zip(qs, answers, ctxss, refs)]
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                      llm=llm, embeddings=emb)
    out = {"n": len(qs), "elapsed_s": round(time.time() - t0, 1), "scores": result.scores}
    json.dump(out, open(os.path.join(BASE, "ragas_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n(3) 阅卷完成:", json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
