# -*- coding: utf-8 -*-
"""补齐 RAGAs NaN：只重跑失败的(条目,指标)，填回 ragas_result.json"""
import json, os, sys, asyncio, time, urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
API = "http://127.0.0.1:8011"
BGE = "C:/Users/lizhihao/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
KEY = json.load(open(os.path.expanduser("~/.claude/settings.json"), encoding="utf-8"))["env"]["ANTHROPIC_AUTH_TOKEN"]

def login():
    req = urllib.request.Request(API + "/login", data=json.dumps({"username": "admin", "password": "admin123"}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=10).read().decode())["token"]

def ask(token, q):
    req = urllib.request.Request(API + "/ask", data=json.dumps({"question": q}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
    d = json.loads(urllib.request.urlopen(req, timeout=240).read().decode())
    ctxs = [b.get("content", "") for b in d.get("trace", {}).get("context_blocks", []) if b.get("content")]
    return d.get("answer", ""), ctxs

async def score_one(metric, sample):
    m = metric
    return await asyncio.wait_for(m.single_turn_ascore(sample), timeout=120)

async def main():
    # 找失败点位
    res = json.load(open(os.path.join(BASE, "ragas_result.json"), encoding="utf-8"))
    meta = json.load(open(os.path.join(BASE, "ragas_sample_meta.json"), encoding="utf-8"))
    chunks = json.load(open(os.path.join(BASE, "chunks_all.json"), encoding="utf-8"))
    text_of = {c["chunk_id"]: c["text"] for c in chunks}
    need = {}
    for i, s in enumerate(res["scores"]):
        for m, v in s.items():
            if not (isinstance(v, (int, float)) and v == v):  # NaN
                need.setdefault(i, []).append(m)
    print("需补:", {k + 1: v for k, v in need.items()})

    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics.base import Metric

    llm = LangchainLLMWrapper(ChatOpenAI(model="deepseek-chat", api_key=KEY,
                                         base_url="https://api.deepseek.com/v1", temperature=0))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=BGE, model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}))
    try:
        answer_relevancy.strictness = 1  # DeepSeek 不支持 n>1
    except Exception:
        pass
    metric_of = {"faithfulness": faithfulness, "answer_relevancy": answer_relevancy,
                 "context_precision": context_precision, "context_recall": None}
    for name, met in metric_of.items():
        if met is not None:
            met.llm = llm
            met.embeddings = emb

    token = login()
    for i, miss in need.items():
        e = meta[i]
        ans, ctxs = ask(token, e["question"])
        gold = text_of.get(e["gold_chunk_id"], "")
        sample = SingleTurnSample(user_input=e["question"], response=ans, retrieved_contexts=ctxs, reference=gold)
        for m in miss:
            met = metric_of[m]
            try:
                v = await score_one(met, sample)
                res["scores"][i][m] = round(float(v), 4)
                print(f"  [条目{i+1}] {m} = {round(float(v), 4)} ✅")
            except Exception as ex:
                print(f"  [条目{i+1}] {m} 失败: {str(ex)[:120]}")
        time.sleep(1)

    json.dump(res, open(os.path.join(BASE, "ragas_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("已回填 ragas_result.json")

if __name__ == "__main__":
    asyncio.run(main())
