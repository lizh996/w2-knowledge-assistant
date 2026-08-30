
import json, os, sys
sys.path.insert(0, r"C:\Users\lizhihao\w2-knowledge-assistant\backend\src")
os.environ.setdefault("SF6_RAG_RUNTIME_DIR", r"C:\Users\lizhihao\w2-knowledge-assistant\data")
from sf6_rag.retrieve import should_reject as _sr
from sf6_rag.api import _build_generation_messages  # noqa
# 直接用拒答判定函数
from sf6_rag.retrieve import dense_similarity, sparse_similarity, REJECT_DENSE_THRESHOLD, REJECT_SPARSE_THRESHOLD

with open(r"C:\Users\lizhihao\w2-knowledge-assistant\eval\safety_set.json", encoding="utf-8") as f:
    safety = json.load(f)

def is_rejected(q):
    d = dense_similarity(q)
    if d >= REJECT_DENSE_THRESHOLD:
        return False
    s = sparse_similarity(q)
    return s < REJECT_SPARSE_THRESHOLD

correct = 0
for i, item in enumerate(safety, 1):
    rej = is_rejected(item["question"])
    expect_rej = item["type"] == "out_of_scope"
    ok = (rej == expect_rej)
    if ok: correct += 1
    print(f"{'✅' if ok else '❌'} #{i} {item['question'][:30]} 拒答={rej} (期望{'拒' if expect_rej else '放'})")
print(f"\n拒答准确率 = {correct}/{len(safety)} = {correct/len(safety):.2%}")
