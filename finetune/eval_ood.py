# -*- coding: utf-8 -*-
"""任务3：OOD 检索子集评测（讲师 V2.0 §8.4）
从 264 干净集抽取 20 条，人工加「错别字 / 口语化 / 新术语」变体（gold 不变），
base vs ft(E1) 各跑 recall@5（dense 检索），确认 FT 在 OOD 上不明显退化。

变体为人工构造，硬编码在 OOD_ITEMS（含原题 + gold_chunk_id + 变体 + 变体类型），
可复现、不依赖抽样。gold_chunk_id 与 eval_retrieval_set.json 完全一致。

用法: D:\\an\\envs\\mineru\\python.exe eval_ood.py
输出: eval_ood.json（评测集）+ eval_ood_result.json（base vs ft recall@5）
"""
import os
import sys
import json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _eval_common import (
    BASE_DIR, load_model, encode, dense_scores, recall_at,
)

# (gold_chunk_id, 原题, 变体, 变体类型 typo/colloquial/newterm)
OOD_ITEMS = [
    ("k1_32637387", "GB/T17623—2026由哪个标准化技术委员会归口？", "GB/T17623-2026归哪个标委会管？", "newterm"),
    ("k1_32637387", "GB/T17623—2026的起草单位有哪些？", "GB/T17623-2026是哪些单位一块起草的？", "colloquial"),
    ("k1_358cea63", "试油分析需要分析几次？", "试油分析得做几遍？", "colloquial"),
    ("k1_358cea63", "试油分析的精密度应满足什么要求？", "试油分析的精密度有啥要求？", "colloquial"),
    ("k1_6a331b49", "分配系数测定时振荡和静置时间分别是多少？", "测分配系数时振荡和静置各要多久？", "colloquial"),
    ("k1_6d060947", "如何用二次溶解平衡法测定气体分配系数？", "二次溶解平衡法怎么测气体分配系数？", "colloquial"),
    ("k1_90d29c6f", "标油中气体组分实测浓度如何测定？", "标油里气体组分实测浓度咋测？", "colloquial"),
    ("k1_ba8348ee", "如何计算50℃下平衡气的体积？", "50℃下平衡气体积咋个算？", "colloquial"),
    ("k1_e4a96047", "测定溶解气体时玻璃注射器的气密性要求是什么？", "测溶解气体时玻璃注射器气密性有啥要求？", "colloquial"),
    ("k2_0595d260", "35kV变压器分接范围调整有哪些具体要求？", "35千伏变压器分接范围调整有什么具体要求？", "newterm"),
    ("k2_2a31e3b1", "配电变压器相电阻不平衡率允许的最大值是多少？", "配电变压器相电阻不平衡率最大能到多少？", "colloquial"),
    ("k2_2a31e3b1", "除配电变压器外的电力变压器相电阻不平衡率限值是多少？", "除配电变压器外，电力变压器相电阻不平衡率限值多少？", "colloquial"),
    ("k2_36e8f225", "GB/T1.1-2020标准文件的起草规则是什么？", "GB/T1.1-2020这个标准文件起草有啥规则？", "colloquial"),
    ("k2_40984f78", "变压器接地处需要设置什么标识？", "变压器接地的地方要设什么标记？", "typo"),
    ("k2_4753058d", "波纹式油箱变压器的压力变形试验压力值是多少？", "波纹油箱变压器的压力变形试验压力是多少？", "typo"),
    ("k2_69a99ec1", "不同温度下测得的直流绝缘电阻值如何换算？", "不同温度下测到的直流绝缘电阻值怎么换算？", "colloquial"),
    ("k3_24e85f2f", "哪些企业参与了该电力变压器标准的制定？", "哪些公司参与了该电力变压器标准的制定？", "newterm"),
    ("k3_2f487d32", "变压器专用设备的工作条件有哪些？", "变压器专用设备的工作条件都有哪些？", "colloquial"),
    ("k3_8d01c301", "可凝性气体水平管道安装坡度如何测量？", "可凝性气体水平管道安装坡度怎么量？", "colloquial"),
    ("k3_f79555d4", "GB/T27743—2025的起草人有哪些？", "GB/T27743-2025是谁起草的？", "colloquial"),
]

if __name__ == "__main__":
    CHUNKS = json.load(open(os.path.join(BASE_DIR, "chunks_all.json"), encoding="utf-8"))
    chunk_texts = [c["text"] for c in CHUNKS]
    gold_idx = {c["chunk_id"]: i for i, c in enumerate(CHUNKS)}

    ood_set = [{"gold_chunk_id": g, "original": o, "variant": v, "ood_type": t}
               for g, o, v, t in OOD_ITEMS]
    # 校验 gold 都在候选库
    missing = [x for x in ood_set if x["gold_chunk_id"] not in gold_idx]
    if missing:
        print(f"[WARN] {len(missing)} 条 gold 不在候选库: {missing}")

    gold_list = [gold_idx.get(x["gold_chunk_id"], -1) for x in ood_set]
    q_texts = [x["variant"] for x in ood_set]

    # 输出评测集（gold 不变）
    json.dump(ood_set, open(os.path.join(BASE_DIR, "eval_ood.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"OOD 评测集: {len(ood_set)} 条（typo/colloquial/newterm 覆盖）")
    type_dist = {}
    for x in ood_set:
        type_dist[x["ood_type"]] = type_dist.get(x["ood_type"], 0) + 1
    print(f"  变体类型分布: {type_dist}")

    result = {"ood_n": len(ood_set), "type_dist": type_dist, "results": {}}
    for mode in ("base", "ft"):
        print(f"\n=== [{mode}] 编码 {len(chunk_texts)} 块 + {len(q_texts)} 条 OOD query ===")
        model = load_model(mode)
        c_dense, _ = encode(model, chunk_texts)
        q_dense, _ = encode(model, q_texts)
        scores = dense_scores(q_dense, c_dense)
        r5, h5, n5 = recall_at(scores, gold_list, 5)
        result["results"][mode] = {"recall@5": round(r5, 4), "hits@5": h5, "valid_n": n5}
        print(f"  recall@5 = {r5:.4f} ({h5}/{n5})")

    r5_base = result["results"]["base"]["recall@5"]
    r5_ft = result["results"]["ft"]["recall@5"]
    result["delta"] = {"recall@5": round(r5_ft - r5_base, 4)}
    # OOD 小样本(20)下允许单条噪声(5pp)，降幅 ≤1 题视为不显著退化
    h5_base = result["results"]["base"]["hits@5"]
    h5_ft = result["results"]["ft"]["hits@5"]
    result["conclusion"] = (
        "FT 在 OOD 上不明显退化（差异在单样本噪声内）"
        if h5_ft - h5_base >= -1
        else "FT 在 OOD 上明显退化，需复查"
    )

    out = os.path.join(BASE_DIR, "eval_ood_result.json")
    json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[OK] 结果写入 {out}")
    print(json.dumps(result, ensure_ascii=False, indent=1))
