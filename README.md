# 变压器检测维护知识库问答助手

> 专高六 W2/W3 结课综合项目 —— V1→V2→V3 迭代 + **W3 嵌入模型微调** + **MinerU 图文问答** + **Agentic RAG**
> 把 3 份国标（GB/T 17623 色谱法 / GB/T 25438 变压器参数 / GB/T 27743 设备检测）构建成可检索、可问答、可溯源、**可带图返回**的知识库。

## ✨ 核心能力

| 能力 | 说明 | 指标/证据 |
|---|---|---|
| 混合检索 | BGE-M3 **dense + learned sparse** 双路 → Qdrant named vectors → **FusionQuery(RRF, k=60)** → bge-reranker 精排 | 264 题干净集：recall@5 **0.890 → 0.928**（微调后）|
| 嵌入微调（W3）| BGE-M3 + **LoRA**（r=8/a=16/q-v，78.6 万参数）；全库重嵌 + 阈值重校准 + 可回滚 | E1 0.9280 / E2 0.9242（保留 E1）|
| 双层评测 | 检索层（recall@5/10/20、MRR、nDCG、四象限）+ 答案层（RAGAs 4 指标）| 四象限：救回 14 / 退化 4 / 净 +10 |
| 拒答安全 | dense+sparse 双阈值校准（0.42 / 3.0），库外明确拒答不编造 | 库外 10/10 拒答，库内 0 误拒 |
| MinerU 图文 | MinerU 版面解析 → 图语义块（正文图 + 表格内嵌图整表渲染）| 14 个图块；问"取气装置结构图"自动返回原图 |
| Agentic RAG | `/ask_agent` function calling：LLM 自主决定 **list_documents / search_knowledge / 直答 / 拒答** | 跨库对比两轮编排、无幻觉、images/citations 透传 |

## 快速开始

```bash
cd backend
set PYTHONPATH=src&& set RERANK_ENABLED=1&& set QUERY_REWRITE_ENABLED=1&& set BGE_MODEL_PATH=C:\Users\lizhihao\w2-knowledge-assistant\data\models\bge-m3-ft-v1&& D:\an\envs\langchain\python.exe -m uvicorn sf6_rag.api:app --port 8011
```

> ⚠️ **必须带 `BGE_MODEL_PATH`**（指向微调合并模型 `data/models/bge-m3-ft-v1`）：漏掉会退回 base 模型 —— 检索结果与拒答阈值（在 FT 向量空间校准）都会失配。

访问 http://127.0.0.1:8011/ （admin/admin123）｜接口文档 /docs、/redoc

## 版本指标

| 版本 | recall@5 | MRR | 核心改进 |
|---|---|---|---|
| V1 | 0.9167 | 0.6625 | MVP 基线（提问→答案→溯源）|
| V2 | 0.9167 | 0.6625 | trace + 复杂元素 + 复合检索 |
| V3 | 0.9167 | 0.6694 | 精排 + 查询改写 + 上下文组织 |
| **W3（微调）** | **0.9280** | **0.7310** | LoRA 嵌入微调 + 全库重嵌 + 阈值重校准（264 题干净集）|

> 口径说明：V1–V3 为 12 题快速回归集（base 模型）；W3 为 264 题干净集（微调 E1）。**跨行不可直接比较**——详见 `docs/09` 附录与 `docs/11`。

## 目录

- `docs/` 需求/架构/接口/迭代/指标/决策 + **11-微调实验记录**
- `backend/` FastAPI 服务（检索 / 生成 / 管线 / Agent）
- `frontend/` 问答页 + 运维控制台
- `eval/` 评测集 + 基线 + 脚本
- `finetune/` 微调数据/脚本/评测（E0/E1/E2、RAGAs）
- `scripts/` MinerU 解析、图块入库、清理等脚本
- `data/` 国标 PDF + 向量库 + 模型（大文件不入库）

## 关键文档

| 文档 | 内容 |
|---|---|
| `docs/11-微调实验记录.md` | W3 微调全流程 + 评测 + 踩坑（含 RAGAs）|
| `docs/09` 附录 | **评测环境与口径说明**（三套数字为什么不同）|
| `finetune/ragas_eval.py` | RAGAs 全链路评测（可复现）|
