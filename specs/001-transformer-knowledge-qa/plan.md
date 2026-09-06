# Implementation Plan: 变压器检测维护知识库问答系统

**Feature**: 001-transformer-knowledge-qa
**Created**: 2026-08-29
**Status**: Approved

---

## 1. 技术栈（按版本演进，详细选型理由见 docs/02）

| 版本 | 新增技术 | 为什么 |
|---|---|---|
| V1 | fitz / BGE-M3 / Qdrant embedded / DeepSeek / FastAPI / 原生前端 | 最小闭环必需品（备选：Chroma 无原生 sparse ❌、Milvus 运维重 ❌）|
| V2 | trace 可视化 / 线程池+SQLite+SSE / RRF 融合 / 复杂元素提取 | 看得见、不丢、找得全 |
| V3 | bge-reranker 精排 / 查询改写 / 上下文组织 / 引用诚实化 | 找得准（多文档去噪）|

## 2. 架构演进

```
V1 六层闭环：用户→前端→API→解析分块向量化→朴素检索+生成→Qdrant
V2 +调度层(线程池/SQLite/SSE) + trace + dense+sparse RRF + 复杂元素
V3 七层终图：用户→API→调度→流水线六步→检索八步→模型层(虚线)→数据层
```

## 3. 目录结构

```
w2-knowledge-assistant/
├── backend/src/sf6_rag/    # api/auth/generate/retrieve/pipeline/extract/chunker/eval_metrics
├── frontend/               # index.html（问答）+ admin.html（运维控制台）
├── data/                   # qdrant/ + uploads/
├── docs/                   # 01-10 交付文档
├── eval/                   # 版本评测集（12 题等）
└── specs/                  # 本文档（spec-kit 产物链）
```

## 4. 实现顺序

1. V1 MVP：数据管线 + /ask 问答 + 12 题基线
2. V2：trace + 上传异步三件套 + RRF + 复杂元素
3. V3：精排/改写/上下文组织/诚实引用 → MRR 提升
4. 后续：嵌入模型微调（独立 W3 阶段，另见讲师 W3 方案与 finetune/ 评测）
