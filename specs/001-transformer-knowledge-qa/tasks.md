# Tasks: 变压器检测维护知识库问答系统

**Feature**: 001-transformer-knowledge-qa
**Created**: 2026-08-29
**依据**: plan.md + docs/09 版本迭代
**状态**: V1-V3 全部落地（git 提交对应见 docs/09）

## Phase V1 - 最小闭环（git 591dd25）
- [x] T001 项目骨架（backend/src/sf6_rag/ + FastAPI + uvicorn）
- [x] T002 数据管线：fitz 解析 3 份国标 → 清洗 → 结构分块 88 块（≤512 token）
- [x] T003 BGE-M3 向量化（dense 1024 + sparse）→ Qdrant 入库（transformer_kb_v1）
- [x] T004 /ask 问答：检索 → DeepSeek 生成 → 页码引用
- [x] T005 拒答红线：dense 阈值 + 「知识库中未找到相关依据」
- [x] T006 12 题版本集评测：recall@5 0.9167 / MRR 0.6625 / 拒答 10/10（git b066501）

## Phase V2 - 增强（可观测 + 调度 + 复合检索）
- [x] T007 trace 八步可视化（每步耗时写进 /ask 响应）
- [x] T008 上传异步三件套：POST /upload + GET /pipeline/{id}（SQLite）+ SSE events
- [x] T009 复杂元素提取（表=原子块 / 图=图语义块 / 公式=LaTeX）
- [x] T010 复合检索 dense+sparse → Qdrant FusionQuery(RRF)
- [x] T011 检索测试台（retrieve-test 四路结果）

## Phase V3 - 进阶（找得准）
- [x] T012 bge-reranker 精排：粗排 15 → 取 5（RERANK_ENABLED 开关）
- [x] T013 查询改写（LLM 口语→术语，开关+缓存+兜底）
- [x] T014 上下文组织（负分过滤 + embedding 去重 + token 预算 + MIN_KEEP=2）
- [x] T015 引用诚实化（citations 只列实际进 LLM 的块）
- [x] T016 MRR 0.6625 → 0.6694（#4 题精排救回 rank2）
