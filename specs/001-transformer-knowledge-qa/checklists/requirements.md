# 验收清单：变压器检测维护知识库问答系统（对齐 docs/01 需求）

**日期**: 2026-08-29
**用途**: 对齐 spec.md 功能需求 FR1-18 / NFR1-9，逐项验收

---

## 一、功能需求（FR）

| # | 需求 | 验收标准 | 状态 | 落点 |
|---|---|---|---|---|
| FR1 | 文档解析 | fitz 正常 PDF / MinerU 兜底乱码扫描 | ✅ | extract.py |
| FR2 | 复杂元素 | 表=原子块/图=语义块/公式=LaTeX | ✅ | pipeline.build_chunks |
| FR3 | 数据清洗 | 文本规范化 + 乱码检测门槛 | ✅ | clean_pages |
| FR4 | 分块 | 结构分块 ≤512 token + 语义类型 + 页码 | ✅ | _split_by_tokens |
| FR5 | 向量化 | BGE-M3 dense(1024)+sparse | ✅ | retrieve._get_model |
| FR6 | 向量库 | Qdrant embedded；内置 1 集合 + 上传独立集合 | ✅ | upsert_collection |
| FR7 | 复合检索 | dense+sparse → RRF 粗排 15 | ✅ | _query_rrf_collections |
| FR8 | 查询改写 | LLM 口语→术语，fail-open | ✅ | _rewrite_query |
| FR9 | 精排 | bge-reranker cross-encoder 取 5（开关）| ✅ | _get_reranker |
| FR10 | 上下文组织 | 负分过滤+去重+token 预算（MIN_KEEP=2）| ✅ | _organize_context |
| FR11 | 问答 | 八步链路 + trace + 页码答案 | ✅ | api.ask |
| FR12 | 无依据拒答 | 库外 refused=true（dense+sparse 双阈值实测校准）| ✅ | reject 判定 |
| FR13 | 文件上传 | PDF → 秒回 task_id → 异步线程池 | ✅ | api.upload |
| FR14 | 构建展示 | 六步状态机 + SSE + 轮询兜底 + SQLite 刷新不丢 | ✅ | api.pipeline/events |
| FR15 | 知识库管理 | 文档列表/删除（内置保护）+ 分块查看 | ✅ | api.documents/chunks |
| FR16 | 检索调试 | retrieve-test 四路结果 | ✅ | api.retrieve-test |
| FR17 | 系统健康 | 三查体检 + 评测历史 | ✅ | api.admin/health |
| FR18 | 评测 | recall@5/MRR；版本集 12 题 | ✅ | eval/ |

## 二、非功能需求（NFR）

| # | 需求 | 验收标准 | 状态 |
|---|---|---|---|
| NFR1 | 可溯源 | 答案带页码；引用=真正进 LLM 的块 | ✅ |
| NFR2 | 不编造 | 库外问题拒答 100%（10/10）| ✅ |
| NFR3 | 可评测 | recall@5/MRR 量化；12 题版本集 | ✅ |
| NFR4 | 原子性 | 构建失败不产生半成品（整体 upsert）| ✅ |
| NFR5 | 可回滚 | 模型/开关可切换，原模型不覆盖 | ✅ |
| NFR6 | 响应 | 问答秒级；上传异步不阻塞 | ✅ |
| NFR7 | 可复现 | SQLite 任务记录；评测脚本可重跑 | ✅ |
| NFR8 | 不过度设计 | 单机教学版（线程池/SQLite/embedded Qdrant/原生前端）| ✅ |
| NFR9 | 可迁移 | Qdrant embedded → API 只改连接 | ✅ |
