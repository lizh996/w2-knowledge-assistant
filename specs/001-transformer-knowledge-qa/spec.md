# Feature Specification: 变压器检测维护知识库问答系统

**Feature Branch**: `001-transformer-knowledge-qa`
**Created**: 2026-08-29
**Status**: Approved（M1-M5 交付，spec 对齐 V3 实际落地）
**Input**: 基于 GB/T 17623-2026 / GB/T 25438-2026 / GB/T 27743-2025 三份变压器国标构建带页码溯源、无据拒答的知识库问答系统

> 本 spec 已对齐落地实际（非初始蓝图）。数据链路：fitz 解析（MinerU 兜底）→ 清洗 → 结构分块 88 块（≤512 token + 语义类型 + Q2Q）→ BGE-M3 dense+sparse → Qdrant embedded（transformer_kb_v1 + 上传独立集合）。

---

## 当前实现状态（M5 交付快照）

| 层 | 实现 | 现状 |
|---|---|---|
| 解析 | fitz（秒级） + MinerU vlm 兜底 | ✅ 完成 |
| 清洗 | clean_pages（乱码检测 + 文本规范化）| ✅ 完成 |
| 分块 | 结构分块 ≤512 token + 语义类型 + Q2Q | ✅ 88 块（17623:33 / 25438:35 / 27743:20）|
| 向量化 | BGE-M3 dense(1024) + sparse | ✅ 完成 |
| 存储 | Qdrant embedded（transformer_kb_v1 88 条 + 上传独立集合）| ✅ 完成 |
| 检索 | 八步链路：改写 → RRF 粗排15 → 精排取5 → 上下文组织 | ✅ 完成（MRR 0.6694）|
| 拒答 | dense + sparse 双阈值（实测校准）| ✅ 完成（10/10）|
| 生成 | DeepSeek（deepseek-v4-flash）带页码引用 | ✅ 完成 |
| API | FastAPI（ask/upload/SSE/admin 等 22 路由）| ✅ 完成 |
| 评测 | 12 题版本集 + 拒答集 | ✅ 完成（recall@5 0.9167 / MRR 0.6694）|

---

## User Stories & Acceptance

### US1 - 知识库问答（P1）
运维人员现场用自然语言查变压器检测标准，得到带页码的可靠答案。
- Given 库内问题，When 提交 /ask，Then 返回答案 + 页码引用（引用 = 实际进 LLM 的块）
- Given 库外问题（天气/数学），When 提交，Then refused=true + 「知识库中未找到相关依据」，无编造

### US2 - 上传扩展（P1）
用户上传新 PDF，后台异步构建，页面实时看六步进度。
- Given 上传 PDF，When POST /upload，Then 秒回 task_id（不阻塞问答）
- Given 任务进行中，When 查 /pipeline/{id}，Then 状态六步可查（SQLite 持久化刷新不丢）
- Given 页面打开，When 有进度变化，Then SSE 事件推送（done 自动断开）

### US3 - 检索可解释（P2）
- Given 任意问题，When 看 trace / 检索测试台，Then 八步耗时与四路结果（dense/sparse/RRF/精排）可见

---

## 功能范围（FR 明细见 docs/01-需求分析.md）

FR1-18 全部落地，代码落点见 docs/01。范围外（YAGNI）：多租户 / 移动端 / 父子分块 / Milvus / 图像问答。
