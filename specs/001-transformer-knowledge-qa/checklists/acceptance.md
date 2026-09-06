# Acceptance Checklist: 变压器检测维护知识库问答系统

**Feature**: 001-transformer-knowledge-qa
**Created**: 2026-08-29 ｜ 验收：2026-08-29
**依据**: spec.md（US1-3）+ docs/01 FR/NFR + docs/09 版本迭代

## A. 功能验收

- [x] A1 库内问答：问组分/取样/重复性 → 答案非空 + 页码正确
- [x] A2 引用溯源：引用 = 实际进 LLM 的块，可对账
- [x] A3 无据拒答：天气/数学等库外问题 → refused=true + 标准话术（10/10）
- [x] A4 上传异步：POST /upload 秒回 task_id，问答不阻塞
- [x] A5 六步进度：SSE 实时 + GET /pipeline 快照恢复（刷新不丢）
- [x] A6 文档管理：内置 3 份（不可删保护）+ 上传文档（可单删）
- [x] A7 检索调试：retrieve-test 四路结果可见

## B. 评测验收

- [x] B1 版本集 12 题：recall@5 0.9167 / MRR 0.6625 → V3 0.6694（recall 守住不降）
- [x] B2 MRR 归因：#4 题「重复性要求」从漏检被精排救回 rank2
- [x] B3 安全层：拒答 10/10（库外全拒，无编造）

## C. 质量验收

- [x] C1 验证手段：12 题评测 + 接口实测 + 检索测试台人工回归（本项目以评测驱动替代 pytest 覆盖）
- [x] C2 无硬编码密钥（DeepSeek key 在 ~/.deepseek.env）
- [x] C3 文档一致性：docs/01-10 与代码对齐（架构图/接口/版本迭代）
- [x] C4 决策留痕：docs/09 决策记录 D1-D6 + 问题日志

## D. 数据验收

- [x] D1 内置 88 块 = 17623:33 + 25438:35 + 27743:20（payload document_id 可精确统计）
- [x] D2 页码口径统一（PDF 物理页号，与引用/评测一致）
- [x] D3 上传文档独立集合（与内置隔离，可单独删除）
