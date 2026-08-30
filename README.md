# 变压器检测维护知识库问答助手

> 专高六 W2 结课综合项目（V1→V2→V3 三轮迭代）
> 把 3 份国标（GB/T 17623 色谱法 / GB/T 25438 变压器参数 / GB/T 27743 设备检测）构建成可检索、可问答、可溯源的知识库。

## 快速开始

```bash
cd backend
set PYTHONPATH=src && set RERANK_ENABLED=1 && set QUERY_REWRITE_ENABLED=1
D:\an\envs\langchain\python.exe -m uvicorn sf6_rag.api:app --port 8011
```

访问 http://127.0.0.1:8011/ （admin/admin123）

## 版本指标

| 版本 | recall@5 | MRR | 核心改进 |
|---|---|---|---|
| V1 | 0.9167 | 0.6625 | MVP 基线（提问→答案→溯源）|
| V2 | 0.9167 | 0.6625 | trace + 复杂元素 + 复合检索 |
| V3 | 0.9167 | 0.6694 | 精排 + 查询改写 + 上下文组织 |

## 目录

- docs/ 需求/架构/接口/迭代/指标/决策
- data/ 3 份国标 PDF
- backend/ FastAPI 服务
- frontend/ 问答页 + 运维控制台
- eval/ 评测集 + 基线 + 脚本
