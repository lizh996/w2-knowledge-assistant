"""评测指标：recall@k / MRR。"""
from __future__ import annotations


def recall_at_k(hits: list[bool], k: int) -> float:
    """recall@k = 命中数 / 问题数。"""
    if not hits:
        return 0.0
    return sum(1 for h in hits[:k] if h) / len(hits)


def mean_reciprocal_rank(ranks: list[int]) -> float:
    """MRR = 平均(1/第一个正确答案排名)。"""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks) / len(ranks)
