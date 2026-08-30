"""FR-005/FR-006: 引用组装 + 无据拒答。"""
from __future__ import annotations

from dataclasses import dataclass, field

FALLBACK_TEXT = "知识库中未找到相关依据"


@dataclass
class Reference:
    source: str
    page: int


@dataclass
class Answer:
    text: str
    citations: list = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        """支持 Answer == '字符串' 比较（兼容测试与调用方）。"""
        if isinstance(other, str):
            return self.text == other
        if isinstance(other, Answer):
            return self.text == other.text and self.citations == other.citations
        return NotImplemented


def build_reference(source: str, page: int) -> Reference:
    """构造引用（文档名 + 页码）。"""
    return Reference(source=source, page=page)


def format_citation(source: str, page: int) -> str:
    """引用格式 [文档名 · 第 X 页]。"""
    return f"[{source} · 第 {page} 页]"


def build_fallback() -> Answer:
    """无依据 → 固定拒答文案，引用为空。"""
    return Answer(text=FALLBACK_TEXT, citations=[])
