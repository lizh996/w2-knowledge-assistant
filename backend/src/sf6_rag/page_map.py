# -*- coding: utf-8 -*-
"""页码映射：PDF 物理页序 → 印刷页码（面向用户的显示页码）。

为什么需要：标准 PDF 用"文件第几张"（物理页），正文页脚印的是原书页码（印刷页），
两者相差 N 页（前 N 页封面/目次/前言）。用户与讲师翻页对照时按页脚号找，
若只给物理页会"翻不到答案"（实测 17623：答案在物理页 23 = 印刷页 17，
而页脚印 23 的那页是参考文献）。

映射由 scripts/gen_page_map.py 生成（众数偏移法，抗表格数字污染），存 data/page_map.json。
缺失映射时回退原行为（只显示物理页），保证向后兼容。
"""
from __future__ import annotations

import json
from pathlib import Path

_MAP_PATH = Path(__file__).resolve().parents[3] / "data" / "page_map.json"
_CACHE: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = json.loads(_MAP_PATH.read_text(encoding="utf-8"))
        except Exception:
            _CACHE = {}
    return _CACHE


def printed_page(source: str, phys_page: int | None) -> int | None:
    """返回印刷页码；无映射/无印刷页（封面类）返回 None。"""
    if not source or not phys_page:
        return None
    info = _load().get(source)
    if not info:
        return None
    return info.get("map", {}).get(str(phys_page))


def page_label(source: str, phys_page: int | None) -> str | None:
    """面向用户的页码文案。

    有印刷页码 → "第 17 页（PDF 第 23 页）"（读者按页脚找 = 17，按阅读器找 = 23，都能对上）
    无印刷页码 → None（调用方回退原格式）
    """
    pp = printed_page(source, phys_page)
    if pp and phys_page:
        return f"第 {pp} 页（PDF 第 {phys_page} 页）"
    return None
