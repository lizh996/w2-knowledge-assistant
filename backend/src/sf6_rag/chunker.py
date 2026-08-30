"""FR-002: 条款分块 + 元数据（page/section）。"""
from __future__ import annotations

import re

_SECTION_RE = re.compile(r"^(第\s*\d+\s*章|[1-9]\d?\s+[^\n]{2,20})", re.M)


def _section_number(header: str) -> str:
    """从标题提取章节号：'第6章' → '6'；'5 设备' → '5'。"""
    import re as _re
    m = _re.match(r"第\s*(\d+)\s*章", header)
    if m:
        return m.group(1)
    m = _re.match(r"(\d+)", header)
    return m.group(1) if m else "unknown"


def chunk_by_section(text: str, page: int = 1) -> list[dict]:
    """按条款（第X章 / 数字标题）切块，带 page/section 元数据。"""
    if not text.strip():
        return []

    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [{"text": text.strip(), "page": page, "section": "unknown"}]

    chunks = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            chunks.append({
                "text": section_text,
                "page": page,
                "section": _section_number(m.group(0)),
            })
    return chunks
