"""FR-001: PDF 纯文本提取，跳过乱码页。"""
from __future__ import annotations

import fitz
from pathlib import Path

READABLE_RATIO = 0.3  # 汉字占比阈值

# 常见汉字（GB 标准正文高频字）——乱码字（犐犆犛等）不在其中
_COMMON_CHARS = set(
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会"
    "可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部"
    "度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应"
    "开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正"
    "心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情"
    "者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长"
    "求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放"
    "西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压"
    "据察究界品形济么清省受讲亲需状华造验标准试检样含"
)


def is_readable_page(text: str) -> bool:
    """汉字占比 > 30% 且含常见字 → 可读页（排除乱码字）。"""
    if not text.strip():
        return False
    hanzi = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    ratio = hanzi / len(text.strip())
    if ratio <= READABLE_RATIO:
        return False
    # 乱码页（犐犆犛）即使码位在汉字区间，也不含常见字 → 判不可读
    common_hits = sum(1 for c in text if c in _COMMON_CHARS)
    return common_hits > 0


def extract_pages(pdf_path: str) -> list[dict]:
    """按页提取，返回可读页列表 [{page, text}]。"""
    doc = fitz.open(str(pdf_path))
    try:
        pages = []
        for i in range(doc.page_count):
            text = doc.load_page(i).get_text("text").strip()
            if is_readable_page(text):
                pages.append({"page": i + 1, "text": text})
        return pages
    finally:
        doc.close()
