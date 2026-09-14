# -*- coding: utf-8 -*-
"""页码映射单元测试（印刷页 vs PDF 物理页）。

守住"引用页码可核对"这条红线：17623 的答案在 PDF 物理第 23 页 = 印刷第 17 页。
"""
from sf6_rag.page_map import page_label, printed_page


def test_printed_page_offset_17623():
    """17623 偏移 6：物理 23 → 印刷 17（此前读者按页脚 23 找会翻到参考文献）。"""
    assert printed_page("GB_T_17623-2026.pdf", 23) == 17
    assert printed_page("GB_T_17623-2026.pdf", 7) == 1
    assert printed_page("GB_T_17623-2026.pdf", 29) == 23


def test_printed_page_offset_27743():
    """27743 偏移 4（不同文档偏移不同，必须逐文档生成）。"""
    assert printed_page("GB_T_27743-2025.pdf", 5) == 1


def test_page_label_format():
    """双标注文案：读者按页脚号或阅读器页序都能对上。"""
    assert page_label("GB_T_17623-2026.pdf", 23) == "第 17 页（PDF 第 23 页）"


def test_page_label_cover_pages_return_none():
    """封面/目次等无印刷页码的页 → 返回 None（调用方回退 "PDF 第 X 页"）。"""
    assert printed_page("GB_T_17623-2026.pdf", 3) is None
    assert page_label("GB_T_17623-2026.pdf", 3) is None


def test_page_label_unknown_document_returns_none():
    """未映射文档（如上传 PDF）不报错、返回 None（向后兼容）。"""
    assert page_label("some_upload.pdf", 12) is None
    assert page_label("", 12) is None
