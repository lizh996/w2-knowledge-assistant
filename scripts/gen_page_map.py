# -*- coding: utf-8 -*-
"""生成 PDF 物理页 → 印刷页码映射表（国标页脚印刷页码 vs PDF 页序）。

背景：标准 PDF 用"物理页序"(文件第几张)，正文页脚印的是"印刷页码"（与前者差 N 页，
前 N 页为封面/目次/前言，无印刷页码）。用户/讲师按页脚号翻页时会对不上。

方法（稳健）：
  1. 逐页取页脚尾部数字行（页脚形如 "<页码>" + "GB/T xxxx—yyyy"，页码数字可能被拆成两块）
  2. 对每页算候选偏移 = 物理页 - 印刷页（过滤不合理值：偏移 负数或 >20、印刷页 > 物理页）
  3. 取**众数偏移**作为该文档偏移（抗表格数字污染）
  4. 输出 {物理页: 印刷页}（偏移后 <1 的页 = 封面/目次等，标 null）

用法: D:\an\envs\langchain\python.exe scripts/gen_page_map.py
输出: data/page_map.json
"""
from __future__ import annotations
import json, os, re, sys
from collections import Counter
from pathlib import Path

BASE = Path(r"C:\Users\lizhihao\w2-knowledge-assistant")
DATA = BASE / "data"
OUT = DATA / "page_map.json"


def footer_printed_page(page) -> int | None:
    """从单页页脚提取印刷页码。

    页脚结构（实测）：[... , "<数字>", ..., "GB/T xxxx—yyyy"]
    例：物理页 10 尾部 ['4','GB/T17623—2026'] → 印刷 4
        物理页 23 尾部 ['7','1','GB/T17623—2026'] → 印刷 17（提取顺序个位在前，需反转拼接）
    """
    lines = [l.strip() for l in page.get_text().splitlines() if l.strip()]
    tail = lines[-5:]
    # 定位 GB/T 行（页脚标志）
    idx = None
    for i in range(len(tail) - 1, -1, -1):
        if re.search(r"GB\s*/?\s*T", tail[i]):
            idx = i
            break
    if idx is None:
        return None
    # 取 GB/T 行之前的连续数字行
    nums: list[str] = []
    j = idx - 1
    while j >= 0 and re.fullmatch(r"\d{1,3}", tail[j]):
        nums.append(tail[j])
        j -= 1
    if not nums:
        return None
    s = "".join(reversed(nums))              # 个位在前 → 反转还原
    return int(s) if s.isdigit() else None


def build(pdf: Path) -> dict:
    import fitz
    doc = fitz.open(str(pdf))
    cands: list[tuple[int, int, int]] = []   # (物理页, 印刷页, 偏移)
    for i, page in enumerate(doc):
        phys = i + 1
        printed = footer_printed_page(page)
        if not printed:
            continue
        off = phys - printed
        # 过滤：偏移必须合理（正文页码不会超过物理页数、不能为负、不能过大）
        if 0 <= off <= 20 and 1 <= printed <= phys:
            cands.append((phys, printed, off))
    if not cands:
        return {"offset": 0, "map": {}, "note": "未识别到页脚页码，全部回退物理页"}
    offset = Counter(o for _, _, o in cands).most_common(1)[0][0]
    total = len(doc)
    mapping = {}
    for phys in range(1, total + 1):
        printed = phys - offset
        mapping[str(phys)] = printed if printed >= 1 else None
    # 一致性检查：与页码提取结果不符的页数（噪声页）
    mismatch = [p for p, pr, o in cands if o != offset]
    return {"offset": offset, "total_pages": total, "map": mapping,
            "samples": len(cands), "inconsistent_pages": mismatch,
            "file": pdf.name}


if __name__ == "__main__":
    result = {}
    for pdf in sorted(DATA.glob("*.pdf")):
        info = build(pdf)
        result[pdf.name] = info
        print(f"[{pdf.name}] 偏移={info['offset']} 样页={info.get('samples')} "
              f"不一致页={info.get('inconsistent_pages')}")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 写出 {OUT}")
    # 抽查 17623
    pm = result.get("GB_T_17623-2026.pdf", {})
    for phys in ["7", "15", "16", "23", "29", "30"]:
        print(f"  17623 物理页{phys} → 印刷页 {pm.get('map', {}).get(phys)}")
