# -*- coding: utf-8 -*-
"""多文档清洗 + 分块：GB/T 12022-2025（工业六氟化硫）+ GB/T 18867-2025（电子工业用气体 SF6）。

复用思路（不修改已提交脚本，本文件为新建）：
- clean_md.py   的清洗逻辑（公式去壳 / 表格 HTML→MD / 标题层级修复 / 符号归一）
- generate_units.py / chunk_units.py 的「通用结构分块」：按章节聚合 + 不跨页 + token 切块
- chunk_units.py 的 Q2Q（LLM 优先，规则兜底，失败不阻塞）

与 44653 的差异：
- 文档无关的通用实现（不硬编码「表1 关键质量指标」「6.3 流程图」等 44653 特化锚点）
- 每块打 document_id + ingest_version（20260826_v1），用于 v8 集合按 document_id 过滤

输入：data/mineru_out_multidoc/<doc>/vlm/<doc>.md + <doc>_content_list.json
输出：data/mineru_out/chunks_multidoc_new.json（12022 + 18867 两个文档的块）

用法:
    D:/an/envs/langchain/python.exe scripts/ingest_multidoc.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from html import unescape

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = r"C:\Users\lizhihao\w1-day5\device-rag-44653"
MULTIDOC_DIR = os.path.join(BASE, "data", "mineru_out_multidoc")
OUT = os.path.join(BASE, "data", "mineru_out", "chunks_multidoc_new.json")
INGEST_VERSION = "20260826_v1"

# 尝试导入 LLM 客户端（缺依赖/无 key 时优雅降级到规则）
sys.path.insert(0, os.path.join(BASE, "scripts"))
try:
    from llm_client import gen_q2q as _llm_gen_q2q
except Exception:
    _llm_gen_q2q = None

# 文档配置（content_list 用于 page 对齐 + 表格页码；dir_name 用于静态图片路径）
DOCS = [
    {
        "document_id": "gb_t_12022_2025",
        "source": "GB/T 12022-2025 工业六氟化硫",
        "dir_name": "GB_T_12022-2025",
        "md": os.path.join(MULTIDOC_DIR, "GB_T_12022-2025", "vlm", "GB_T_12022-2025.md"),
        "content_list": os.path.join(MULTIDOC_DIR, "GB_T_12022-2025", "vlm", "GB_T_12022-2025_content_list.json"),
    },
    {
        "document_id": "gb_t_18867_2025",
        "source": "GB/T 18867-2025 电子工业用气体 六氟化硫",
        "dir_name": "GB_T_18867-2025",
        "md": os.path.join(MULTIDOC_DIR, "GB_T_18867-2025", "vlm", "GB_T_18867-2025.md"),
        "content_list": os.path.join(MULTIDOC_DIR, "GB_T_18867-2025", "vlm", "GB_T_18867-2025_content_list.json"),
    },
]

MAX_CHARS = 800        # 切大块上限（字）
MAX_TOKENS = 380       # content 估算 token 上限（留余量给 kwd 拼接 ≤512）
MIN_CHARS = 100        # 合并碎片下限

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_TITLE_RE = re.compile(r"^表\s*(\d+)\s+(.+)$")


# ---------------------------------------------------------------------------
# 清洗：公式去壳（复制自 clean_md.py，未改动）
# ---------------------------------------------------------------------------

def clean_formula(s: str) -> str:
    B1 = chr(92)          # 单反斜杠
    B2 = B1 * 2           # 正则匹配 1 个反斜杠 / replace 单反斜杠

    def repl(m):
        t = m.group(1)
        t = t.replace(B2, B1)
        t = re.sub(B2 + r"(?:text|mathrm|mathbf|mbox)" + B1 + r"s*" + B1 + r"{([^{}]*)" + B1 + r"}", B1 + r"1", t)
        t = t.replace(B1 + r"left", "").replace(B1 + r"right", "").replace(B1 + r",", "").replace(B1 + r":", "")
        def frac_repl(m):
            a = re.sub(B1 + r"{([^{}]*)" + B1 + r"}", B1 + r"1", m.group(1)[1:-1])
            b = re.sub(B1 + r"{([^{}]*)" + B1 + r"}", B1 + r"1", m.group(2)[1:-1])
            return "(" + a + ")/(" + b + ")"
        frac_pat = (B2 + r"frac" + B1 + r"s*(" + B1 + r"{(?:[^{}]|" + B1 + r"{[^{}]*" + B1 + r"})*" + B1 + r"})"
                    + B1 + r"s*(" + B1 + r"{(?:[^{}]|" + B1 + r"{[^{}]*" + B1 + r"})*" + B1 + r"})")
        t = re.sub(frac_pat, frac_repl, t)
        t = t.replace(B1 + r"geqslant", "≥").replace(B1 + r"leqslant", "≤")
        t = t.replace(B1 + r"times", "×").replace(B1 + r"pm", "±")
        t = t.replace(B1 + r"cdot", "·").replace(B1 + r"%", "%")
        t = t.replace(B1 + r"quad", " ").replace(B1 + r"qquad", " ")
        t = t.replace(B1 + r"dots", "...").replace(B1 + r"ldots", "...")
        t = t.replace(B1 + r"sim", "~").replace(B1 + r"~", "~")
        t = t.replace(B1 + r"mu", "μ").replace(B1 + r"micro", "µ")
        t = re.sub(B2 + r"tag" + B1 + r"s*" + B1 + r"{?([^{}]*)}?", B1 + r"1", t)
        t = re.sub(B2 + r"mathrm|" + B2 + r"mathbf|" + B2 + r"text|" + B2 + r"mbox|"
                   + B2 + r"left|" + B2 + r"right|" + B2 + r",|" + B2 + r":", "", t)
        t = re.sub(B1 + r"{([^}]*)" + B1 + r"}", B1 + r"1", t)
        t = re.sub(B1 + r"s+", "", t)
        t = t.replace("_", "")
        return t
    s = re.sub(B1 + r"$" + B1 + r"$(.*?)" + B1 + r"$" + B1 + r"$", lambda m: repl(m), s, flags=re.S)
    return re.sub(B1 + r"$([^$]+)" + B1 + r"$", repl, s)


def fix_heading_level(line: str) -> str:
    """# 6.1 采样 -> ## 6.1 采样（按编号点数定层级，兼容附录 A.x）。"""
    m = re.match(r"^(#{1,6})\s+((?:[A-Z]\.)?\d+(?:\.\d+)*)\s*(.*)$", line)
    if m:
        depth = m.group(2).count(".") + 1
        return "#" * min(depth, 6) + " " + m.group(2) + ((" " + m.group(3)) if m.group(3) else "")
    return line


def _attr_int(attrs: str, name: str) -> int:
    m = re.search(name + r'\s*=\s*["\'](\d+)["\']', attrs)
    return int(m.group(1)) if m else 1


def parse_html_table(html: str) -> list[list[str]]:
    """解析 <table> HTML，展开 rowspan/colspan 成规则矩阵（每单元格一值）。"""
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
    if not trs:
        return []
    raw = []
    for tr in trs:
        cells = []
        for m in re.finditer(r"<td([^>]*)>(.*?)</td>", tr, re.S):
            attrs, inner = m.group(1), m.group(2)
            text = re.sub(r"\s+", " ", clean_formula(unescape(inner))).strip()
            cells.append((text, _attr_int(attrs, "colspan"), _attr_int(attrs, "rowspan")))
        raw.append(cells)
    ncols = max((sum(c[1] for c in row) for row in raw), default=0)
    if ncols == 0:
        return []
    # 展开：occ[(r,c)] -> 值（rowspan 向下填充，colspan 向右填充）
    occ: dict[tuple[int, int], str] = {}
    for r, row in enumerate(raw):
        c = 0
        for text, cs, rs in row:
            while (r, c) in occ:
                c += 1
            for dc in range(cs):
                for dr in range(rs):
                    occ[(r + dr, c + dc)] = text
            c += cs
    return [[occ.get((r, c), "") for c in range(ncols)] for r in range(len(raw))]


def build_table_atomic_block(caption: str, table_body: str, page: int) -> dict | None:
    """从完整 table_body HTML 生成【整表原子块】（对齐 44653 表1，不拆行级碎块）。

    格式对齐 chunk_units.py 的 build_table1：
        {caption}：{首列表头}：{其余列名用 / 连接}
        {行名}：{col1 值} / {col2 值} / ...
    多列值（如 12022 表1 的 Ⅰ型/Ⅱ型）用「 / 」连接保留，避免丢列。
    """
    caption_clean = re.sub(r"^表\s+(\d)", r"表\1", caption.strip())
    matrix = parse_html_table(table_body)
    if not matrix:
        return None

    # 表头行：所有非空单元格都含「表头特征词」（项目/指标/名称/检验/批量/抽样/型）
    header_idx = []
    for r, row in enumerate(matrix):
        nonempty = [c for c in row if c]
        if nonempty and all(any(k in c for k in ("项目", "指标", "名称", "检验", "批量", "抽样", "型")) for c in nonempty):
            header_idx.append(r)
    data_start = max(header_idx) + 1 if header_idx else 0
    data = matrix[data_start:]
    if not data:
        return None

    lines = []
    if header_idx:
        header_row = matrix[header_idx[-1]]
        head0 = header_row[0].strip()
        head_cols = [c.strip() for c in header_row[1:] if c.strip()]
        lines.append(f"{caption_clean}：{head0}：" + "/".join(head_cols) if head_cols
                     else f"{caption_clean}：{head0}")
    else:
        lines.append(caption_clean)

    for row in data:
        name = row[0].strip()
        if not name:
            continue
        vals = [c.strip() for c in row[1:] if c.strip()]
        if vals:
            lines.append(f"{name}：{' / '.join(vals)}")
        else:
            lines.append(name)

    # 符号归一：去掉「六氟化硫( SF6 )」空格，对齐 44653 表1 的「六氟化硫(SF6)」格式
    content = normalize_symbols("\n".join(lines))
    return {"content": content, "type": "table", "section": caption_clean,
            "page": page, "leaf": None}


# 图语义块流程化描述（基于标引序号 + mermaid/正文流向人工整理，去裸数字链）。
# 图4 毒性试验装置对齐任务验收口径。
FIGURE_DESCRIPTIONS = {
    "图1 矿物油吸收装置": "SF6气体依次经转子流量计、吸收瓶（冰水浴冷却）、湿式气体流量计，尾气经尾气处理装置排出",
    "图2 酸度吸收装置": "SF6气体依次经缓冲瓶、吸收瓶、湿式气体流量计，尾气经尾气处理装置排出",
    "图3 可水解氟化物测定取样装置": "SF6气体进入取样瓶（抽真空，经真空三通活塞），用U形管压力计测压",
    "图4 毒性试验装置": "六氟化硫气体经调压阀→流量计→缓冲瓶→染毒缸（内设饮水、食物、小白鼠），氧气经调压阀→流量计混入，尾气经尾气处理装置排出",
}


def extract_legend(md_text: str, img_path: str) -> dict[str, str]:
    """从 md 提取某张图的「标引序号说明」（序号 -> 部件名）。

    md 里图片引用 + <details> 之后紧跟「标引序号说明：N——部件；...」，截到「图 N XXX」标题为止。
    """
    fname = img_path.split("/")[-1]
    idx = md_text.find(fname)
    if idx < 0:
        return {}
    tail = md_text[idx:idx + 2000]
    lm = re.search(r"标引序号说明", tail)
    if not lm:
        return {}
    seg = tail[lm.end():]
    seg = re.split(r"\n\s*图\s*\d+", seg)[0]
    legend: dict[str, str] = {}
    for m in re.finditer(r"(\d+(?:\s*,\s*\d+)*)\s*[—\-–]+\s*([^\n；;]+)", seg):
        nums = [n.strip() for n in m.group(1).split(",")]
        name = m.group(2).strip().rstrip("。").strip().lstrip("—–- ")
        for n in nums:
            legend[n] = name
    return legend


def build_image_atomic_block(img_path: str, page: int, img_content: str, title: str,
                             legend: dict, full_image_path: str) -> dict:
    """从 content_list 的 image 元素生成【图原子块】（图语义）。

    content = 图标题 + 流程化描述（FIGURE_DESCRIPTIONS，去裸数字链）+ 标引序号展开；
    保留原始 mermaid 供前端渲染流程图，image_path 供前端显示原图。
    """
    # 提取 mermaid（如有），去掉围栏
    mermaid = ""
    m = re.search(r"```mermaid\s*(.*?)```", img_content, re.S)
    if m:
        mermaid = m.group(1).strip()

    desc = FIGURE_DESCRIPTIONS.get(title, "")
    # 标引序号展开：1—部件；2—部件...
    legend_parts = []
    for n in sorted(legend, key=lambda x: (int(x) if x.isdigit() else 0)):
        name = legend[n].strip().lstrip("—–- ").strip()
        legend_parts.append(f"{n}—{name}")
    legend_str = "；".join(legend_parts)

    if desc and legend_str:
        content = f"{title}：{desc}。（标引序号：{legend_str}。）"
    elif desc:
        content = f"{title}：{desc}。"
    elif legend_str:
        content = f"{title}：（标引序号：{legend_str}。）"
    else:
        # 兜底：mermaid 转可读文本
        nodes = re.findall(r'\["([^"]+)"\]', img_content)
        text = " → ".join(nodes) if nodes else re.sub(r"\n+", "；", img_content.strip())
        content = f"{title}：{text}" if text else title

    return {"content": content, "type": "image", "section": title, "page": page, "leaf": None,
            "image_path": full_image_path, "mermaid": mermaid}


def normalize_symbols(md: str) -> str:
    """符号归一：统一中文全称(化学式)，两路检索都命中。"""
    for chem in ("六氟化硫", "二氧化硫", "四氟化碳", "硫化氢"):
        pass
    md = re.sub("六氟化硫\s*\(\s*SF6\s*\)", "六氟化硫(SF6)", md)
    md = re.sub("二氧化硫\s*\(\s*SO2\s*\)", "二氧化硫(SO2)", md)
    md = re.sub("四氟化碳\s*\(\s*CF4\s*\)", "四氟化碳(CF4)", md)
    md = re.sub("硫化氢\s*\(\s*H2S\s*\)", "硫化氢(H2S)", md)
    md = re.sub("(?<!六氟化硫\()SF6", "六氟化硫(SF6)", md)
    md = re.sub("(?<!二氧化硫\()SO2", "二氧化硫(SO2)", md)
    md = re.sub("(?<!四氟化碳\()CF4", "四氟化碳(CF4)", md)
    md = re.sub("(?<!硫化氢\()H2S", "硫化氢(H2S)", md)
    return md


# ---------------------------------------------------------------------------
# 分块：通用结构分块（对齐 chunk_units.py 思路）
# ---------------------------------------------------------------------------

def calc_tokens(s: str) -> int:
    """token 估算（CJK≈1 / ASCII≈0.5），加载失败退回估算。"""
    total = 0.0
    for ch in s:
        o = ord(ch)
        if o >= 0x2E80:
            total += 1.0
        elif ch.strip() == "":
            continue
        else:
            total += 0.5
    return int(round(total))


def split_blocks(md_text: str):
    """按标题切块，返回 [(level, title, body_lines)]（正文顺序保留）。"""
    lines = md_text.split("\n")
    blocks = []
    cur_level, cur_title, cur_body = None, None, []
    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            if cur_title is not None:
                blocks.append((cur_level, cur_title, cur_body))
            cur_level = len(m.group(1))
            cur_title = m.group(2).strip()
            cur_body = []
        else:
            cur_body.append(line)
    if cur_title is not None:
        blocks.append((cur_level, cur_title, cur_body))
    return blocks


def top_section(section: str) -> str:
    return section.split(" / ")[0].strip()


def leaf_title(section: str, prefix: str) -> str | None:
    leaf = section.split(" / ")[-1].strip()
    if not leaf or leaf == prefix:
        return None
    return leaf


def build_default(prefix: str, units: list[dict]) -> str:
    """普通文本合并：主标题 + 各单元（子标题 + 内容）。"""
    if len(units) == 1:
        return f"{prefix}：{units[0]['content'].strip()}"
    parts = []
    last_leaf = None
    for u in units:
        leaf = u.get("leaf")
        txt = u["content"].strip()
        if leaf and leaf != last_leaf:
            parts.append(f"{leaf}：{txt}")
            last_leaf = leaf
        else:
            parts.append(txt)
    return f"{prefix}：{chr(10).join(parts)}"


# ---------------------------------------------------------------------------
# 语义分类（规则为主，LLM 兜底）
# ---------------------------------------------------------------------------

def rule_semantic(top: str, unit_types: set, content: str) -> str | None:
    if "安全" in top:
        return "安全要求"
    if "术语" in top:
        return "术语定义"
    if "image" in unit_types:
        return "图语义"
    if "table" in unit_types or "equation" in unit_types:
        return "参数查询"
    if any(k in top for k in ("规范性引用", "范围", "参考文献", "原则", "前言", "附录")):
        return "概述"
    if any(k in top for k in ("技术要求", "要求", "指标", "限值", "采样", "检验规则")):
        return "参数查询"
    if any(k in top for k in ("试验方法", "测定", "标志", "包装", "运输", "贮存")):
        return "标准要求"
    if re.search(r"[≥≤]\s*\d", content):
        return "参数查询"
    if any(k in content for k in ("步骤", "方式", "方法", "过程", "分类")):
        return "流程步骤"
    return None


def decide_semantic(top: str, unit_types: set, content: str) -> str:
    s = rule_semantic(top, unit_types, content)
    if s:
        return s
    return "概述"


# ---------------------------------------------------------------------------
# Q2Q：LLM 优先，规则兜底（对齐 chunk_units.py）
# ---------------------------------------------------------------------------

SUBSTANCE = ("六氟化硫", "SF6", "四氟化碳", "CF4", "二氧化硫", "SO2", "硫化氢", "H2S",
             "矿物油", "可水解氟化物", "空气", "湿度", "酸度", "分解产物",
             "一氧化碳", "二氧化碳", "甲烷", "六氟乙烷", "八氟丙烷", "氢", "氮", "氧")
PARAM = ("回收率", "净化率", "纯度", "湿度", "酸度", "空气含量", "分解产物",
         "质量分数", "体积分数", "含量", "技术要求", "极限真空度")
TOPIC = ("现场检测", "现场回收", "净化处理", "回充", "安全防护", "循环再利用",
         "术语和定义", "规范性引用", "采样", "试验方法", "检验规则", "尾气处理")
STD_RE = re.compile(r"(?:GB/T|GB |DL/T|T/CEC|TSG)\s?\d[\d.、—–-]*")


def fallback_important_kwd(content: str, section: str) -> list[str]:
    kws = []
    for std in STD_RE.findall(content):
        std = std.strip()
        if std and std not in kws:
            kws.append(std)
    for w in SUBSTANCE:
        if w in content and w not in kws:
            kws.append(w)
    for w in PARAM:
        if w in content and w not in kws:
            kws.append(w)
    for w in TOPIC:
        if w in content and w not in kws:
            kws.append(w)
    if len(kws) < 3:
        leaf = section.split(" / ")[-1].strip()
        if leaf and leaf not in kws:
            kws.append(leaf)
    return kws[:3]


def fallback_question_kwd(content: str, section: str, semantic: str) -> list[str]:
    q = []
    if semantic == "参数查询":
        found = False
        for p in ("纯度", "含量", "回收率", "净化率", "湿度", "要求"):
            if p in content:
                q.append(f"六氟化硫{p}要求是多少?")
                found = True
                break
        if not found:
            q.append("该技术指标/参数要求是多少?")
        q.append("相关限值或要求是什么?")
    elif semantic == "流程步骤":
        q.append("该操作的步骤是什么?")
        q.append("如何执行该操作?")
    elif semantic == "标准要求":
        q.append("该技术指标/参数要求是多少?")
        q.append("相关测定/试验方法是什么?")
    elif semantic == "术语定义":
        q.append("该术语的定义是什么?")
        q.append("相关术语是什么意思?")
    elif semantic == "图语义":
        q.append("该流程是什么?")
    elif semantic == "安全要求":
        q.append("安全防护/安全信息要求是什么?")
    else:
        if "规范性引用" in section:
            q.append("本标准引用了哪些文件?")
        elif "范围" in section:
            q.append("本文件的适用范围是什么?")
        elif "参考文献" in section:
            q.append("本标准引用了哪些参考文献?")
        else:
            q.append("该部分内容讲了什么?")
    for f in ("该部分内容讲了什么?", "相关限值或要求是什么?"):
        if f not in q:
            q.append(f)
    return q[:3]


def make_q2q(content: str, section: str, semantic: str):
    if _llm_gen_q2q is not None:
        r = _llm_gen_q2q(content, section)
        if r and r.get("important_kwd") and r.get("question_kwd"):
            return r["important_kwd"][:3], r["question_kwd"][:3]
    return fallback_important_kwd(content, section), fallback_question_kwd(content, section, semantic)


# ---------------------------------------------------------------------------
# 主流程：单文档 → 块列表
# ---------------------------------------------------------------------------

def load_content_list(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_text_elems(cl: list[dict]) -> list[tuple[str, int]]:
    """content_list 里 text 元素（已清洗公式）→ [(text, page)]，用于 page 对齐。"""
    return [(clean_formula(e.get("text", "")).strip(), e.get("page_idx", 0) + 1)
            for e in cl if e.get("type") == "text" and e.get("text", "").strip()]


def build_table_page_map(cl: list[dict]) -> dict[str, int]:
    """表格 caption → page（按 caption 首段匹配，如「表1 技术要求」→ 6）。"""
    m = {}
    for e in cl:
        if e.get("type") != "table":
            continue
        caps = e.get("table_caption", [])
        if caps:
            m[re.sub(r"^表\s+(\d)", r"表\1", caps[0].strip())] = e.get("page_idx", 0) + 1
    return m


def norm(s: str) -> str:
    """page 对齐时的反向符号归一 + 去噪（对齐 content_list 原始文本）。"""
    s = s.replace("六氟化硫(SF6)", "SF6").replace("二氧化硫(SO2)", "SO2")
    s = s.replace("四氟化碳(CF4)", "CF4").replace("硫化氢(H2S)", "H2S")
    return re.sub(r"[#|—–|::\s]", "", s)


def align_page(probe: str, text_elems: list[tuple[str, int]], cursor: int) -> tuple[int | None, int]:
    """双指针：在 text_elems 中找 probe 首次出现位置，返回 (page, 新 cursor)。"""
    probe_n = norm(probe)
    if not probe_n:
        return (text_elems[cursor][1] if cursor < len(text_elems) else None), cursor
    for i in range(cursor, len(text_elems)):
        t, pg = text_elems[i]
        tn = norm(t)
        if probe_n[:12] in tn or tn.startswith(probe_n[:8]):
            return pg, i
    return (text_elems[cursor][1] if cursor < len(text_elems) else None), cursor


def split_by_page_and_size(units: list[dict], prefix: str) -> list[list[dict]]:
    """组内先按 page 拆（不跨页），再按估算 token 上限贪心拆。"""
    by_page, cur, cur_page = [], [], None
    for u in units:
        if cur_page is not None and u["page"] != cur_page:
            by_page.append(cur)
            cur = []
        cur.append(u)
        cur_page = u["page"]
    if cur:
        by_page.append(cur)

    result = []
    for sub in by_page:
        cur2, cur_tok = [], 0
        for u in sub:
            leaf = u.get("leaf")
            piece = f"{leaf}：{u['content']}" if leaf else u["content"]
            t = calc_tokens(piece)
            if cur2 and cur_tok + t > MAX_TOKENS:
                result.append(cur2)
                cur2, cur_tok = [u], t
            else:
                cur2.append(u)
                cur_tok += t
        if cur2:
            result.append(cur2)
    return result


def _sort_key(chunk: dict):
    # 多文档按 page 自然顺序（正文/表格/附录/参考文献都按页码排）
    return (chunk["page"] if chunk["page"] else 0, chunk["section"])


def build_doc_chunks(doc: dict) -> list[dict]:
    md_text = open(doc["md"], encoding="utf-8").read()
    cl = load_content_list(doc["content_list"])
    text_elems = build_text_elems(cl)
    doc_id = doc["document_id"]
    dir_name = doc["dir_name"]
    # 表格元素：从 content_list 直接取（含完整 table_body HTML + page_idx）
    tables_cl = []
    for e in cl:
        if e.get("type") != "table":
            continue
        caps = e.get("table_caption", [])
        caption = re.sub(r"^表\s+(\d)", r"表\1", caps[0].strip()) if caps else ""
        tables_cl.append((caption, e.get("table_body", ""), e.get("page_idx", 0) + 1))
    # 图片元素：img_path -> (page, content)；完整相对路径供前端显示原图
    images_cl = []
    for e in cl:
        if e.get("type") != "image":
            continue
        rel = e.get("img_path", "")
        images_cl.append((rel, e.get("page_idx", 0) + 1, e.get("content", "") or ""))

    # 1) 删封面/前言：正文从「# 1 ...」开始（对齐 44653 做法，跳过封面与前言）
    body_start = re.search(r"^#\s+1\s", md_text, re.M)
    if not body_start:
        body_start = re.search(r"^#\s+前言", md_text, re.M)
    if body_start:
        md_text = md_text[body_start.start():]

    # 1.5) 提取图片标题 + 标引序号（此时 md 还含图片引用与标引序号说明，剥离前提取）
    img_titles: dict[str, str] = {}
    img_legends: dict[str, dict] = {}
    for m in re.finditer(r"!\[[^\]]*\]\((images/[^)]+)\)", md_text):
        img_path = m.group(1)
        tail = md_text[m.end():m.end() + 500]
        tm = re.search(r"图\s*(\d+)\s*([^\n|]{1,30})", tail)
        if tm:
            img_titles[img_path] = f"图{tm.group(1)} {tm.group(2).strip()}"
        img_legends[img_path] = extract_legend(md_text, img_path)

    # 2) 剥离图片引用 + <details> 包裹（图片已由 content_list 生成原子块）
    md_text = re.sub(r"!\[[^\]]*\]\(images/[^)]+\)\s*", "", md_text)
    md_text = re.sub(r"<details>.*?</details>", "", md_text, flags=re.S)
    # 表格 HTML 清空（表格改由 content_list 的 table_body 生成，避免丢失 rowspan/colspan）
    md_text = re.sub(r"<table>.*?</table>", "", md_text, flags=re.S)

    # 3) 公式去壳 + 符号归一 + 标题层级修复
    md_text = clean_formula(md_text)
    md_text = normalize_symbols(md_text)
    lines = [fix_heading_level(l) if l.startswith("#") else l for l in md_text.split("\n")]
    md_text = "\n".join(lines)
    md_text = re.sub(r"\n{3,}", "\n\n", md_text)
    md_text = re.sub(r"[ \t]+$", "", md_text, flags=re.M)

    # 4) 按标题切块
    blocks = split_blocks(md_text)

    # 5) 构建「叶子单元」（表格/公式/文本），维护标题栈
    stack: list[tuple[int, str]] = []
    units = []
    for level, title, body in blocks:
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        section = " / ".join(t for _, t in stack)

        body_clean = [ln for ln in body if ln.strip() != ""]
        # 标题锚点：有正文才成单元
        if not body_clean:
            continue

        # 文本单元（公式已由 clean_formula 转为文本并融入正文，不再单独标 equation）
        content = "\n".join(body_clean).strip()
        if not content:
            continue
        units.append({"content": content, "type": "text", "section": section,
                      "page": None, "leaf": leaf_title(section, top_section(section)),
                      "_section_path": section})

    # 5.5) 表格原子块：从 content_list 的 table_body 生成整表（不再拆行级碎块）
    for caption, body, page in tables_cl:
        u = build_table_atomic_block(caption, body, page)
        if u:
            units.append(u)
    # 5.6) 图片原子块：从 content_list 的 image 生成（图语义 + 标引序号 + mermaid + 原图路径）
    for img_path, page, content in images_cl:
        title = img_titles.get(img_path, "")
        legend = img_legends.get(img_path, {})
        full_image_path = f"/images/{dir_name}/vlm/{img_path}"
        units.append(build_image_atomic_block(img_path, page, content, title, legend, full_image_path))

    # 6) page 对齐：文本单元用双指针，表格/图片单元已有 page
    cursor = 0
    for u in units:
        if u["type"] in ("table", "image"):
            continue
        probe = u["content"][:16]
        pg, cursor = align_page(probe, text_elems, cursor)
        u["page"] = pg

    # 7) 按顶级章节聚合（表格/图片独立成原子块，文本聚合 + 不跨页 + token 切块）
    atomic = [u for u in units if u["type"] in ("table", "image")]
    normal = [u for u in units if u["type"] not in ("table", "image")]

    chunks = []
    for u in atomic:
        top = top_section(u["section"])
        sem = "图语义" if u["type"] == "image" else "参数查询"
        ik, qk = make_q2q(u["content"], top, sem)
        chunks.append(_make_chunk(doc, u["content"], u["type"], u["section"],
                                  u["page"], sem, ik, qk,
                                  image_path=u.get("image_path"),
                                  mermaid=u.get("mermaid")))

    groups = []
    for u in normal:
        ts = top_section(u["section"])
        if groups and groups[-1]["top"] == ts:
            groups[-1]["units"].append(u)
        else:
            groups.append({"top": ts, "units": [u]})

    for g in groups:
        top = g["top"]
        for sub in split_by_page_and_size(g["units"], top):
            content = build_default(top, sub)
            types = {u["type"] for u in sub}
            semantic = decide_semantic(top, types, content)
            typ = "equation" if "equation" in types else "text"
            page = min((u["page"] for u in sub if u["page"] is not None), default=None)
            ik, qk = make_q2q(content, top, semantic)
            chunks.append(_make_chunk(doc, content, typ, top, page, semantic, ik, qk))

    chunks.sort(key=_sort_key)
    return chunks


def _make_chunk(doc, content, typ, section, page, semantic, ik, qk,
                image_path=None, mermaid=None):
    chunk = {
        "content": content,
        "source": doc["source"],
        "page": page,
        "type": typ,
        "section": section,
        "semantic_type": semantic,
        "char_len": len(content),
        "token_len": calc_tokens(content),
        "important_kwd": ik,
        "question_kwd": qk,
        "document_id": doc["document_id"],
        "ingest_version": INGEST_VERSION,
    }
    if image_path:
        chunk["image_path"] = image_path
    if mermaid:
        chunk["mermaid"] = mermaid
    return chunk


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    all_chunks = []
    for doc in DOCS:
        if not os.path.exists(doc["md"]):
            print(f"⚠️ 跳过（md 不存在）: {doc['document_id']}")
            continue
        chunks = build_doc_chunks(doc)
        print(f"✅ {doc['document_id']}: {len(chunks)} 块")
        for c in chunks:
            print(f"   p{c['page'] if c['page'] else '?'} {c['type']:<9} {c['semantic_type']:<4} "
                  f"{c['char_len']:>4}字 {c['section'][:28]}")
        all_chunks.extend(chunks)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    from collections import Counter
    print(f"\n=== 总计 {len(all_chunks)} 块 → {OUT}")
    print("document_id 分布:", dict(Counter(c['document_id'] for c in all_chunks)))
    print("semantic_type 分布:", dict(Counter(c['semantic_type'] for c in all_chunks)))


if __name__ == "__main__":
    main()
