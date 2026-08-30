# -*- coding: utf-8 -*-
"""Markdown 清洗：删目录 / 公式去壳 / 表格转Markdown / 标题层级修复 / 空行压缩
输入: data/mineru_out/*.md + content_list.json
输出: data/mineru_out/GB_T_44653_clean.md + chunks.json
"""
import os, re, json

OUT = r"C:\Users\lizhihao\w1-day5\device-rag-44653\data\mineru_out"
md_file = [f for f in os.listdir(OUT) if f.endswith(".md") and "clean" not in f][0]
md = open(os.path.join(OUT, md_file), encoding="utf-8").read()
cl = json.load(open(os.path.join(OUT, "content_list.json"), encoding="utf-8"))

def clean_formula(s: str) -> str:
    """去 $ 壳 + LaTeX 转文本（先统一为单反斜杠）"""
    B1 = chr(92)          # 单反斜杠
    B2 = B1 * 2           # 正则匹配 1 个反斜杠 / replace 单反斜杠
    def repl(m):
        t = m.group(1)
        t = t.replace(B2, B1)   # 双反斜杠(正文公式) -> 单反斜杠(表格公式)
        # 0) 带参命令: \text{X} -> X
        t = re.sub(B2 + r"(?:text|mathrm|mathbf|mbox)" + B1 + r"s*" + B1 + r"{([^{}]*)" + B1 + r"}", B1 + r"1", t)
        t = t.replace(B1 + r"left", "").replace(B1 + r"right", "").replace(B1 + r",", "").replace(B1 + r":", "")
        # 分式: \frac{A}{B} -> (A)/(B)
        def frac_repl(m):
            a = re.sub(B1 + r"{([^{}]*)" + B1 + r"}", B1 + r"1", m.group(1)[1:-1])
            b = re.sub(B1 + r"{([^{}]*)" + B1 + r"}", B1 + r"1", m.group(2)[1:-1])
            return "(" + a + ")/(" + b + ")"
        frac_pat = (B2 + r"frac" + B1 + r"s*(" + B1 + r"{(?:[^{}]|" + B1 + r"{[^{}]*" + B1 + r"})*" + B1 + r"})"
                    + B1 + r"s*(" + B1 + r"{(?:[^{}]|" + B1 + r"{[^{}]*" + B1 + r"})*" + B1 + r"})")
        t = re.sub(frac_pat, frac_repl, t)
        # 符号
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
    """# 6.1 分类 -> ## 6.1 分类（按编号点数定层级）"""
    m = re.match(r"^(#{1,6})\s+(\d+(?:\.\d+)*)\s*(.*)$", line)
    if m:
        depth = m.group(2).count(".") + 1
        return "#" * min(depth, 6) + " " + m.group(2) + ((" " + m.group(3)) if m.group(3) else "")
    return line

def html_table_to_md(html_str: str):
    """<table><tr><td>... -> Markdown 表格 + 行级语义块"""
    rows = re.findall(r"<tr>(.*?)</tr>", html_str, re.S)
    parsed = []
    for r in rows:
        cells = [clean_formula(c) for c in re.findall(r"<td>(.*?)</td>", r, re.S)]
        parsed.append(cells)
    if not parsed:
        return "", []
    md_tbl = ["| " + " | ".join(parsed[0]) + " |", "| " + " | ".join(["---"] * len(parsed[0])) + " |"]
    for row in parsed[1:]:
        md_tbl.append("| " + " | ".join(row) + " |")
    row_blocks = []
    if len(parsed[0]) == 2 and len(parsed) > 1:
        caption = "表1 关键质量指标技术要求"
        # 讲师表格清洗 ③④：解析 "名称(类型)/ 单位" → 指标名称/含量类型/单位
        pat1 = '^(.*)\\((.*?)\\)\\s*/\\s*(.*?)$'
        pat2 = '^(.*)\\((.*?)\\)$'
        def parse_indicator(cell):
            m = re.match(pat1, cell)
            if m:
                return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            m2 = re.match(pat2, cell)
            if m2:
                return m2.group(1).strip(), m2.group(2).strip(), ""
            return cell.strip(), "", ""
        items = []
        for row in parsed[1:]:
            name, typ, unit = parse_indicator(row[0])
            items.append((name, typ, unit, row[1].strip()))
        # ⑤ 表格摘要（讲师：规则生成）
        names = "、".join(i[0] for i in items)
        row_blocks.append(f"{caption}：本表规定六氟化硫(SF6)气体的关键质量指标技术要求，指标包括 {names}。")
        # ③④ 每行语义补全：指标名称 + 含量类型 + 单位 + 要求
        for name, typ, unit, val in items:
            parts = [f"指标名称：{name}"]
            if typ:
                parts.append(f"含量类型：{typ}")
            if unit:
                parts.append(f"单位：{unit}")
            parts.append(f"要求：{val}")
            row_blocks.append(f"{caption}：{'；'.join(parts)}。")
    return "\n".join(md_tbl), row_blocks

# ========== 1) 删目录段（目 次 → 正文第一个 # 标题） ==========
body_start = re.search(r"^#\s+1\s+\u8303\u56f4", md, re.M)
if body_start:
    md = md[body_start.start():]

# ========== 2) 表格 HTML → Markdown（先于公式去壳，避免污染） ==========
row_blocks = []
def table_repl(m):
    global row_blocks
    md_tbl, blocks = html_table_to_md(m.group(1))
    row_blocks = blocks
    return "\n" + md_tbl
md = re.sub(r"<table>(.*?)</table>", table_repl, md, flags=re.S)

# ========== 3) 公式去壳 ==========
md = clean_formula(md)

# ========== 4) 标题层级修复 ==========
lines = md.split("\n")
lines = [fix_heading_level(l) if l.startswith("#") else l for l in lines]

# 4.1) 拆行标题合并: "## 3.3\n# SF6 气体回充 refilling..." -> "## 3.3 SF6 气体回充 refilling..."
merged_lines = []
i = 0
while i < len(lines):
    m = re.match(r"^(#{1,6})\s+(\d+(?:\.\d+)*)\s*$", lines[i])
    if m:
        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j < len(lines) and re.match(r"^#{1,6}\s+[^\d]", lines[j]):
            title = re.match(r"^#{1,6}\s+(.+)$", lines[j]).group(1)
            num = m.group(2)
            depth = num.count(".") + 1
            merged_lines.append("#" * min(depth, 6) + " " + num + " " + title)
            merged_lines.append("")
            i = j + 1
            continue
    merged_lines.append(lines[i])
    i += 1
md = "\n".join(merged_lines)

# ========== 4.5) 符号归一（讲师方案一 §三③：统一 中文全称(化学式)，两路检索都命中） ==========
# a) 清理已有'中文全称( 化学式 )'空格形式
md = re.sub("六氟化硫\s*\(\s*SF6\s*\)", "六氟化硫(SF6)", md)
md = re.sub("二氧化硫\s*\(\s*SO2\s*\)", "二氧化硫(SO2)", md)
md = re.sub("四氟化碳\s*\(\s*CF4\s*\)", "四氟化碳(CF4)", md)
md = re.sub("硫化氢\s*\(\s*H2S\s*\)", "硫化氢(H2S)", md)
# b) 剩余裸化学式（前面不是'中文全称('）→ 中文全称(化学式)
md = re.sub("(?<!六氟化硫\()SF6", "六氟化硫(SF6)", md)
md = re.sub("(?<!二氧化硫\()SO2", "二氧化硫(SO2)", md)
md = re.sub("(?<!四氟化碳\()CF4", "四氟化碳(CF4)", md)
md = re.sub("(?<!硫化氢\()H2S", "硫化氢(H2S)", md)
# c) 表格行级块同步归一
row_blocks = [re.sub("(?<!六氟化硫\()SF6", "六氟化硫(SF6)", rb.replace("( SF6 )", "(SF6)")) for rb in row_blocks]

# ========== 5) 空行压缩 + 行尾空格清理 ==========
md = re.sub(r"\n{3,}", "\n\n", md)
md = re.sub(r"[ \t]+$", "", md, flags=re.M)

clean_path = os.path.join(OUT, "GB_T_44653_clean.md")
open(clean_path, "w", encoding="utf-8").write(md)

# ========== 6) 分块：标题级（H1 段内按 H2 分，超 512 字符按 H3/空行再拆） ==========
def split_chunks(clean_text: str) -> list:
    lines = clean_text.split("\n")
    units, cur, cur_h1 = [], [], ""
    for l in lines:
        if l.startswith("# "):
            if cur: units.append((cur_h1, "\n".join(cur))); cur = []
            cur_h1 = l[2:].strip()
        elif l.startswith("## ") or l.startswith("### "):
            if cur: units.append((cur_h1, "\n".join(cur))); cur = []
            cur.append(l)
        else:
            cur.append(l)
    if cur: units.append((cur_h1, "\n".join(cur)))

    chunks = []
    for h1, body in units:
        body = body.strip()
        if not body: continue
        if len(body) <= 512:
            chunks.append({"section": h1, "content": body})
        else:
            buf = ""
            for para in re.split(r"\n{2,}", body):
                if len(buf) + len(para) > 512 and buf:
                    chunks.append({"section": h1, "content": buf.strip()})
                    buf = para
                else:
                    buf = buf + "\n\n" + para if buf else para
            if buf.strip():
                chunks.append({"section": h1, "content": buf.strip()})
    return chunks

chunks = split_chunks(md)
for rb in row_blocks:
    chunks.append({"section": "表1 关键质量指标技术要求", "content": rb, "page": 9, "locked": True})

# ========== 7) 页码对齐：md 分块与 content_list 双指针 ==========
text_elems = [(clean_formula(e.get("text", "")).strip(), e.get("page_idx", 0) + 1)
              for e in cl if e.get("type") == "text" and e.get("text", "").strip()]
cursor = 0
def norm(s):
    # 反向符号归一（对齐 content_list 原始文本）+ 去噪
    s = s.replace("六氟化硫(SF6)", "SF6").replace("二氧化硫(SO2)", "SO2")
    s = s.replace("四氟化碳(CF4)", "CF4").replace("硫化氢(H2S)", "H2S")
    return re.sub(r"[#|\u2014\u2013|::\s]", "", s)
for c in chunks:
    if c.get("locked"):
        continue
    probe = norm(c["content"])[:12]
    if not probe:
        c["page"] = text_elems[cursor][1] if cursor < len(text_elems) else None
        continue
    found = None
    for i in range(cursor, len(text_elems)):
        t, pg = text_elems[i]
        if probe in norm(t) or t.startswith(probe[:8]):
            found = pg; cursor = i; break
    c["page"] = found or (text_elems[cursor][1] if cursor < len(text_elems) else None)

for c in chunks:
    c.pop("locked", None)
    c.setdefault("source", "GB_T_44653-2024")

json.dump(chunks, open(os.path.join(OUT, "chunks.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print(f"\u2705 \u6e05\u6d17\u5b8c\u6210: {clean_path}")
print(f"   md: {len(md)} chars")
print(f"   chunks: {len(chunks)} \u4e2a")
print(f"   \u8868\u683c\u884c\u7ea7\u5757: {len(row_blocks)} \u4e2a")
pages = sorted(set(c.get("page", 0) for c in chunks))
print(f"   \u9875\u7801\u8986\u76d6: {pages}")
