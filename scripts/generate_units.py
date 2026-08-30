# -*- coding: utf-8 -*-
"""
严格版「增强原子内容单元」生成器
================================
按《数据清洗方案》重做清洗+分块，产出对齐讲师分块效果的 clean_units.json。

内容源：data/mineru_out/GB_T_44653_clean.md（保持已清洗内容不变）
元数据对齐：content_list.json（type/img_path/bbox/page_idx）+ chunks.json（page 兜底）

产出：data/mineru_out/clean_units.json（覆盖）
"""
import json
import re
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "mineru_out")
CLEAN_MD = os.path.join(DATA, "GB_T_44653_clean.md")
CONTENT_LIST = os.path.join(DATA, "content_list.json")
CHUNKS = os.path.join(DATA, "chunks.json")
OUT = os.path.join(DATA, "clean_units.json")

SOURCE = "GB/T 44653-2024 六氟化硫气体现场循环再利用导则"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
IMG_RE = re.compile(r"^!\[.*?\]\((.+?)\)\s*$")
TABLE_TITLE_RE = re.compile(r"^表\s*(\d+)\s+(.+)$")
STD_RE = re.compile(r"(?:GB/T|DL/T|T/CEC|TSG)\s?\d[\d.\-—–]*")
NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)")


# ---------------------------------------------------------------- 读取
def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------- 元数据对齐
def build_content_list_index(cl):
    """从 content_list 提取 table/image 元素的元数据。"""
    idx = {"table": None, "image": None}
    for el in cl:
        t = el.get("type")
        if t == "table" and idx["table"] is None:
            idx["table"] = {
                "img_path": el.get("img_path"),
                "bbox": el.get("bbox"),
                "page": el.get("page_idx", 0) + 1,
            }
        elif t == "image" and idx["image"] is None:
            idx["image"] = {
                "img_path": el.get("img_path"),
                "bbox": el.get("bbox"),
                "page": el.get("page_idx", 0) + 1,
            }
    return idx


def extract_number(s):
    m = NUM_RE.match(s.strip())
    return m.group(1) if m else ""


def build_page_map(chunks):
    """从 chunks.json 构建「叶子标题 → page」映射。"""
    m = {}
    for ch in chunks:
        content = ch.get("content", "")
        page = ch.get("page")
        if not page:
            continue
        hm = re.search(r"^#{1,6}\s+(.+)$", content, re.M)
        if hm:
            leaf = hm.group(1).strip()
        else:
            s = content.strip()
            if s.startswith("下列术语和定义"):
                leaf = "3 术语和定义"
            elif s.startswith("表1") or s.startswith("表 1"):
                leaf = "表1 关键质量指标技术要求"
            elif s.startswith("[1]"):
                leaf = "参考文献"
            elif s.startswith("现场不具备处理条件"):
                leaf = "9.3.3.2 回收处理基地净化处理"
            else:
                leaf = ch.get("section", "")
        if leaf:
            m.setdefault(leaf, page)
    return m


def lookup_page(leaf, page_map):
    if leaf in page_map:
        return page_map[leaf]
    num = extract_number(leaf)
    if num:
        parts = num.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            for key, pg in page_map.items():
                if extract_number(key) == prefix:
                    return pg
    return None


# ---------------------------------------------------------------- 语义分类
def classify_semantic(section, utype):
    if utype in ("table", "equation"):
        return "参数查询"
    if utype == "image":
        return "流程描述"
    leaf = section.split("/")[-1].strip()
    # 要求/限值/指标类 → 参数查询（含数值限值，如回收率≥96%、净化率≥98%）
    if any(k in leaf for k in ("要求", "限值", "指标", "技术参数")):
        return "参数查询"
    if "术语" in section or bool(re.search(r"(^|\s)3\.\d", section)):
        return "术语定义"
    if "安全防护" in section or leaf.startswith("11"):
        return "安全规范"
    if "规范性引用" in section or leaf.startswith("2"):
        return "其他"
    if leaf.startswith("1") and "范围" in leaf:
        return "其他"
    if "参考文献" in section:
        return "其他"
    if "循环再利用原则" in section:
        return "其他"
    if any(k in leaf for k in ("步骤", "方法", "方式", "流程", "分类", "处理")):
        return "流程描述"
    return "参数查询"


# ---------------------------------------------------------------- 长度估算
def calc_tokens(s):
    total = 0.0
    for ch in s:
        o = ord(ch)
        if o >= 0x2E80:  # CJK 及全角字符（含中文标点/≥≤等全角符号）
            total += 1.0
        elif ch.strip() == "":
            continue
        else:
            total += 0.5
    return int(round(total))


# ---------------------------------------------------------------- 关键词 / 问题
SUBSTANCE_MAP = [
    ("六氟化硫", "SF6"),
    ("四氟化碳", "CF4"),
    ("二氧化硫", "SO2"),
    ("硫化氢", "H2S"),
    ("矿物油", "矿物油"),
    ("可水解氟化物", "可水解氟化物"),
    ("空气", "空气"),
    ("湿度", "湿度"),
    ("酸度", "酸度"),
    ("分解产物", "分解产物"),
]


def row_substance_param(name):
    """从表格指标名提取 (物质, 参数)。"""
    substance = ""
    param = ""
    if "SF6" in name or "六氟化硫" in name:
        substance = "SF6"
    elif "四氟化碳" in name or "CF4" in name:
        substance = "CF4"
    elif "二氧化硫" in name or "SO2" in name:
        substance = "SO2"
    elif "硫化氢" in name or "H2S" in name:
        substance = "H2S"
    elif "空气" in name:
        substance = "空气"
    elif "湿度" in name:
        substance = "湿度"
    elif "酸度" in name:
        substance = "酸度"
    elif "矿物油" in name:
        substance = "矿物油"
    elif "可水解氟化物" in name:
        substance = "可水解氟化物"

    if "纯度" in name:
        param = "纯度"
    elif "含量" in name:
        param = "含量"
    else:
        param = "指标"
    return substance, param


def gen_important_kwd(content, section, stype):
    text = content + " " + section
    kw = []
    for chem in ("SF6", "CF4", "SO2", "H2S", "HF"):
        if chem in text and chem not in kw:
            kw.append(chem)
    for name in ("六氟化硫", "四氟化碳", "二氧化硫", "硫化氢", "矿物油", "可水解氟化物"):
        if name in text and name not in kw:
            kw.append(name)
    for p in ("纯度", "回收率", "净化率", "湿度", "酸度", "空气含量", "分解产物", "质量分数", "体积分数"):
        if p in text and p not in kw:
            kw.append(p)
    for std in STD_RE.findall(text):
        std = std.strip()
        if std and std not in kw:
            kw.append(std)
    for topic in ("现场检测", "现场回收", "净化处理", "回充", "安全防护", "循环再利用", "术语定义"):
        if topic in text and topic not in kw:
            kw.append(topic)
    if len(kw) < 3:
        for seg in re.split(r"[/\s]+", section):
            seg = seg.strip()
            if seg and seg not in kw and len(seg) <= 12:
                kw.append(seg)
            if len(kw) >= 3:
                break
    dedup = []
    for k in kw:
        if k not in dedup:
            dedup.append(k)
    return dedup[:8]


def gen_question_kwd(content, section, stype, imp):
    q = []
    if stype == "参数查询":
        if "回收率" in content:
            q.append("SF6气体回收率怎么计算?")
            q.append("SF6气体回收率要求是多少?")
        elif "净化率" in content:
            q.append("SF6气体净化率怎么计算?")
            q.append("SF6气体净化率要求是多少?")
        elif "纯度" in content:
            q.append("SF6纯度要求是多少?")
            q.append("六氟化硫气体纯度限值是多少?")
        elif "湿度" in content:
            q.append("六氟化硫气体湿度限值是多少?")
        else:
            q.append("该技术指标/参数要求是多少?")
            q.append("相关限值或要求是什么?")
    elif stype == "流程描述":
        if "流程图" in content or "循环再利用" in section:
            q.append("SF6气体现场循环再利用流程是什么?")
            q.append("SF6气体回收、净化、回充流程是什么?")
        elif "步骤" in section:
            q.append("该操作的步骤是什么?")
            q.append("如何执行该操作?")
        else:
            q.append("该处理方法是什么?")
            q.append("如何操作?")
    elif stype == "术语定义":
        m = re.search(r"3\.\d+\s*(六氟化硫[^r]*?)(?:refilling|purification|recovery|recycle)?", content)
        if "回收" in content and "净化" not in content:
            q.append("SF6气体回收是什么意思?")
        elif "净化" in content:
            q.append("SF6气体净化处理是什么意思?")
        elif "回充" in content:
            q.append("SF6气体回充是什么意思?")
        elif "循环再利用" in content:
            q.append("SF6气体循环再利用是什么意思?")
        else:
            q.append("该术语是什么意思?")
    elif stype == "安全规范":
        q.append("SF6气体安全防护要求是什么?")
        q.append("SF6气体使用有哪些安全注意事项?")
    else:
        if "规范性引用" in section:
            q.append("本标准引用了哪些文件?")
            q.append("SF6纯度测量依据哪个标准?")
        elif "范围" in section:
            q.append("本文件的适用范围是什么?")
            q.append("SF6气体循环再利用包括哪些环节?")
        else:
            q.append("该部分内容讲了什么?")
    # 去重保序，限制 2-4
    dedup = []
    for x in q:
        if x not in dedup:
            dedup.append(x)
    return dedup[:4]


# ---------------------------------------------------------------- 表格
def parse_table_rows(rows):
    data = []
    for r in rows:
        r = r.strip()
        if not r.startswith("|"):
            continue
        cells = [c.strip() for c in r.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if all(re.match(r"^:?-{2,}:?$", c) for c in cells if c):
            continue
        data.append(cells)
    # 去掉表头行
    if data and data[0][0] in ("项目名称", "指标", "名称"):
        data = data[1:]
    return data


def parse_cell(cell):
    cell = cell.strip()
    if "/" in cell:
        before, unit = cell.rsplit("/", 1)
    else:
        before, unit = cell, ""
    before = before.strip()
    unit = unit.strip()
    m = re.search(r"\(([^()]*)\)\s*$", before)
    if m:
        kind = m.group(1).strip()
        name = before[: m.start()].strip()
    else:
        kind, name = "", before
    return name, kind, unit


def table_units(caption, rows, tbl_meta):
    """生成 10 个表格单元（摘要 + 9 行）。"""
    caption_clean = re.sub(r"^表\s+(\d)", r"表\1", caption.strip())  # "表 1 ..." -> "表1 ..."
    data = parse_table_rows(rows)
    units = []

    names = []
    for cells in data:
        if len(cells) >= 2:
            nm, _, _ = parse_cell(cells[0])
            names.append(nm)

    summary = (
        f"{caption_clean}：本表规定六氟化硫(SF6)气体的关键质量指标技术要求，"
        f"指标包括 " + "、".join(names) + "。"
    )
    units.append(make_unit(
        summary, "table", caption_clean, tbl_meta["page"],
        image_path=tbl_meta["img_path"], bbox=tbl_meta["bbox"],
        semantic_type="参数查询",
        important_kwd=["六氟化硫", "SF6", "质量指标", "纯度", "空气含量", "湿度"][:8],
        question_kwd=["六氟化硫气体关键质量指标有哪些?", "SF6纯度限值是多少?"],
    ))

    for cells in data:
        if len(cells) < 2:
            continue
        name, kind, unit = parse_cell(cells[0])
        value = cells[1].strip()
        parts = [f"指标名称：{name}", f"含量类型：{kind}", f"单位：{unit}", f"要求：{value}"]
        content = caption_clean + "：" + "；".join(parts) + "。"
        substance, param = row_substance_param(name)
        imp = [f"{substance}{param}要求" if substance else f"{param}要求", "质量指标"]
        if substance and substance != "SF6":
            imp.append("六氟化硫")
        elif substance == "SF6":
            imp.append("SF6")
        dedup = []
        for k in imp:
            if k not in dedup:
                dedup.append(k)
        q = []
        if substance and param:
            q.append(f"{substance}{param}要求是多少?")
        q.append("六氟化硫气体关键质量指标限值是多少?")
        units.append(make_unit(
            content, "table", caption_clean, tbl_meta["page"],
            image_path=tbl_meta["img_path"], bbox=tbl_meta["bbox"],
            semantic_type="参数查询",
            important_kwd=dedup[:8],
            question_kwd=q[:4],
        ))
    return units


# ---------------------------------------------------------------- 单元构造
def make_unit(content, utype, section, page, image_path=None, bbox=None,
              semantic_type=None, important_kwd=None, question_kwd=None):
    content = normalize_text(content)
    if not semantic_type:
        semantic_type = classify_semantic(section, utype)
    if important_kwd is None:
        important_kwd = gen_important_kwd(content, section, semantic_type)
    if question_kwd is None:
        question_kwd = gen_question_kwd(content, section, semantic_type, important_kwd)
    meta = {
        "source": SOURCE,
        "page": page,
        "type": utype,
        "section": section,
        "semantic_type": semantic_type,
        "char_len": len(content),
        "token_len": calc_tokens(content),
        "important_kwd": important_kwd,
        "question_kwd": question_kwd,
    }
    if image_path:
        meta["image_path"] = image_path
    if bbox:
        meta["bbox"] = bbox
    return {"content": content, "metadata": meta}


def normalize_text(s):
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ---------------------------------------------------------------- 解析 clean.md
def split_blocks(md_text):
    lines = md_text.split("\n")
    blocks = []
    cur_level = None
    cur_title = None
    cur_body = []
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


def strip_image_refs(body):
    img_path = None
    out = []
    for ln in body:
        m = IMG_RE.match(ln.strip())
        if m:
            img_path = m.group(1).strip()
            continue
        out.append(ln)
    return out, img_path


def find_table(body):
    for i, ln in enumerate(body):
        if TABLE_TITLE_RE.match(ln.strip()):
            j = i + 1
            while j < len(body) and body[j].strip() == "":
                j += 1
            rows = []
            while j < len(body) and body[j].strip().startswith("|"):
                rows.append(body[j].strip())
                j += 1
            return i, j, rows, ln.strip()
    return None


def clean_mermaid_body(body):
    """去掉 <details>/<summary> 包裹，保留正文 + mermaid 代码块。"""
    out = []
    for ln in body:
        s = ln.strip()
        if s.startswith("<details>") or s.startswith("</details>"):
            continue
        if s.startswith("<summary>") and s.endswith("</summary>"):
            continue
        out.append(ln)
    return out


def build_units(md_text, cl_index, page_map):
    blocks = split_blocks(md_text)
    stack = []  # [(level, title)]
    units = []

    for level, title, body in blocks:
        # 维护标题栈
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        section = " / ".join(t for _, t in stack)

        body, img_path = strip_image_refs(body)

        # 空父标题：不单独成单元，仅作 section 上下文
        body_clean = [ln for ln in body if ln.strip() != ""]
        if not body_clean:
            continue

        # 图片单元（6.3 流程图）
        if img_path:
            body_clean2 = clean_mermaid_body(body)
            img_meta = cl_index["image"] or {}
            content = normalize_text("\n".join(body_clean2))
            units.append(make_unit(
                content, "image", section, img_meta.get("page") or lookup_page(title, page_map),
                image_path=img_path, bbox=img_meta.get("bbox"),
                semantic_type="流程描述",
            ))
            continue

        # 表格单元
        found = find_table(body)
        if found:
            t_start, t_end, rows, caption = found
            before = body[:t_start]
            after = body[t_end:]
            # 表格前的正文 → text 单元
            if normalize_text("\n".join(before)):
                units.append(make_unit(
                    normalize_text("\n".join(before)), "text", section,
                    lookup_page(title, page_map),
                ))
            tbl_meta = cl_index["table"] or {"img_path": None, "bbox": None, "page": 9}
            units.extend(table_units(caption, rows, tbl_meta))
            # 表格后的正文 → text 单元
            if normalize_text("\n".join(after)):
                units.append(make_unit(
                    normalize_text("\n".join(after)), "text", section,
                    lookup_page(title, page_map),
                ))
            continue

        # 公式单元（8.3 / 9.2）
        is_equation = bool(re.match(r"^8\.3", title)) or bool(re.match(r"^9\.2", title))
        utype = "equation" if is_equation else "text"
        content = normalize_text("\n".join(body))
        if not content:
            continue
        units.append(make_unit(
            content, utype, section, lookup_page(title, page_map),
        ))

    return units


# ---------------------------------------------------------------- 主流程
def main():
    md_text = read_text(CLEAN_MD)
    cl = read_json(CONTENT_LIST)
    chunks = read_json(CHUNKS)

    cl_index = build_content_list_index(cl)
    page_map = build_page_map(chunks)

    units = build_units(md_text, cl_index, page_map)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(units, f, ensure_ascii=False, indent=2)

    # ---- 验证打印 ----
    from collections import Counter
    type_dist = Counter(u["metadata"]["type"] for u in units)
    sem_dist = Counter(u["metadata"]["semantic_type"] for u in units)
    img_count = sum(1 for u in units if u["metadata"].get("image_path"))
    img_residual = sum(1 for u in units if "![](" in u["content"])
    empty_units = sum(1 for u in units if not u["content"].strip())
    char_lens = [u["metadata"]["char_len"] for u in units]
    token_lens = [u["metadata"]["token_len"] for u in units]

    print("=" * 60)
    print("单元总数:", len(units))
    print("type 分布:", dict(type_dist))
    print("semantic_type 分布:", dict(sem_dist))
    print("带 image_path 数量:", img_count)
    print("![]() 残留数量:", img_residual)
    print("空 content 单元数:", empty_units)
    print("char_len 范围:", min(char_lens), "-", max(char_lens), "平均:", round(sum(char_lens) / len(char_lens), 1))
    print("token_len 范围:", min(token_lens), "-", max(token_lens), "平均:", round(sum(token_lens) / len(token_lens), 1))

    print("=" * 60)
    print("抽样 1 — 表格行级块（SF6 纯度行）:")
    sample_table = next((u for u in units if u["metadata"]["type"] == "table"
                         and "指标名称：六氟化硫(SF6)纯度" in u["content"]), None)
    if sample_table:
        print(json.dumps(sample_table, ensure_ascii=False, indent=2))

    print("=" * 60)
    print("抽样 2 — 图片单元（流程图）:")
    sample_img = next((u for u in units if u["metadata"]["type"] == "image"), None)
    if sample_img:
        print(json.dumps(sample_img, ensure_ascii=False, indent=2))

    print("=" * 60)
    print("抽样 3 — 公式单元（8.3 回收率）:")
    sample_eq = next((u for u in units if u["metadata"]["type"] == "equation"
                      and "8.3" in u["metadata"]["section"]), None)
    if sample_eq:
        print(json.dumps(sample_eq, ensure_ascii=False, indent=2))

    print("=" * 60)
    print("输出文件:", OUT)


if __name__ == "__main__":
    main()
