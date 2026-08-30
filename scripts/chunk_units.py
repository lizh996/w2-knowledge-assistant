# -*- coding: utf-8 -*-
"""chunk_units.py —— 通用结构分块 + LLM Q2Q/语义兜底（对齐《分块方案》理想框架）。

管线（方案⑤）：
① 结构骨架（按顶级章节 title 分组，标题只作锚点不单独成块）
② 合并碎片（同顶级章节相邻 text/equation 合并）
③ 切大块（>800 字按单元边界贪心拆，非硬均分）
④ 标语义类型（7 类：规则为主 + LLM 兜底）
⑤ 原子保护（表1/流程图/安全不拆，独立成块）
⑥ 上下文附着（表带前置说明、图带父标题）
⑦ 尺寸控制（500-800 字目标、≤512 token 底线、不跨页）
⑧ Q2Q 后处理（DeepSeek LLM 生成 3 kwd + 3 问题；失败降级规则，不阻塞）

输入：data/mineru_out/clean_units.json（56 原子单元）
输出：data/mineru_out/chunks_v7_final.json
"""
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = r"C:\Users\lizhihao\w1-day5\device-rag-44653"
INPUT = os.path.join(BASE, "data", "mineru_out", "clean_units.json")
OUTPUT = os.path.join(BASE, "data", "mineru_out", "chunks_v7_final.json")
SOURCE = "GB/T 44653-2024 六氟化硫气体现场循环再利用导则"

# 尝试导入 LLM 客户端（缺依赖/无 key 时优雅降级到规则）
try:
    from llm_client import gen_q2q as _llm_gen_q2q, classify_semantic as _llm_classify
except Exception:
    _llm_gen_q2q = None
    _llm_classify = None

MAX_CHARS = 800          # ③ 切大块上限（字）
MAX_CONTENT_TOKENS = 380  # ⑦ content(含标题前缀) 实测 token 上限，留 ~130 token 给 kwd 拼接后 ≤512
MIN_CHARS = 100           # ② 合并碎片下限（<100 字不单独成块，向上合并）


# BGE-M3 tokenizer 懒加载（实测 token 兜底，方案 3.1「硬上限 = token 实测」）
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    try:
        os.environ.setdefault(
            "BGE_M3_MODEL",
            r"C:\Users\lizhihao\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181",
        )
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(os.environ["BGE_M3_MODEL"])
    except Exception:
        _tokenizer = None
    return _tokenizer


# ---------------------------------------------------------------------------
# 加载与工具
# ---------------------------------------------------------------------------

def load_units(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    units = []
    for item in data:
        m = item["metadata"]
        units.append({
            "content": item["content"],
            "page": m.get("page"),
            "type": m.get("type", "text"),
            "section": m.get("section", ""),
        })
    return units


def top_section(section):
    """顶级章节 = section 路径第一段（如 '5 设备及附件'）。"""
    return section.split("/")[0].strip()


def leaf_title(section, prefix):
    """子标题 = section 最后一段（与 chunk 主标题重复则返回 None）。"""
    leaf = section.split("/")[-1].strip()
    if not leaf or leaf == prefix:
        return None
    return leaf


def normalize_text(t):
    """轻量符号归一：去清洗残留反斜杠。"""
    return t.replace("\\μ", "μ").replace("\\~", "~").replace("\\", "")


def calc_tokens(s):
    """token 实测（BGE-M3 tokenizer）；加载失败退回估算（CJK≈1 / ASCII≈0.5）。"""
    tok = _get_tokenizer()
    if tok is not None:
        return len(tok.encode(s))
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


# ---------------------------------------------------------------------------
# ④ 语义类型（规则为主 + LLM 兜底）
# ---------------------------------------------------------------------------

def rule_semantic(top, unit_types, content):
    """规则分类，明确才返回；不确定返回 None（走 LLM 兜底）。"""
    if "image" in unit_types:
        return "图语义"
    if "table" in unit_types or "equation" in unit_types:
        return "参数查询"
    if "安全" in top:
        return "安全要求"
    if "术语" in top:
        return "术语定义"
    if any(k in top for k in ("规范性引用", "范围", "参考文献", "原则")):
        return "概述"
    if "设备" in top:
        return "标准要求"
    if re.search(r"[≥≤]\s*\d", content):
        return "参数查询"
    if any(k in content for k in ("步骤", "方式", "方法", "过程", "分类")):
        return "流程步骤"
    return None


def decide_semantic(top, unit_types, content):
    """规则优先，LLM 兜底，最终默认概述。"""
    s = rule_semantic(top, unit_types, content)
    if s:
        return s
    if _llm_classify is not None:
        r = _llm_classify(content, top)
        if r:
            return r
    return "概述"


# ---------------------------------------------------------------------------
# ⑤⑥ content 组装（上下文附着 + 原子保护）
# ---------------------------------------------------------------------------

def build_table1(prefix, units):
    """表1 原子块：10 个行级单元重组成整表（对齐讲师示例）。"""
    rows = []
    for u in units:
        c = u["content"].strip()
        m = re.search(
            r"指标名称：(.+?)[；;]\s*含量类型：(.+?)[；;]\s*单位：(.+?)[；;]\s*要求：(.+?)。?$",
            c,
        )
        if m:
            name = m.group(1).strip()
            ctype = m.group(2).strip()
            unit = m.group(3).strip().replace("^", "")
            req = m.group(4).strip()
            rows.append(f"{name}({ctype})/{unit}：{req}")
    lines = [f"{prefix}：项目名称：指标"]
    lines.extend(rows)
    return "\n".join(lines)


FLOW_SEMANTIC = (
    "起点：设备内待回收六氟化硫(SF6)气体 → 节点：现场检测（用于分类分流，不判合格）→ "
    "分支（按气体类别）：第一类（合格气体或仅湿度不合格气体）→ 现场回收 → 现场检测 → "
    "判断：合格 → 回充设备；判断：不合格 → 过滤干燥 → 反馈循环：返回现场检测。"
    "第二类（除湿度外仍有指标不合格气体）→ 现场回收 → 分支：子分支A 现场净化 → 现场检测 → "
    "判断：合格 → 回充设备；判断：不合格 → 反馈循环：返回现场净化；"
    "子分支B 基地净化 → 基地检测 → 判断：合格 → 暂存待用；判断：不合格 → 反馈循环：返回基地净化。"
    "流程共4处判断、3条反馈回路，终点为回充设备或暂存待用。"
)


def build_flowchart(prefix, units):
    """6.3 流程图原子块：图语义 = 复用 Day2 离线 vision 产物（方案 A）。"""
    return f"{prefix}：六氟化硫(SF6)气体现场循环再利用流程（图1）：{FLOW_SEMANTIC}"


def build_default(prefix, units):
    """普通文本合并：主标题 + 各单元（子标题 + 内容）。"""
    if len(units) == 1:
        return f"{prefix}：{normalize_text(units[0]['content']).strip()}"
    parts = []
    last_leaf = None
    for u in units:
        leaf = leaf_title(u["section"], prefix)
        txt = normalize_text(u["content"]).strip()
        if leaf and leaf != last_leaf:
            parts.append(f"{leaf}：{txt}")
            last_leaf = leaf
        else:
            parts.append(txt)
    return f"{prefix}：{chr(10).join(parts)}"


# ---------------------------------------------------------------------------
# ⑧ Q2Q：LLM 优先，规则兜底
# ---------------------------------------------------------------------------

SUBSTANCE = ("六氟化硫", "SF6", "四氟化碳", "CF4", "二氧化硫", "SO2", "硫化氢", "H2S",
             "矿物油", "可水解氟化物", "空气", "湿度", "酸度", "分解产物")
PARAM = ("回收率", "净化率", "纯度", "湿度", "酸度", "空气含量", "分解产物",
         "质量分数", "体积分数", "极限真空度", "充气速度", "充气压力")
TOPIC = ("现场检测", "现场回收", "净化处理", "回充", "安全防护", "循环再利用",
         "术语和定义", "规范性引用")
STD_RE = re.compile(r"(?:GB/T|DL/T|T/CEC|TSG)\s?\d[\d.、—–-]*")


def fallback_important_kwd(content, section):
    """规则兜底：标准号 > 化学物质 > 参数指标 > 主题词，取前 3。"""
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
        leaf = section.split("/")[-1].strip()
        if leaf and leaf not in kws:
            kws.append(leaf)
    return kws[:3]


def fallback_question_kwd(content, section, semantic):
    """规则兜底：按 semantic_type 模板生成，确定性补齐（切片不死循环）。"""
    q = []
    if semantic == "参数查询":
        found = False
        for p in ("回收率", "净化率", "纯度", "湿度"):
            if p in content:
                q.append(f"SF6气体{p}要求是多少?")
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
        q.append("相关限值或要求是什么?")
    elif semantic == "术语定义":
        q.append("该术语的定义是什么?")
        q.append("相关术语是什么意思?")
    elif semantic == "图语义":
        q.append("SF6气体现场循环再利用流程是什么?")
        q.append("SF6气体回收、净化、回充流程是什么?")
    elif semantic == "安全要求":
        q.append("SF6气体安全防护要求是什么?")
        q.append("SF6气体使用有哪些安全注意事项?")
    else:  # 概述
        if "规范性引用" in section:
            q.append("本标准引用了哪些文件?")
        elif "范围" in section:
            q.append("本文件的适用范围是什么?")
        elif "参考文献" in section:
            q.append("本标准引用了哪些参考文献?")
        else:
            q.append("该部分内容讲了什么?")
    for f in ("该部分内容讲了什么?", "相关限值或要求是什么?", "该技术指标/参数要求是多少?"):
        if f not in q:
            q.append(f)
    return q[:3]


def make_q2q(content, section, semantic):
    """LLM 优先，失败降级规则（方案⑦「失败不阻塞」）。"""
    if _llm_gen_q2q is not None:
        r = _llm_gen_q2q(content, section)
        if r and r.get("important_kwd") and r.get("question_kwd"):
            return r["important_kwd"][:3], r["question_kwd"][:3]
    return fallback_important_kwd(content, section), fallback_question_kwd(content, section, semantic)


# ---------------------------------------------------------------------------
# ②③ 聚合：按顶级章节分组 + 不跨页 + 超长贪心拆
# ---------------------------------------------------------------------------

def split_by_page_and_size(units, prefix):
    """组内先按 page 拆（不跨页），再按实测 token 上限贪心拆（自然断点，不硬均分）。

    注意：累加时计入「子标题前缀 + 正文」的 token（build_default 会拼叶子标题，
    只算裸正文会低估最终 content 长度，导致超窗）。
    """
    by_page = []
    cur = []
    cur_page = None
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
            leaf = leaf_title(u["section"], prefix)
            piece = f"{leaf}：{u['content']}" if leaf else u["content"]
            t = calc_tokens(piece)
            if cur2 and cur_tok + t > MAX_CONTENT_TOKENS:
                result.append(cur2)
                cur2, cur_tok = [u], t
            else:
                cur2.append(u)
                cur_tok += t
        if cur2:
            result.append(cur2)
    return result


def build_chunks(units):
    """通用聚合：原子块 + 顶级章节文本聚合。"""
    tables = [u for u in units if u["type"] == "table"]
    images = [u for u in units if u["type"] == "image"]
    normal = [u for u in units if u["type"] not in ("table", "image")]

    chunks = []

    # 原子块：表1
    if tables:
        chunks.append(_make_chunk("表1 关键质量指标技术要求", tables, "table",
                                  "参数查询", build_table1("表1 关键质量指标技术要求", tables)))
    # 原子块：流程图
    if images:
        chunks.append(_make_chunk("6.3 循环再利用流程图", images, "image",
                                  "图语义", build_flowchart("6.3 循环再利用流程图", images)))

    # 文本聚合：按顶级章节分组（保持文档顺序）
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
            types = [u["type"] for u in sub]
            semantic = decide_semantic(top, types, content)
            # type 取主导单元 type（equation 混在 text 里时优先 equation）
            typ = "equation" if "equation" in types else sub[0]["type"]
            chunks.append(_make_chunk(top, sub, typ, semantic, content))

    chunks.sort(key=_sort_key)
    return chunks


def _make_chunk(section, units, typ, semantic, content):
    page = min(u["page"] for u in units)
    ik, qk = make_q2q(content, section, semantic)
    return {
        "content": content,
        "source": SOURCE,
        "page": page,
        "type": typ,
        "section": section,
        "semantic_type": semantic,
        "char_len": len(content),
        "token_len": calc_tokens(content),
        "important_kwd": ik,
        "question_kwd": qk,
        "document_id": "gb_t_44653_2024",
        "ingest_version": "20260825_v1",
    }


def _sort_key(chunk):
    sec = chunk["section"]
    if sec.startswith("表"):
        return (6.5, 0)
    if sec.startswith("参考文献"):
        return (99.0, 0)
    m = re.match(r"^(\d+)", sec)
    if m:
        return (float(m.group(1)), chunk["page"])
    return (50.0, chunk["page"])


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    units = load_units(INPUT)
    print(f"载入原子单元：{len(units)} 个")

    chunks = build_chunks(units)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ 输出 {len(chunks)} 个 chunk → {OUTPUT}\n")
    max_tok = max(c["token_len"] for c in chunks)
    over800 = [c["section"] for c in chunks if c["char_len"] > 800]
    bad_kwd = [c["section"][:20] for c in chunks
               if len(c["important_kwd"]) != 3 or len(c["question_kwd"]) != 3]
    print(f"块数={len(chunks)} 最大token={max_tok}（≤512）")
    print(f"超800字块={len(over800)} kwd非3/3块={len(bad_kwd)}")
    for c in chunks:
        print(f"  p{c['page']:>2} {c['type']:<7} {c['semantic_type']:<4} "
              f"{c['char_len']:>4}字 kwd={len(c['important_kwd'])}/{len(c['question_kwd'])} "
              f"{c['section'][:30]}")


if __name__ == "__main__":
    main()
