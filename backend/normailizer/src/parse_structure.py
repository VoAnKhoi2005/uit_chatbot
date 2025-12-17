import re

CHAPTER_RE = re.compile(r"^\s*Chương\s+([IVXLCDM]+|\d+)[\.\:]?\s*(.*)$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^\s*Điều\s+(\d+)[\.\:]?\s*(.*)$", re.IGNORECASE)
KHOAN_RE   = re.compile(r"(?m)^\s*(\d+)[\.\)]\s+")
DIEM_RE    = re.compile(r"(?m)^\s*([a-zA-Z])\)\s+")

def parse_chapters_and_articles(text: str):
    chapters = []
    current_ch = None
    current_ar = None
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        m_ch = CHAPTER_RE.match(ln)
        m_ar = ARTICLE_RE.match(ln)
        if m_ch:
            if current_ar is not None and current_ch is not None:
                current_ch["articles"].append(current_ar); current_ar=None
            if current_ch is not None:
                chapters.append(current_ch)
            g1 = m_ch.group(1); heading = (m_ch.group(2) or "").strip()
            current_ch = {"no": g1, "heading": heading, "articles": []}
            continue
        if m_ar:
            if current_ar is not None and current_ch is not None:
                current_ch["articles"].append(current_ar)
            current_ar = {"no": int(m_ar.group(1)), "heading": (m_ar.group(2) or "").strip(), "content": ""}
            continue
        if current_ar is not None:
            current_ar["content"] += ln + "\n"
    if current_ar is not None and current_ch is not None:
        current_ch["articles"].append(current_ar)
    if current_ch is not None:
        chapters.append(current_ch)
    if not chapters:
        chapters = [{"no": 0, "heading": "", "articles": [{"no": 1, "heading": "", "content": text}]}]
    return chapters

def split_khoan_diem(text: str):
    text = (text or "").strip()
    spans = []
    for m in KHOAN_RE.finditer(text):
        k = int(m.group(1)); spans.append((k, m.start()))
    parts_k = []
    if spans:
        spans = sorted(spans, key=lambda x: x[1])
        for i,(k,s) in enumerate(spans):
            e = spans[i+1][1] if i+1 < len(spans) else len(text)
            body = text[s:e]
            body = KHOAN_RE.sub("", body, count=1).strip()
            parts_k.append((k, body))
    else:
        parts_k.append((1, text))
    out = []
    for k, body in parts_k:
        d_spans = []
        for m in DIEM_RE.finditer(body):
            d_spans.append((m.group(1).lower(), m.start()))
        parts_d = []
        if d_spans:
            d_spans = sorted(d_spans, key=lambda x: x[1])
            for i,(ch,s) in enumerate(d_spans):
                e = d_spans[i+1][1] if i+1 < len(d_spans) else len(body)
                seg = body[s:e]
                seg = DIEM_RE.sub("", seg, count=1).strip()
                parts_d.append((ch, seg))
        out.append((k, body, parts_d))
    return out
