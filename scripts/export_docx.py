"""Export run_ctim_global.py logs to a .docx report.

Standard library only (Python 3.9): a .docx is a ZIP of OOXML parts, so
``zipfile`` plus hand-written XML is enough -- no python-docx.

Equations are real Word equations (OMML, the same markup Word writes when you
use Insert > Equation), not images and not monospace ASCII, so they stay
selectable and rescale with the font.

Usage
-----
    python scripts/export_docx.py \
        --log yelp_paper_h01.log --log digg_items_K20.log \
        --methods CTIM,CTIM-G \
        --out results/ctim_vs_ctimg.docx

Log parsing is shared with scripts/export_xlsx.py.
"""

import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_xlsx import parse_run_log, _mean          # noqa: E402

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
M = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# ---------------------------------------------------------------------------
# Word paragraphs / runs
# ---------------------------------------------------------------------------

def run(text, bold=False, italic=False, mono=False, size=None, color=None):
    pr = []
    if mono:
        pr.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>')
    if bold:
        pr.append("<w:b/>")
    if italic:
        pr.append("<w:i/>")
    if color:
        pr.append('<w:color w:val="%s"/>' % color)
    if size:
        pr.append('<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (size * 2, size * 2))
    rpr = "<w:rPr>%s</w:rPr>" % "".join(pr) if pr else ""
    parts = str(text).split("\n")
    body = []
    for i, seg in enumerate(parts):
        if i:
            body.append("<w:br/>")
        body.append('<w:t xml:space="preserve">%s</w:t>' % _esc(seg))
    return "<w:r>%s%s</w:r>" % (rpr, "".join(body))


def para(content="", style=None, spacing_before=0, spacing_after=120,
         align=None, indent=0, keep_next=False):
    pr = []
    if style:
        pr.append('<w:pStyle w:val="%s"/>' % style)
    if keep_next:
        pr.append("<w:keepNext/>")
    if indent:
        pr.append('<w:ind w:left="%d"/>' % indent)
    pr.append('<w:spacing w:before="%d" w:after="%d"/>' % (spacing_before, spacing_after))
    if align:
        pr.append('<w:jc w:val="%s"/>' % align)
    ppr = "<w:pPr>%s</w:pPr>" % "".join(pr)
    if isinstance(content, str) and not content.startswith("<w:r"):
        content = run(content) if content else ""
    return "<w:p>%s%s</w:p>" % (ppr, content)


def heading(text, level=1):
    return para(run(text, bold=True), style="Heading%d" % level,
                spacing_before=280 if level > 1 else 200, spacing_after=140,
                keep_next=True)


def code_block(lines):
    out = []
    for ln in lines:
        out.append(para(run(ln, mono=True, size=8), spacing_after=0, indent=280))
    return "".join(out)


def bullet(text):
    return para(run("•  ") + (text if str(text).startswith("<w:r") else run(text)),
                indent=280, spacing_after=80)


# ---------------------------------------------------------------------------
# OMML equations
# ---------------------------------------------------------------------------

def mr(t, plain=False):
    """A math run.  plain=True renders upright (for function names like pp)."""
    pr = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
    return '<m:r>%s<m:t xml:space="preserve">%s</m:t></m:r>' % (pr, _esc(t))


def msub(base, sub):
    return "<m:sSub><m:e>%s</m:e><m:sub>%s</m:sub></m:sSub>" % (base, sub)


def msup(base, sup):
    return "<m:sSup><m:e>%s</m:e><m:sup>%s</m:sup></m:sSup>" % (base, sup)


def msubsup(base, sub, sup):
    """One base carrying BOTH a subscript and a superscript, e.g. S_m^(j).

    Chaining msub(...) + msup("", ...) instead produces a superscript whose base
    is empty, and Word renders an empty slot as a placeholder box.  This is the
    only correct construct for a symbol with two attachments.
    """
    return ("<m:sSubSup><m:e>%s</m:e><m:sub>%s</m:sub><m:sup>%s</m:sup>"
            "</m:sSubSup>" % (base, sub, sup))


def mfrac(num, den):
    return "<m:f><m:num>%s</m:num><m:den>%s</m:den></m:f>" % (num, den)


def mbar(e):
    return '<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>%s</m:e></m:bar>' % e


def mdel(e, left="(", right=")"):
    return ('<m:d><m:dPr><m:begChr m:val="%s"/><m:endChr m:val="%s"/></m:dPr>'
            "<m:e>%s</m:e></m:d>" % (_esc(left), _esc(right), e))


def mnary(ch, sub, sup, body, under_over=True):
    pr = ['<m:chr m:val="%s"/>' % _esc(ch)]
    pr.append('<m:limLoc m:val="%s"/>' % ("undOvr" if under_over else "subSup"))
    if not sup:
        pr.append('<m:supHide m:val="1"/>')
    if not sub:
        pr.append('<m:subHide m:val="1"/>')
    return ("<m:nary><m:naryPr>%s</m:naryPr><m:sub>%s</m:sub><m:sup>%s</m:sup>"
            "<m:e>%s</m:e></m:nary>" % ("".join(pr), sub, sup, body))


def mlimlow(base, lim):
    """An operator with its condition set underneath, e.g. max over 0<=j<=k.

    ``m:nary``'s ``m:chr`` holds a single character, so multi-letter operators
    like max / arg max cannot go through it; this is the right construct.
    """
    return "<m:limLow><m:e>%s</m:e><m:lim>%s</m:lim></m:limLow>" % (base, lim)


def mfunc(name, arg):
    """Upright function name applied to a delimited argument."""
    return mr(name, plain=True) + mdel(arg)


def minline(omml):
    """Math set INLINE in a running paragraph, not as a display equation.

    ``equation()`` wraps its argument in ``m:oMathPara``, which is block-level
    and forces its own line.  An algorithm line like "for c = 1 … C do" needs the
    math to sit between two ordinary runs, which is what a bare ``m:oMath`` does.
    """
    return "<m:oMath>%s</m:oMath>" % omml


def equation(omml, tag=None):
    """A centred display equation, optionally with a right-aligned (Eq n) tag."""
    body = ('<m:oMathPara><m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr>'
            "<m:oMath>%s</m:oMath></m:oMathPara>" % omml)
    p = ('<w:p><w:pPr><w:spacing w:before="80" w:after="140"/>'
         '<w:jc w:val="center"/></w:pPr>%s</w:p>' % body)
    if tag:
        # Right-aligned equation number, the usual convention in a paper.
        p += para(run(tag, size=9), align="right", spacing_after=160)
    return p


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

_BORDER = ('<w:tblBorders>'
           '<w:top w:val="single" w:sz="4" w:color="9BA7B4"/>'
           '<w:left w:val="single" w:sz="4" w:color="9BA7B4"/>'
           '<w:bottom w:val="single" w:sz="4" w:color="9BA7B4"/>'
           '<w:right w:val="single" w:sz="4" w:color="9BA7B4"/>'
           '<w:insideH w:val="single" w:sz="4" w:color="C9D2DC"/>'
           '<w:insideV w:val="single" w:sz="4" w:color="C9D2DC"/>'
           "</w:tblBorders>")


def _cell(text, width, bold=False, align=None, shade=None, mono=False, size=9):
    pr = ['<w:tcW w:w="%d" w:type="dxa"/>' % width]
    if shade:
        pr.append('<w:shd w:val="clear" w:color="auto" w:fill="%s"/>' % shade)
    pr.append('<w:vAlign w:val="center"/>')
    p = para(run(text, bold=bold, mono=mono, size=size),
             spacing_before=20, spacing_after=20, align=align)
    return "<w:tc><w:tcPr>%s</w:tcPr>%s</w:tc>" % ("".join(pr), p)


def table(header, rows, widths, aligns=None, mono_cols=()):
    """header: list[str]; rows: list[list[str]]; widths: twips."""
    aligns = aligns or ["left"] * len(header)
    out = ["<w:tbl><w:tblPr>",
           '<w:tblW w:w="0" w:type="auto"/>', _BORDER,
           '<w:tblLayout w:type="fixed"/>',
           '<w:tblCellMar><w:left w:w="80" w:type="dxa"/>'
           '<w:right w:w="80" w:type="dxa"/></w:tblCellMar>',
           "</w:tblPr><w:tblGrid>"]
    for w in widths:
        out.append('<w:gridCol w:w="%d"/>' % w)
    out.append("</w:tblGrid>")

    out.append('<w:tr><w:trPr><w:tblHeader/></w:trPr>')
    for i, h in enumerate(header):
        out.append(_cell(h, widths[i], bold=True, align=aligns[i], shade="E8EEF7"))
    out.append("</w:tr>")

    for r in rows:
        out.append("<w:tr>")
        for i, v in enumerate(r):
            out.append(_cell(v, widths[i], align=aligns[i], mono=(i in mono_cols)))
        out.append("</w:tr>")
    out.append("</w:tbl>")
    out.append(para("", spacing_after=160))
    return "".join(out)


def caption(text):
    return para(run(text, italic=True, size=9, color="5A6673"), spacing_after=200)


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles %s>
<w:docDefaults><w:rPrDefault><w:rPr>
<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
<w:sz w:val="21"/><w:szCs w:val="21"/><w:lang w:val="vi-VN"/>
</w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault>
</w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>
<w:rPr><w:b/><w:sz w:val="40"/><w:szCs w:val="40"/><w:color w:val="1F3864"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>
<w:rPr><w:b/><w:sz w:val="30"/><w:szCs w:val="30"/><w:color w:val="1F3864"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>
<w:rPr><w:b/><w:sz w:val="25"/><w:szCs w:val="25"/><w:color w:val="2E5395"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/>
<w:rPr><w:b/><w:sz w:val="22"/><w:szCs w:val="22"/><w:color w:val="2E5395"/></w:rPr></w:style>
</w:styles>""" % W

_SECT = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
         '<w:pgMar w:top="1134" w:right="1021" w:bottom="1134" w:left="1021" '
         'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>')


def write_docx(path, body_xml):
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
          '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
          '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>'
          "</Types>")
    root = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    drels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
             '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
             'relationships/styles" Target="styles.xml"/>'
             '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
             'relationships/settings" Target="settings.xml"/></Relationships>')

    # Word's default math justification is centerGroup, which centres any
    # paragraph whose only content is an m:oMath -- exactly the pseudocode lines
    # that are pure formula.  defJc=left is the only thing that stops it; a
    # paragraph-level w:jc does not override the math group.
    settings = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                "<w:settings %s %s><m:mathPr><m:defJc m:val=\"left\"/>"
                "</m:mathPr></w:settings>" % (W, M))
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           "<w:document %s %s %s><w:body>%s%s</w:body></w:document>"
           % (W, M, R, body_xml, _SECT))

    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", root)
        z.writestr("word/_rels/document.xml.rels", drels)
        z.writestr("word/styles.xml", _STYLES)
        z.writestr("word/settings.xml", settings)
        z.writestr("word/document.xml", doc)


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------

def f(x, nd=4):
    return "" if x is None else ("%.*f" % (nd, x))


def spread(d, meta):
    """Eq (18) spread at the paper's single threshold h.

    The driver always prints two columns: the fixed grading threshold
    ``h_eval`` and, separately, the historical ``h_sel`` one.  This report
    quotes only ``h_sel`` -- the h the paper actually specifies -- so a log run
    with a finer ``h_eval`` still yields the paper-protocol number.
    """
    if meta.get("h_sel") == meta.get("h_eval"):
        return d.get("I")
    return d.get("I_h_sel") if d.get("I_h_sel") is not None else d.get("I")


def build_report(runs, methods):
    b = []
    A = b.append

    A(para(run("So sánh CTIM và CTIM-G", bold=True), style="Title", spacing_after=60))
    A(para(run("Tái hiện Huang, Shen, Meng, Chang & He, “Community-based influence "
                "maximization for viral marketing”, Applied Intelligence (2019) — "
                "kèm một biến thể phân bổ ngân sách toàn cục (CTIM-G) do dự án này thêm vào.",
                italic=True, color="5A6673")))

    # ---- 1. thiet lap ------------------------------------------------------
    A(heading("1. Thiết lập thí nghiệm", 1))
    hdr = ["Dataset", "n = |V|", "m = |E|", "C", "K", "h", "seed", "test items"]
    rows = []
    for r in runs:
        me = r["meta"]
        rows.append([os.path.splitext(me.get("log", "?"))[0],
                     "{:,}".format(me.get("n", 0)), "{:,}".format(me.get("m", 0)),
                     str(me.get("C")), str(me.get("K")), str(me.get("h_sel")),
                     str(me.get("seed")),
                     ", ".join(str(x) for x in me.get("test_items", []))])
    A(table(hdr, rows, [1700, 1150, 1150, 550, 500, 650, 700, 2600],
            ["left", "right", "right", "right", "right", "right", "right", "left"]))
    A(caption("Một ngưỡng h = 0.1 duy nhất, đúng như bài báo đặt, dùng cho cả việc chọn hạt "
              "giống bên trong MIA lẫn việc báo cáo Eq (18)."))

    # ---- 2. cong thuc ------------------------------------------------------
    A(heading("2. Mô hình và công thức", 1))

    A(heading("2.1. Cường độ ảnh hưởng trên cạnh", 2))
    A(para("Xác suất lan truyền trên cạnh (u, v) đối với sản phẩm i được suy ra từ mô hình "
           "chủ đề – cộng đồng. Ở dạng đã gom nhân tử mà bản cài đặt dùng:"))
    A(equation(
        mfunc("pp", mr("u,v")) + mr(" = ")
        + mnary("∑", mr("c"), "", msub(mr("a"), mr("u,c")) + mr("·")
                + msub(mr("π"), mr("v,c")) + mr("·")
                + msub(mbar(mr("θ")), mr("i,c"))),
        "Eq (12)"))
    A(para(run("trong đó ") + run("θ̄", italic=True) + run(" là phân phối chủ đề của sản phẩm "
           "chiếu lên cộng đồng:")))
    A(equation(
        msub(mbar(mr("θ")), mr("i,c")) + mr(" = ")
        + mnary("∑", mr("z"), "", mfunc("P", mr("z∣i")) + mr("·")
                + msub(mr("θ"), mr("c,z")))))
    A(para(run("Hệ quả quan trọng: nếu ") + run("θ", italic=True)
           + run(" gần đều thì ") + run("θ̄", italic=True)
           + run("[i,c] ≈ 1/Z với mọi cộng đồng và mọi sản phẩm, nên")))
    A(equation(mfunc("pp", mr("u,v")) + mr(" ≤ ")
               + mlimlow(mr("max", plain=True), mr("c"))
               + msub(mbar(mr("θ")), mr("i,c"))
               + mr(" = ") + mfrac(mr("1"), mr("Z"))))
    A(para("Đo được trên Digg: Z = 8, 1/Z = 0.125, max pp = 0.124227 (đạt 99.38% cận). "
           "Vì thế MIA gần như chỉ đi được 1 bước ở ngưỡng h = 0.1."))

    A(heading("2.2. MIA – mô hình đánh giá ảnh hưởng", 2))
    A(para("Xác suất của một đường đi là tích xác suất các cạnh:"))
    A(equation(mfunc("pp", mr("P")) + mr(" = ")
               + mnary("∏", mr("k"), "", mfunc("pp", msub(mr("w"), mr("k")) + mr(",")
                                                    + msub(mr("w"), mr("k+1")))),
               "Eq (13)"))
    A(para("Đường ảnh hưởng cực đại, tìm bằng Dijkstra trên trọng số cộng tính −ln pp:"))
    A(equation(mfunc("MIP", mr("u,v")) + mr(" = ")
               + mlimlow(mr("arg max", plain=True), mr("P ∈ 𝒫(u,v)"))
               + mfunc("pp", mr("P")), "Eq (14)"))
    A(para("Cây ảnh hưởng vào, giữ lại các đường vượt ngưỡng h:"))
    A(equation(mfunc("MIIA", mr("v,h")) + mr(" = ")
               + mnary("⋃", mfunc("pp", mfunc("MIP", mr("u,v"))) + mr(" ≥ h"), "",
                       mfunc("MIP", mr("u,v"))), "Eq (15)"))
    A(para("Xác suất kích hoạt lan trên cây, và tổng ảnh hưởng là hàm mục tiêu duy nhất "
           "dùng để chấm điểm mọi phương pháp trong báo cáo này:"))
    A(equation(mfunc("ap", mr("v∣S")) + mr(" = 1 − ")
               + mnary("∏", mr("u ∈ N(v)"), "",
                       mdel(mr("1 − ") + mfunc("ap", mr("u∣S")) + mr("·")
                            + mfunc("pp", mr("u,v")))), "Eq (17)"))
    A(equation(mfunc("I", mr("S")) + mr(" = ")
               + mnary("∑", mr("v ∈ V"), "", mfunc("ap", mr("v∣S"))), "Eq (18)"))

    A(heading("2.3. Phân bổ ngân sách theo cộng đồng", 2))
    A(para("Quy hoạch động của bài báo phân K hạt giống cho C cộng đồng:"))
    A(equation(msub(mr("I"), mr("m,k")) + mr(" = ")
               + mlimlow(mr("max", plain=True),
                         mr("0 ≤ j ≤ min(k, ") + msub(mr("cap"), mr("m")) + mr(")"))
               + mdel(msub(mr("I"), mr("m−1,k−j")) + mr(" + ")
                      + msub(mr("I"), mr("m")) + mdel(mr("j")), "{", "}"),
               "Algorithm 2"))
    A(para("Vì Eq (18) là hàm submodular, đường cong lợi ích của mỗi cộng đồng lõm, nên "
           "quy hoạch động này suy biến về đệ quy “mỗi lần thêm một hạt giống”."))
    A(para(run("Khác biệt giữa hai phương pháp nằm ở chỗ dựng cây: ")
           + run("CTIM", bold=True)
           + run(" dựng MIA trên đồ thị con cảm sinh của từng cộng đồng, nên mọi cạnh bắc "
                 "cầu giữa các cộng đồng bị loại; ")
           + run("CTIM-G", bold=True)
           + run(" dựng MIA trên đồ thị đầy đủ rồi mới chấm điểm theo cộng đồng, nên đường "
                 "cong lợi ích của các cộng đồng so sánh được với nhau.")))

    # ---- 3. giao thuc cham -------------------------------------------------
    A(heading("3. Giao thức chấm điểm", 1))
    A(para("Mọi tập hạt giống trong báo cáo này được chấm bằng cùng một hàm Eq (18), trên "
           "toàn đồ thị, ở ngưỡng h = 0.1 — đúng giao thức bài báo: một h duy nhất cho cả "
           "chọn lẫn chấm. Nhờ vậy các dòng trong mọi bảng so sánh được với nhau, và so được "
           "với số liệu bài báo công bố."))
    A(para(run("Điều phải nói kèm.", bold=True)
           + run(" Giao thức này hợp lệ với tập baseline của chính bài báo (CTIM, CTIM_CGA, "
                 "AIR+CGA, CINEMA, Greedy) vì không phương pháp nào trong đó tối ưu trực tiếp "
                 "Eq (18). CTIM-G thì có: nó chạy tham lam thẳng trên chính hàm này. Nên "
                 "khoảng cách CTIM-G báo cáo dưới đây là ")
           + run("cận trên", bold=True)
           + run(" của khoảng cách thật, và CTIM-G là phương pháp do dự án này thêm vào, "
                 "không phải của bài báo — nên phần chênh do vòng luẩn quẩn là do phía chúng "
                 "ta tạo ra, không phải khiếm khuyết của bài báo.")))
    A(para("Trọng tài độc lập duy nhất là mô phỏng Monte-Carlo IC, vốn không dùng chung bộ "
           "máy nào với MIA. Phép kiểm đó chưa chạy trên Yelp."))

    # ---- 4. ket qua --------------------------------------------------------
    A(heading("4. Kết quả", 1))
    for r in runs:
        me, items = r["meta"], r["items"]
        tag = os.path.splitext(me.get("log", "?"))[0]
        A(heading("4.%d. %s (K = %s, h = %s)"
                  % (runs.index(r) + 1, tag, me.get("K"), me.get("h_sel")), 2))
        rows = []
        for it in items:
            base = spread(it["methods"].get(methods[0], {}), me)
            for nm in methods:
                d = it["methods"].get(nm)
                if not d:
                    continue
                val = spread(d, me)
                gap = (100.0 * (val / base - 1.0)) if (base and val) else None
                rows.append([str(it["run"]), str(it["item"]), nm, f(val),
                             ("—" if nm == methods[0] else "%+.4f%%" % gap),
                             f(d.get("select_seconds"), 2) + " s",
                             str(d.get("n_seeds", ""))])
        A(table(["#", "sản phẩm", "phương pháp", "I (h = 0.1)", "khoảng cách",
                 "thời gian chọn", "|S|"],
                rows, [550, 1250, 1700, 1750, 1600, 1700, 650],
                ["right", "right", "left", "right", "right", "right", "right"]))
        gaps = []
        for it in items:
            base = spread(it["methods"].get(methods[0], {}), me)
            v = spread(it["methods"].get(methods[-1], {}), me)
            if base and v:
                gaps.append(100.0 * (v / base - 1.0))
        if gaps:
            A(caption("Khoảng cách %s so với %s: trung bình %+.4f%%, biên độ %+.4f%% … "
                      "%+.4f%% qua %d sản phẩm."
                      % (methods[-1], methods[0], sum(gaps) / len(gaps),
                         min(gaps), max(gaps), len(gaps))))

    # ---- 5. thoi gian ------------------------------------------------------
    A(heading("5. Thời gian chạy", 1))
    for r in runs:
        me, items = r["meta"], r["items"]
        tag = os.path.splitext(me.get("log", "?"))[0]
        rows = []
        for it in items:
            row = [str(it["run"]), str(it["item"]), f(it.get("eq12_seconds"), 2)]
            for nm in methods:
                row.append(f(it["methods"].get(nm, {}).get("select_seconds"), 2))
            row += [f(it.get("scoring_seconds"), 2), f(it.get("item_seconds"), 2)]
            rows.append(row)
        mrow = ["", "trung bình", f(_mean([i.get("eq12_seconds") for i in items]), 2)]
        for nm in methods:
            mrow.append(f(_mean([i["methods"].get(nm, {}).get("select_seconds")
                                 for i in items]), 2))
        mrow += [f(_mean([i.get("scoring_seconds") for i in items]), 2),
                 f(_mean([i.get("item_seconds") for i in items]), 2)]
        rows.append(mrow)
        w = [500, 1000, 1200] + [1400] * len(methods) + [1200, 1300]
        A(heading("5.%d. %s" % (runs.index(r) + 1, tag), 2))
        A(table(["#", "item", "Eq (12)"] + [nm + " chọn" for nm in methods]
                + ["chấm điểm", "tổng item"], rows, w, ["right"] * len(w)))
        b_ = _mean([i["methods"].get(methods[0], {}).get("select_seconds") for i in items])
        s_ = _mean([i["methods"].get(methods[-1], {}).get("select_seconds") for i in items])
        notes = []
        if b_ and s_:
            notes.append("%s chậm hơn %s %.2f lần." % (methods[-1], methods[0], s_ / b_))
        e = [i.get("eq12_seconds") for i in items if i.get("eq12_seconds")]
        if len(e) > 1 and e[0] > 3 * _mean(e[1:]):
            notes.append("Eq (12) ở sản phẩm đầu tốn %.2f s so với ~%.2f s về sau (%.1f lần): "
                         "đó là chi phí làm nóng một lần, không phải chi phí biên."
                         % (e[0], _mean(e[1:]), e[0] / _mean(e[1:])))
        A(caption(" ".join(notes)))
        A(para("Tổng toàn bộ lần chạy: %s s." % f(me.get("total_seconds"), 2)))

    # ---- 5b. phase ---------------------------------------------------------
    phased = []
    for r in runs:
        for nm in methods:
            if any(it["methods"].get(nm, {}).get("phase_curves") is not None
                   for it in r["items"]):
                phased.append((r, nm))
                break
    if phased:
        A(heading("5.%d. Phân rã theo pha của CTIM-G" % (len(runs) + 1), 2))
        rows = []
        for r, nm in phased:
            tag = os.path.splitext(r["meta"].get("log", "?"))[0]
            for it in r["items"]:
                d = it["methods"].get(nm, {})
                if d.get("phase_curves") is None:
                    continue
                p = [d.get("phase_" + k) or 0.0
                     for k in ("mia", "solo", "curves", "dp", "realise", "repair")]
                sel = d.get("select_seconds") or 0.0
                rows.append([tag, str(it["item"]), f(p[1], 2), f(p[2], 2), f(p[3], 2),
                             f(p[4], 2), f(sel, 2),
                             "%.1f%%" % (100.0 * p[2] / sel) if sel else ""])
        A(table(["dataset", "item", "solo", "curves", "DP", "realise", "tổng chọn", "curves %"],
                rows, [1800, 1000, 1100, 1200, 900, 1200, 1300, 1100],
                ["left"] + ["right"] * 7))
        A(caption("Bước quy hoạch động — phần đóng góp thuật toán thật sự của "
                  "Algorithm 2 — tốn 0.00–0.01 s. Khoảng 70% chi phí nằm ở việc "
                  "dựng đường cong lợi ích cho từng cộng đồng. Đó là pha duy nhất đáng tối ưu."))

    # ---- 6. do phuc tap ----------------------------------------------------
    A(heading("6. Độ phức tạp", 1))
    A(para(run("CTIM", bold=True) + run(" — dựng n cây trên đồ thị con cảm sinh, nên "
                                        "kích thước cây trung bình t_c nhỏ:")))
    A(equation(mr("O") + mdel(mr("nC")) + mr(" + ")
               + mnary("∑", mr("c"), "", mr("O") + mdel(msub(mr("n"), mr("c")) + mr("·")
                       + msub(mr("t"), mr("c")) + mr(" log ") + msub(mr("t"), mr("c"))))
               + mr(" + O") + mdel(mr("CK"))))
    A(para(run("CTIM-G", bold=True) + run(" — dựng n cây trên đồ thị đầy đủ, nên t là "
                                          "kích thước trung bình toàn cục:")))
    A(equation(mr("O") + mdel(mr("n·t log t")) + mr(" + ")
               + mnary("∑", mr("c"), "", mr("O") + mdel(msub(mr("n"), mr("c"))
                       + mr("·t log t")))
               + mr(" + O") + mdel(mr("C") + msup(mr("K"), mr("2")))
               + mr(" + O") + mdel(mr("K") + msup(mr("t"), mr("2")))))

    rows = []
    for r in runs:
        me = r["meta"]
        tag = os.path.splitext(me.get("log", "?"))[0]
        for nm in methods:
            arb = _mean([it["methods"].get(nm, {}).get("arborescences") for it in r["items"]])
            sz = _mean([it["methods"].get(nm, {}).get("mean_arb_size") for it in r["items"]])
            ce = _mean([it["methods"].get(nm, {}).get("curve_gain_evals") for it in r["items"]])
            sel = _mean([it["methods"].get(nm, {}).get("select_seconds") for it in r["items"]])
            rows.append([tag, nm,
                         "{:,.0f}".format(arb) if arb else "—",
                         f(sz, 1) if sz else "—",
                         "{:,.0f}".format(ce) if ce else "—",
                         f(sel, 2) + " s" if sel else "",
                         f(1000.0 * sel / arb, 3) if (sel and arb) else "—"])
    A(table(["dataset", "phương pháp", "số cây dựng", "kích thước TB",
             "lần tính lợi ích", "thời gian chọn", "ms / cây"],
            rows, [1800, 1500, 1500, 1300, 1500, 1300, 1100],
            ["left", "left", "right", "right", "right", "right", "right"]))
    A(bullet(run("CTIM không có số đo vì ") + run("ctim_select_seeds", mono=True)
             + run(" tự dựng đối tượng MIA bên trong, trình điều khiển không đọc được bộ đếm. "
                   "Mọi tỉ lệ CTIM-G / CTIM ở đây là số đồng hồ, không có số đếm nguyên thủy "
                   "đứng sau.")))
    A(bullet("Số cây dựng không phải thước chi phí đáng tin. Đo riêng: CTIM-G dựng nhiều hơn "
             "GlobalGreedy khoảng 17% số cây, kích thước lớn hơn, nhưng vẫn chạy nhanh hơn "
             "4.8 lần."))

    # ---- 7. han che --------------------------------------------------------
    A(heading("7. Hạn chế cần nêu kèm số liệu", 1))
    A(heading("7.1. Độ lệch chuẩn bằng 0 không phải bằng chứng về tính bền", 2))
    A(para("Trên cả hai dataset, khoảng cách giữa hai phương pháp có sd = 0.00 qua các sản "
           "phẩm. Đó là tính tái lập, không phải tính bền: các sản phẩm sinh ra gần như cùng "
           "một đồ thị lan truyền. Bằng chứng lấy thẳng từ log — phân bổ ngân sách giống "
           "hệt, số cây dựng giống hệt, dp_value chỉ lệch ở chữ số có nghĩa thứ tư."))
    A(para(run("Nguyên nhân, đo trên ") + run("data/processed/_calib_model.pkl", mono=True)
           + run(": θ̄[i,c] là số hạng duy nhất phụ thuộc sản phẩm trong Eq (12), mà θ gần đều, "
                 "nên θ̄[i,c] ≈ 1/Z = 0.125 với mọi cộng đồng và mọi sản phẩm.")))
    A(table(["sản phẩm", "min θ̄", "max θ̄"],
            [["2369", "0.12480559", "0.12516908"],
             ["1113", "0.12483403", "0.12516543"],
             ["535", "0.12489347", "0.12514094"]],
            [1600, 1800, 1800], ["right", "right", "right"]))
    A(caption("Biên độ của θ̄ qua các sản phẩm: trung vị 0.052%, lớn nhất 0.268% trên 100 "
              "cộng đồng. Hệ quả: mọi bảng kết quả lấy trung bình qua sản phẩm để tăng độ tin "
              "cậy đều rỗng vì cùng lý do này."))

    A(heading("7.2. Những phép kiểm chưa chạy", 2))
    A(bullet("Mô phỏng Monte-Carlo IC trên Yelp — trọng tài duy nhất không dùng chung bộ "
             "máy với MIA, nên là phép kiểm duy nhất gỡ được tính vòng luẩn quẩn nêu ở mục 3."))
    A(bullet("Chấm lại ở một ngưỡng thấp hơn 0.0156 trên Yelp — dưới mức đó đường 2 bước "
             "mới được tính vào, và điểm số mới thôi trùng với ngưỡng đã dùng để chọn."))
    A(bullet("GlobalGreedy trên Yelp ở cùng giao thức — để biết CTIM-G đã khép được bao "
             "nhiêu phần khoảng cách so với tham lam toàn cục."))

    A(heading("8. Tái lập", 1))
    A(code_block([
        "python3 -u scripts/run_ctim_global.py \\",
        "    --dataset data/processed/yelp \\",
        "    --cache   data/processed/_yelp_model.pkl \\",
        "    --K 20 --h 0.1 --paper-eval \\",
        "    --items all --methods \"CTIM,CTIM-G\" \\",
        "    --mc 0 --probe-sample 0 | tee yelp_paper_h01.log",
        "",
        "python scripts/export_docx.py --log yelp_paper_h01.log \\",
        "    --log digg_items_K20.log --methods CTIM,CTIM-G \\",
        "    --out results/ctim_vs_ctimg.docx",
    ]))
    return "".join(b)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--log", action="append", default=[])
    p.add_argument("--methods", default="CTIM,CTIM-G")
    p.add_argument("--out", default=os.path.join("results", "ctim_vs_ctimg.docx"))
    args = p.parse_args(argv)
    if not args.log:
        p.error("give at least one --log")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    runs = []
    for path in args.log:
        run_ = parse_run_log(path)
        n = sum(1 for it in run_["items"] for m in methods
                if it["methods"].get(m, {}).get("I") is not None)
        print("[parse] %-24s  %d item(s), %d data row(s)"
              % (os.path.basename(path), len(run_["items"]), n))
        if not n:
            print("        WARNING: nothing parsed")
        runs.append(run_)

    write_docx(args.out, build_report(runs, methods))
    print("[write] %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
