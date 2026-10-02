"""CTIM and CTIM-G written out in the paper's algorithm style, in a .docx.

Three algorithms, factored so the shared prefix is stated once:

    Algorithm 1  the part CTIM and CTIM-G share -- Eq (12) edge weights and
                 Eq (19) community detection.  Both methods stop at exactly
                 these three outputs and then diverge.
    Algorithm 2  CTIM.  Line 1 invokes Algorithm 1; the rest is the paper's
                 Algorithm 2 lines 25-45.
    Algorithm 3  CTIM-G.  Line 1 invokes Algorithm 1; the rest is the global
                 objective, the exact resource-allocation DP, the quota
                 realisation and the independence gap.

The optional repair / local-search phase is NOT part of Algorithm 3: it is a
separate method (``CTIM-G+repair``), ``repair=False`` is the default at
ctim/ctim_global.py:303, and none of the six archived run logs used it.

Formatted like the ``algorithm`` environment of the CTIM paper: a rule above and
below, numbered lines, ``for ... do / end for``, ``if ... then / else / end if``,
and every formula spelled out inline rather than hidden behind a helper call.

A third column carries the file:line that implements each line, so the
transcription can be checked against the source one line at a time.

Sources read for this document:

    ctim/ctim.py           detect_communities        170-191
                           _CommunitySeedState       198-320
                           ctim_select_seeds         329-544
    ctim/ctim_global.py    celf_select               120-189
                           build_global_curves       197-264
                           realise_quota_greedy      272-294
                           select_seeds_global       302-463
    ctim/ea_dp.py          CommunityCurve            70-98
                           allocate_exact            332-378
    ctim/local_search.py   solo_influence            105-135
                           solo_influence_all        137-140
    ctim/influence.py      EdgeWeights / MIA         180-509

Usage
-----
    python scripts/export_algorithms_docx.py --out results/algorithms.docx
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_docx import (_esc, bullet, caption, heading, mbar, mdel,   # noqa: E402
                         mfrac, mfunc, minline, mlimlow, mnary, mr,
                         msub, msubsup, msup, para, run, table, write_docx)

# ---------------------------------------------------------------------------
# tiny math shortcuts
# ---------------------------------------------------------------------------


def M(*parts):
    """Inline math from concatenated OMML fragments.

    NEVER pass a ``w:r`` run into this -- a run nested inside ``m:oMath`` is
    invalid OOXML and Word renders it inconsistently.  Close the math, then
    append the run.
    """
    return minline("".join(parts))


def sub(base, s):
    return msub(mr(base), mr(s))


def subsup(base, s, t):
    return msubsup(mr(base), mr(s), mr(t))


def SUM(idx, body):
    return mnary("∑", mr(idx), "", body)


def PROD(idx, body):
    return mnary("∏", mr(idx), "", body)


def _op(name, cond, body):
    """An operator, with its condition set underneath only if there is one.

    An empty ``m:lim`` is not harmless: Word reserves the slot and adds the
    vertical space for a limit that is never drawn.  ``cond`` may be a plain
    string (wrapped in a math run) or ready-made OMML.
    """
    op = mr(name, plain=True)
    if not cond:
        return op + body
    lim = cond if str(cond).startswith("<m:") else mr(cond)
    return mlimlow(op, lim) + body


def ARGMAX(cond, body):
    return _op("arg max", cond, body)


def MAX(cond, body):
    return _op("max", cond, body)


def MIN(cond, body):
    return _op("min", cond, body)


def IF(arg):
    """The spread function I(·).

    ``mfunc`` sets its name UPRIGHT, which is right for pp/ap/P but wrong for
    I: the paper sets I italic, and ``I_m`` elsewhere in this document is
    italic too.  Two typefaces for one letter in one document reads as two
    different symbols.
    """
    return mr("I") + mdel(arg)


def up(s):
    """A multi-letter identifier, upright.

    Set italic, ``cap`` reads as c·a·p and ``dp`` as d·p.
    """
    return mr(s, plain=True)


def brace(inner):
    return mdel(inner, "{", "}")


def brack(inner):
    return mdel(inner, "[", "]")


def angle(inner):
    return mdel(inner, "⟨", "⟩")


# ---------------------------------------------------------------------------
# algorithm-environment renderer
# ---------------------------------------------------------------------------

W_NUM, W_BODY, W_CITE = 480, 7080, 1500

# In clean mode the citation column is dropped and the body absorbs its width,
# so every row still sums to the same grid total.
W_BODY_CLEAN = W_BODY + W_CITE

# Set by build(clean=True).  A module-level toggle rather than a parameter
# threaded through all 89 pseudocode lines: the line builders stay one
# expression per line, and there is exactly one place that can go stale.
_CLEAN = [False]

_RULE = ('<w:tcBorders><w:bottom w:val="single" w:sz="10" w:color="000000"/>'
         "</w:tcBorders>")


def _tc(content, width, borders="", indent=0, align=None, valign="top", span=1):
    # A cell wider than its grid column MUST declare w:gridSpan, otherwise Word
    # keeps it in column 1 and re-lays the whole table out to fit the conflict.
    pr = ['<w:tcW w:w="%d" w:type="dxa"/>' % width]
    if span > 1:
        pr.append('<w:gridSpan w:val="%d"/>' % span)
    pr += [borders, '<w:vAlign w:val="%s"/>' % valign]
    ppr = ['<w:spacing w:before="14" w:after="14" w:line="240" w:lineRule="auto"/>']
    if indent:
        ppr.append('<w:ind w:left="%d"/>' % indent)
    if align:
        ppr.append('<w:jc w:val="%s"/>' % align)
    p = "<w:p><w:pPr>%s</w:pPr>%s</w:p>" % ("".join(ppr), content)
    return "<w:tc><w:tcPr>%s</w:tcPr>%s</w:tc>" % ("".join(pr), p)


def algo(title, io_rows, lines):
    """lines: list of (indent, content_runs_or_str, cite).

    In clean mode the third column is not emitted at all -- the grid itself is
    two columns wide, so Word is never asked to reconcile a cell against a
    column that does not exist.
    """
    clean = _CLEAN[0]
    w_body = W_BODY_CLEAN if clean else W_BODY
    span_all = 2 if clean else 2          # title/Input/Output span num+body
    out = ["<w:tbl><w:tblPr>",
           '<w:tblW w:w="0" w:type="auto"/>',
           '<w:tblBorders><w:top w:val="single" w:sz="14" w:color="000000"/>'
           '<w:bottom w:val="single" w:sz="14" w:color="000000"/>'
           '<w:left w:val="none" w:sz="0" w:color="auto"/>'
           '<w:right w:val="none" w:sz="0" w:color="auto"/>'
           '<w:insideH w:val="none" w:sz="0" w:color="auto"/>'
           '<w:insideV w:val="none" w:sz="0" w:color="auto"/></w:tblBorders>',
           '<w:tblLayout w:type="fixed"/>',
           '<w:tblCellMar><w:left w:w="50" w:type="dxa"/>'
           '<w:right w:w="50" w:type="dxa"/>'
           '<w:top w:w="0" w:type="dxa"/>'
           '<w:bottom w:w="0" w:type="dxa"/></w:tblCellMar>',
           "</w:tblPr><w:tblGrid>",
           ('<w:gridCol w:w="%d"/><w:gridCol w:w="%d"/>' % (W_NUM, w_body))
           if clean else
           ('<w:gridCol w:w="%d"/><w:gridCol w:w="%d"/><w:gridCol w:w="%d"/>'
            % (W_NUM, W_BODY, W_CITE)),
           "</w:tblGrid>"]

    # -- title row, ruled underneath ---------------------------------------
    out.append("<w:tr>")
    out.append(_tc(run(title, bold=True, size=11), W_NUM + w_body, _RULE,
                   span=span_all))
    if not clean:
        out.append(_tc(run("nguồn", bold=True, size=7, color="7A8794"),
                       W_CITE, _RULE))
    out.append("</w:tr>")

    # -- Input / Output ----------------------------------------------------
    for i, (label, body, cite) in enumerate(io_rows):
        b = _RULE if i == len(io_rows) - 1 else ""
        content = run(label + " ", bold=True, size=9) + (
            body if str(body).startswith("<w:r") or str(body).startswith("<m:")
            else run(body, size=9))
        out.append("<w:tr>")
        out.append(_tc(content, W_NUM + w_body, b, span=span_all))
        if not clean:
            out.append(_tc(run(cite, mono=True, size=7, color="7A8794"),
                           W_CITE, b))
        out.append("</w:tr>")

    # -- numbered lines ----------------------------------------------------
    n = 0
    for indent, body, cite in lines:
        n += 1
        content = body if (str(body).startswith("<w:r")
                           or str(body).startswith("<m:")) else run(body, size=9)
        out.append("<w:tr>")
        out.append(_tc(run("%d:" % n, size=8, color="55606B"), W_NUM,
                       align="right"))
        out.append(_tc(content, w_body, indent=indent * 200))
        if not clean:
            out.append(_tc(run(cite, mono=True, size=7, color="7A8794"), W_CITE))
        out.append("</w:tr>")

    out.append("</w:tbl>")
    out.append(para("", spacing_after=200))
    return "".join(out)


def kw(s):
    return run(s, bold=True, size=9)


def tx(s):
    return run(s, size=9)


def cm(s):
    """An inline reading note.  Emitted only by the annotated build."""
    if _CLEAN[0]:
        return ""
    return run("   ▷ " + s, italic=True, size=8, color="6B7683")


def ref(s):
    """A pointer to another algorithm, set apart from ordinary prose."""
    return run(s, bold=True, size=9, color="1F4E79")


# All paths are relative to the package directory ctim/ ; the prefix is
# dropped from the citation column to leave room for the algorithm body.
C = "ctim.py"
G = "ctim_global.py"
E = "ea_dp.py"
L = "local_search.py"
I = "influence.py"


# ---------------------------------------------------------------------------
# Algorithm 1 -- the shared part
# ---------------------------------------------------------------------------

def shared_lines():
    return [
        (0, M(SUM("c′ = 1..C",
                  msub(mr("π"), mr("u,c′")) + mr(" · ") + msub(mr("η"), mr("c′,c")))
              + mr(" → ") + msub(mr("a"), mr("u")) + brack(mr("c")))
         + cm("nửa KHÔNG phụ thuộc sản phẩm của Eq (11); đệm theo u, O(C²) một lần"),
         I + ":188-192"),
        (0, M(SUM("z = 1..Z", mfunc("P", mr("z | i")) + mr(" · ")
                  + msub(mr("θ"), mr("c,z")))
              + mr(" → ") + msub(mbar(mr("θ")), mr("i")) + brack(mr("c")))
         + cm("nửa PHỤ THUỘC sản phẩm của Eq (12); đệm theo i"),
         I + ":205-211"),
        (0, kw("for ") + tx("mỗi cung có hướng ") + M(mdel(mr("u, v")) + mr(" ∈ ℰ"))
         + kw(" do"), I + ":261"),
        (1, M(mfunc("pp", mr("u, v")) + mr(" ← ")
              + SUM("c = 1..C",
                    msub(mr("a"), mr("u")) + brack(mr("c")) + mr(" · ")
                    + msub(mr("π"), mr("v,c")) + mr(" · ")
                    + msub(mbar(mr("θ")), mr("i")) + brack(mr("c"))))
         + cm("Eq (10)+(11)+(12) đã nhân tử hoá; kẹp về [0,1]. Quét MỌI cung của "
              "𝒢 chứ không chỉ các cặp trong 𝒟 — độ lệch ghi ở ctim.py:59-65"),
         I + ":224-252"),
        (0, kw("end for"), ""),
        (0, kw("for ") + M(mr("v = 1..U")) + kw(" do")
         + cm("Algorithm 2 dòng in 22–24 của bài báo"), C + ":178"),
        (1, M(msup(mr("c"), mr("v")) + mr(" ← ")
              + ARGMAX("c = 1..C", msub(mr("π"), mr("v,c"))))
         + tx("   (hoà ⇒ c nhỏ nhất;  hàng ")
         + M(msub(mr("π"), mr("v"))) + tx(" rỗng ⇒ ")
         + M(msup(mr("c"), mr("v")) + mr(" ← 1")) + tx(")")
         + cm("Eq (19). So sánh “>” CHẶT ⇒ hoà lấy chỉ số cộng đồng NHỎ NHẤT; "
              "hàng π_v rỗng ⇒ c^v ← 1"), C + ":183-190"),
        (0, kw("end for"), ""),
        (0, M(msub(mr("c"), mr("m")) + mr(" ← ")
              + brace(mr("v ∈ 𝒰 : ") + msup(mr("c"), mr("v")) + mr(" = m"))
              + mr(",   m = 1..C"))
         + cm("phân hoạch 𝒰 thành C tập RỜI NHAU; mỗi c_m tự động sắp tăng theo v"),
         C + ":413-417"),
        (0, kw("return ") + M(mfunc("pp", mr("·, ·")) + mr(", ")
                              + brace(msup(mr("c"), mr("v"))) + mr(", ")
                              + brace(msub(mr("c"), mr("m")))),
         C + ":405-417"),
    ]


# ---------------------------------------------------------------------------
# Algorithm 2 -- CTIM
# ---------------------------------------------------------------------------

def ctim_lines():
    return [
        (0, kw("thực hiện ") + ref("Thuật toán 1") + tx("  ⟶  ")
         + M(mfunc("pp", mr("·, ·")) + mr(", ") + brace(msup(mr("c"), mr("v")))
             + mr(", ") + brace(msub(mr("c"), mr("m"))))
         + cm("CTIM luôn tự dựng pp: lời gọi ở :406 là VÔ ĐIỀU KIỆN, hàm không "
              "có tham số pp. C = |η| lấy ở :375"), C + ":405-417"),
        (0, M(mr("𝒮 ← ∅;   ") + msub(mr("𝒮"), mr("1")) + mr(" ← … ← ")
              + msub(mr("𝒮"), mr("C")) + mr(" ← ∅"))
         + cm("dòng in 25. 𝒮 là danh sách theo thứ tự chọn; 𝒮_m nằm trong "
              "_CommunitySeedState :225-226"), C + ":422-423, 225-226"),
        (0, kw("for ") + M(mr("m = 1..C")) + kw(" do"), C + ":432"),
        (1, kw("if ") + M(msub(mr("c"), mr("m")) + mr(" = ∅")) + kw(" then ")
         + tx("bỏ qua m") + kw(" end if")
         + cm("cộng đồng rỗng không có trạng thái ⇒ ΔI_m ≡ 0; nguồn gốc nhánh "
              "dự phòng ở dòng 25"), C + ":433-435"),
        (1, M(msub(mr("𝒢"), mr("m")) + mr(" ← ")
              + mdel(msub(mr("c"), mr("m")) + mr(", ") + msub(mr("ℰ"), mr("m")))
              + mr(",   ") + msub(mr("ℰ"), mr("m")) + mr(" = ")
              + brace(mdel(mr("u, v")) + mr(" ∈ ℰ :  u ∈ ")
                      + msub(mr("c"), mr("m")) + mr("  ∧  v ∈ ")
                      + msub(mr("c"), mr("m"))))
         + cm("ĐIỂM RẼ NHÁNH. MIA(…, nodes = c_m): một cung chỉ tồn tại khi CẢ "
              "HAI đầu mút nằm trong c_m, nên S ∩ c_m = S_m"),
         C + ":437 → " + I + ":300-319"),
        (1, M(mr("∀v ∈ ") + msub(mr("c"), mr("m")) + mr(":   ")
              + mfunc("MIIA_m", mr("v, h")) + mr(" ← ")
              + brace(mr("x ∈ ") + msub(mr("c"), mr("m")) + mr(" : ")
                      + mfunc("pp", mfunc("MIP", mr("x, v"))) + mr(" ≥ h")))
         + cm("Eq (13)–(16). pp(P) là TÍCH nên cực đại pp(P) ⇔ cực tiểu "
              "Σ(−ln pp): cây Dijkstra chính là cây MIP, cắt tại −ln h"),
         I + ":321-326, 333-390"),
        (1, M(msub(mr("IncInf"), mr("m")) + brack(mr("u")) + mr(" ← 0,   ∀u ∈ ")
              + msub(mr("c"), mr("m")))
         + cm("bất biến: IncInf_m[u] = I_m(𝒮_m ∪ {u}) − I_m(𝒮_m), với I_m là "
              "Eq (18) TRÊN 𝒢_m"), C + ":227"),
        (1, kw("for ") + M(mr("v ∈ ") + msub(mr("c"), mr("m"))) + kw(" do"),
         C + ":244"),
        (2, M(mr("∀w ∈ ") + mfunc("MIIA_m", mr("v, h")) + mr(":   ")
              + msub(mr("IncInf"), mr("m")) + brack(mr("w")) + mr(" ← ")
              + msub(mr("IncInf"), mr("m")) + brack(mr("w")) + mr(" + ")
              + mfunc("ap", mr("v | ") + brace(mr("w"))))
         + cm("với 𝒮_m = ∅ thì IncInf_m[u] = I_m({u}); ap(v|{w}) ≠ 0 chỉ khi "
              "w ∈ MIIA(v,h) nên vòng này chạm đúng các số hạng khác 0 của Eq (18)"),
         C + ":245-248"),
        (1, kw("end for"), ""),
        (1, M(msub(mr("H"), mr("m")) + mr(" ← heap ")
              + brace(mdel(mr("−") + msub(mr("IncInf"), mr("m")) + brack(mr("u"))
                           + mr(", u")) + mr(" : u ∈ ") + msub(mr("c"), mr("m"))))
         + cm("khoá phụ là chỉ số đỉnh ⇒ hoà lấy đỉnh NHỎ NHẤT; trả lời dòng in "
              "34 trong O(log|c_m|)"), C + ":229-230"),
        (0, kw("end for"), ""),
        (0, M(mr("I") + brack(mr("m, k")) + mr(" ← 0,   s") + brack(mr("m, k"))
              + mr(" ← 0,   ∀m = 0..C, ∀k = 0..K"))
         + cm("dòng in 26–31; hàng 0 và cột 0 là biên"), C + ":443-449"),
        (0, kw("for ") + M(mr("k = 1..K")) + kw(" do")
         + cm("dòng in 32; mỗi vòng k chọn ĐÚNG MỘT hạt giống"), C + ":456"),
        (1, kw("for ") + M(mr("m = 1..C")) + kw(" do")
         + cm("dòng in 33"), C + ":458"),
        (2, M(mr("Δ") + msub(mr("I"), mr("m")) + mr(" ← ")
              + MAX("u ∈ c_m ∖ 𝒮_m",
                    mdel(msub(mr("I"), mr("m"))
                         + mdel(msub(mr("𝒮"), mr("m")) + mr(" ∪ ") + brace(mr("u")))
                         + mr(" − ") + msub(mr("I"), mr("m"))
                         + mdel(msub(mr("𝒮"), mr("m"))))))
         + cm("dòng in 34; đọc từ bảng IncInf_m qua H_m, hoà ⇒ chỉ số nút nhỏ nhất"),
         C + ":459-466 → :252-271"),
        (2, M(msubsup(mr("u"), mr("m"), mr("*")) + mr(" ← ")
              + ARGMAX("u ∈ c_m ∖ 𝒮_m",
                       mdel(msub(mr("I"), mr("m"))
                            + mdel(msub(mr("𝒮"), mr("m")) + mr(" ∪ ")
                                   + brace(mr("u")))
                            + mr(" − ") + msub(mr("I"), mr("m"))
                            + mdel(msub(mr("𝒮"), mr("m"))))))
         + tx("   (hoà ⇒ u nhỏ nhất;  nếu ")
         + M(msub(mr("c"), mr("m")) + mr(" = ∅")) + tx(" thì ")
         + M(mr("Δ") + msub(mr("I"), mr("m")) + mr(" ← 0")) + tx(" và ")
         + M(msubsup(mr("u"), mr("m"), mr("*")) + mr(" ← ⊥")) + tx(")")
         + cm("nhánh ∅ này là nguồn gốc của dòng 23"), C + ":459-466"),
        (2, M(mr("I") + brack(mr("m, k")) + mr(" ← ")
              + MAX("", mdel(mr("I") + brack(mr("m−1, k")) + mr(",   I")
                             + brack(mr("C, k−1")) + mr(" + Δ")
                             + msub(mr("I"), mr("m")))))
         + cm("dòng in 35, chế độ mặc định paper-true. I[C,k−1] KHÔNG phụ thuộc m "
              "⇒ khai triển thành I[C,k] = I[C,k−1] + max_m ΔI_m: DP suy biến "
              "thành THAM LAM trên cộng đồng, O(K·C)"), C + ":471-474"),
        (2, kw("if ") + M(mr("I") + brack(mr("C, k−1")) + mr(" + Δ")
                          + msub(mr("I"), mr("m")) + mr(" ≥ I")
                          + brack(mr("m−1, k"))) + kw(" then ")
         + M(mr("s") + brack(mr("m, k")) + mr(" ← m"))
         + cm("dòng in 36–37; dấu ≥ KHÔNG chặt ⇒ hoà thì cộng đồng chỉ số LỚN "
              "hơn thắng"), C + ":479-481"),
        (2, kw("else ") + M(mr("s") + brack(mr("m, k")) + mr(" ← s")
                            + brack(mr("m−1, k"))) + kw(" end if")
         + cm("dòng in 38–39; truyền con trỏ quay lui dọc chiều m"),
         C + ":482-483"),
        (1, kw("end for"), ""),
        (1, M(mr("j ← s") + brack(mr("C, k")) + mr(";   ")
              + msub(mr("u"), mr("k")) + mr(" ← ")
              + msubsup(mr("u"), mr("j"), mr("*")))
         + cm("dòng in 42–43. “I” ở dòng in 43 phải đọc là I_j (cảm sinh), không "
              "phải I toàn cục — xem ctim.py:84-88"), C + ":487-491"),
        (1, kw("if ") + M(msub(mr("u"), mr("k")) + mr(" = ⊥")) + kw(" then")
         + cm("DỰ PHÒNG: DP trỏ vào cộng đồng rỗng hoặc đã cạn ứng viên"),
         C + ":492"),
        (2, M(mr("j ← ")
              + ARGMAX("m : u*_m ≠ ⊥", mr("Δ") + msub(mr("I"), mr("m")))
              + mr(";   ") + msub(mr("u"), mr("k")) + mr(" ← ")
              + msubsup(mr("u"), mr("j"), mr("*")))
         + tx("   (hoà ⇒ m nhỏ nhất)")
         + cm("duyệt m TĂNG DẦN, so sánh “>” chặt ⇒ hoà lấy m nhỏ nhất; "
              "n_fallbacks ← n_fallbacks + 1 ở :507"), C + ":496-507"),
        (2, kw("if ") + M(msub(mr("u"), mr("k")) + mr(" = ⊥")) + kw(" then ")
         + tx("thoát vòng lặp k") + kw(" end if")
         + cm("không còn đỉnh chọn được ở bất kỳ cộng đồng nào ⇒ |𝒮| < K"),
         C + ":503-505"),
        (1, kw("end if"), ""),
        (1, kw("for ") + M(mr("v ∈ ") + mfunc("MIOA_j", msub(mr("u"), mr("k"))
                                              + mr(", h"))) + kw(" do")
         + cm("trừ đóng góp của các đỉnh bị ảnh hưởng, tính tại 𝒮_j CŨ"),
         C + ":290-294"),
        (2, M(mr("∀w ∈ ") + mfunc("MIIA_j", mr("v, h")) + mr(" ∖ ")
              + mdel(msub(mr("𝒮"), mr("j")) + mr(" ∪ ")
                     + brace(msub(mr("u"), mr("k")))) + mr(":   ")
              + msub(mr("IncInf"), mr("j")) + brack(mr("w")) + mr(" −= ")
              + brack(mfunc("ap", mr("v | ") + msub(mr("𝒮"), mr("j"))
                            + mr(" ∪ ") + brace(mr("w")))
                      + mr(" − ") + mfunc("ap", mr("v | ")
                                          + msub(mr("𝒮"), mr("j"))))),
         C + ":296-300"),
        (1, kw("end for"), ""),
        (1, M(msub(mr("𝒮"), mr("j")) + mr(" ← ") + msub(mr("𝒮"), mr("j"))
              + mr(" ∪ ") + brace(msub(mr("u"), mr("k"))))
         + cm("dòng in 44, phần cộng đồng"), C + ":302-303"),
        (1, kw("for ") + M(mr("v ∈ ") + mfunc("MIOA_j", msub(mr("u"), mr("k"))
                                              + mr(", h"))) + kw(" do")
         + cm("cộng lại đóng góp, tính tại 𝒮_j MỚI"), C + ":307"),
        (2, M(mr("∀w ∈ ") + mfunc("MIIA_j", mr("v, h")) + mr(" ∖ ")
              + msub(mr("𝒮"), mr("j")) + mr(":   ")
              + msub(mr("IncInf"), mr("j")) + brack(mr("w")) + mr(" += ")
              + brack(mfunc("ap", mr("v | ") + msub(mr("𝒮"), mr("j"))
                            + mr(" ∪ ") + brace(mr("w")))
                      + mr(" − ") + mfunc("ap", mr("v | ")
                                          + msub(mr("𝒮"), mr("j"))))),
         C + ":309-314"),
        (1, kw("end for"), ""),
        (1, tx("đẩy lại mọi w vừa đổi vào ") + M(msub(mr("H"), mr("j")))
         + tx(" theo thứ tự tăng dần")
         + cm("chỉ các khoá bẩn được đẩy lại; heap giữ tính lười"),
         C + ":316-317"),
        (1, M(mr("𝒮 ← 𝒮 ∪ ") + brace(msub(mr("u"), mr("k"))))
         + cm("dòng in 44, phần toàn cục; nối vào cuối để giữ thứ tự chọn"),
         C + ":514-515"),
        (0, kw("end for"), ""),
        (0, kw("return ") + M(mr("𝒮")), C + ":544"),
    ]


# ---------------------------------------------------------------------------
# Algorithm 3 -- CTIM-G  (no repair phase)
# ---------------------------------------------------------------------------

def ctimg_lines():
    return [
        (0, kw("thực hiện ") + ref("Thuật toán 1") + tx("  ⟶  ")
         + M(mfunc("pp", mr("·, ·")) + mr(", ") + brace(msup(mr("c"), mr("v")))
             + mr(", ") + brace(msub(mr("c"), mr("m"))))
         + cm("pp là tham số VỊ TRÍ (:302) và comm là TUỲ CHỌN (:305): caller có "
              "thể tiêm sẵn cả hai. Eq (19) dùng chung BYTE-FOR-BYTE với Thuật "
              "toán 2 — cùng hàm, import ở :98"), G + ":375-388"),
        (0, M(mr("∀v ∈ 𝒰:   ") + mfunc("MIIA", mr("v, h")) + mr(" ← ")
              + brace(mr("x ∈ 𝒰 : ") + mfunc("pp", mfunc("MIP", mr("x, v")))
                      + mr(" ≥ h")))
         + cm("ĐIỂM RẼ NHÁNH. MỘT bộ đánh giá duy nhất trên TOÀN 𝒢, KHÔNG có "
              "tham số nodes=. Đây là toàn bộ khác biệt về hàm mục tiêu so với "
              "Thuật toán 2 dòng 5"), G + ":224"),
        (0, M(mr("∀u ∈ 𝒰:   ") + mfunc("σ", mr("u")) + mr(" = ")
              + IF(brace(mr("u"))) + mr(" = ")
              + SUM("v ∈ MIOA(u,h)", mfunc("pp", mfunc("MIP", mr("u, v")))))
         + cm("với |S| = 1 đệ quy Eq (17) sụp đổ thành tích pp dọc MIP: một lượt "
              "Dijkstra xuôi, không dựng MIIA. Tính MỘT lần, dùng lại cho mọi "
              "cộng đồng và cho pha 3"), L + ":119-140"),
        (0, run("PHA 1 — đường cong độc lập theo cộng đồng", bold=True, size=9,
                color="1F4E79"), ""),
        (0, kw("for ") + M(mr("m = 1..C")) + kw(" với ")
         + M(msub(mr("c"), mr("m")) + mr(" ≠ ∅")) + kw(" do"), G + ":235-238"),
        (1, M(msub(mr("I"), mr("m")) + mdel(mr("0")) + mr(" ← 0;   ")
              + msubsup(mr("𝒮"), mr("0"), mr("m")) + mr(" ← ∅;   ")
              + msub(up("cap"), mr("m")) + mr(" ← ")
              + MIN("", mdel(mr("K, |") + msub(mr("c"), mr("m")) + mr("|"))))
         + cm("cap_m là trần thật sự của đường cong; nó phụ thuộc K nên đường "
              "cong KHÔNG tái dùng được giữa các K khác nhau"),
         E + ":85-86; " + G + ":240"),
        (1, kw("for ") + M(mr("j = 1..") + msub(up("cap"), mr("m"))) + kw(" do"),
         G + ":245"),
        (2, M(msubsup(mr("u"), mr("j"), mr("m")) + mr(" ← ")
              + ARGMAX("u ∈ c_m ∖ 𝒮_{j−1}^m",
                       brack(IF(msubsup(mr("𝒮"), mr("j−1"), mr("m"))
                                   + mr(" ∪ ") + brace(mr("u")))
                             + mr(" − ") + IF(msubsup(mr("𝒮"), mr("j−1"),
                                                              mr("m"))))))
         + tx("   (hoà ⇒ id nút nhỏ nhất)")
         + cm("bể ứng viên VẪN là c_m — hệt dòng in 34 — nhưng I là Eq (18) trên "
              "𝒢 TOÀN CỤC, không phải I_m cảm sinh. Đo với 𝒮 chỉ gồm hạt giống "
              "của CHÍNH cộng đồng này: đó là nguồn gốc tính lạc quan"),
         G + ":239-240"),
        (3, tx("CELF lười, ε = 10⁻¹²:  khoá đầu ") + M(msub(mr("κ"), mr("u"))
                                                       + mr(" ← ") + mfunc("σ", mr("u")))
         + tx("; lấy đỉnh heap, tính lại ")
         + M(mr("g = ") + SUM("v ∈ MIOA(u,h)",
                              brack(mfunc("ap", mr("v | 𝒮 ∪ ") + brace(mr("u")))
                                    + mr(" − ") + mfunc("ap", mr("v | 𝒮")))))
         + tx("; nếu ") + M(mr("|") + msub(mr("κ"), mr("u")) + mr(" − g| > ε"))
         + tx(" thì hạ khoá và xét lại, ngược lại chấp nhận")
         + cm("σ(u) là chặn TRÊN hợp lệ nhờ tính dưới mô-đun của Eq (18); "
              "chấp nhận chỉ khi khoá BẰNG giá trị vừa tính lại"),
         G + ":159, 168-176"),
        (2, M(msubsup(mr("𝒮"), mr("j"), mr("m")) + mr(" ← ")
              + msubsup(mr("𝒮"), mr("j−1"), mr("m")) + mr(" ∪ ")
              + brace(msubsup(mr("u"), mr("j"), mr("m"))) + mr(";   ")
              + msub(mr("I"), mr("m")) + mdel(mr("j")) + mr(" ← ")
              + msub(mr("I"), mr("m")) + mdel(mr("j−1")) + mr(" + ")
              + msubsup(mr("g"), mr("j"), mr("m")))
         + cm("lợi ích biên CHÍNH XÁC ⇒ tổng luỹ tiến triệt tiêu dây chuyền, "
              "I_m(j) = I(𝒮_j^m) đúng bằng Eq (18)"), G + ":246-249"),
        (1, kw("end for"), ""),
        (1, kw("for ") + M(mr("j = 1..") + msub(up("cap"), mr("m")))
         + kw(" do  if ") + M(msub(mr("I"), mr("m")) + mdel(mr("j")) + mr(" < ")
                              + msub(mr("I"), mr("m")) + mdel(mr("j−1")))
         + kw(" then ") + M(msub(mr("I"), mr("m")) + mdel(mr("j")) + mr(" ← ")
                            + msub(mr("I"), mr("m")) + mdel(mr("j−1")))
         + kw(" end if end for")
         + cm("ép đơn điệu không giảm. Với CELF đây là phép KHÔNG LÀM GÌ, nhưng "
              "nó là tiền điều kiện mà DP dựa vào"), E + ":95-98"),
        (0, kw("end for"), ""),
        (0, run("PHA 2 — quy hoạch động phân bổ ngân sách, chính xác",
                bold=True, size=9, color="1F4E79"), ""),
        (0, M(mr("I") + brack(mr("0, k")) + mr(" ← 0  ∀k;   I") + brack(mr("m, k"))
              + mr(" ← −∞  ∀m ≥ 1;   take") + brack(mr("m, k")) + mr(" ← 0"))
         + cm("hai bảng (C+1)×(K+1): I giữ giá trị, take giữ con trỏ quay lui"),
         E + ":342-347"),
        (0, kw("for ") + M(mr("m = 1..C")) + kw(" do"), E + ":349"),
        (1, kw("for ") + M(mr("k = 0..K")) + kw(" do"), E + ":353"),
        (2, M(mr("I") + brack(mr("m, k")) + mr(" ← ")
              + MAX("0 ≤ j ≤ min(cap_m, k)",
                    mdel(mr("I") + brack(mr("m−1, k−j")) + mr(" + ")
                         + msub(mr("I"), mr("m")) + mdel(mr("j")))))
         + cm("KHÁC HẲN dòng in 35 của Thuật toán 2: có vòng lặp j và có số hạng "
              "cap, nên cấp được NHIỀU hạt giống cho một cộng đồng trong một "
              "quyết định. Số hạng cap cũng là điều kiện ĐÚNG ĐẮN, không phải "
              "tối ưu hoá: I_m(j) không tồn tại khi j > cap_m"), E + ":354-365"),
        (2, M(mr("take") + brack(mr("m, k")) + mr(" ← j"))
         + tx(" nhỏ nhất đạt cực đại")
         + cm("so sánh CHẶT v > best + 10⁻¹⁵ với j tăng dần ⇒ hoà lấy j NHỎ hơn: "
              "ít hạt giống hơn cho cùng giá trị, nên Σ a_m có thể < K"),
         E + ":362-366"),
        (1, kw("end for"), ""),
        (0, kw("end for"), ""),
        (0, M(mr("k ← K;   a ← ∅;   ") + msub(mr("𝒮"), up("cat")) + mr(" ← ")
              + angle(mr("")))
         , E + ":368-370"),
        (0, kw("for ") + M(mr("m = C, C−1, …, 1")) + kw(" do"), E + ":371"),
        (1, kw("if ") + M(mr("take") + brack(mr("m, k")) + mr(" = j > 0"))
         + kw(" then ") + M(msub(mr("a"), mr("m")) + mr(" ← j;   ")
                            + msub(mr("𝒮"), up("cat")) + mr(" ← ")
                            + msub(mr("𝒮"), up("cat")) + mr(" ⊕ ")
                            + msubsup(mr("𝒮"), mr("j"), mr("m")) + mr(";   k ← k − j"))
         + kw(" end if")
         + cm("a khoá theo NHÃN CỘNG ĐỒNG GỐC, không theo chỉ số mảng; cộng đồng "
              "0 suất VẮNG MẶT khỏi a chứ không mang giá trị 0"), E + ":372-376"),
        (0, kw("end for"), ""),
        (0, M(mr("dp ← I") + brack(mr("C, K")) + mr(" = ")
              + SUM("m", msub(mr("I"), mr("m"))
                    + mdel(msub(mr("a"), mr("m")))))
         + cm("DỰ BÁO. Từng số hạng chính xác, nhưng phép CỘNG thì không: các "
              "đường cong dựng độc lập nên phần chồng lấn bị đếm hai lần. "
              "dp ≥ I(𝒮_cat) luôn đúng do dưới cộng tính"), E + ":378"),
        (0, run("PHA 3 — hiện thực hoá dưới ràng buộc hạn ngạch",
                bold=True, size=9, color="1F4E79"), ""),
        (0, M(mr("A ← ") + mnary("⋃", mr("m : a_m > 0"), "", msub(mr("c"), mr("m")))
              + mr(";   B ← ") + MIN("", mdel(mr("K, ") + SUM("m", msub(mr("a"), mr("m"))))))
         + cm("nếu a = ∅ thì trả về ⟨⟩ ngay (:283). Cộng đồng 0 suất KHÔNG có "
              "trong A ⇒ nút của nó không bao giờ được chọn"), G + ":283-293"),
        (0, M(msub(mr("𝒮"), mr("r")) + mr(" ← ∅;   used") + brack(mr("m"))
              + mr(" ← 0  ∀m")), G + ":150-156"),
        (0, kw("while ") + M(mr("|") + msub(mr("𝒮"), mr("r")) + mr("| < B"))
         + kw(" và còn ứng viên do")
         + cm("số vòng lặp NHIỀU HƠN số hạt giống: có vòng bị hạn ngạch chặn và "
              "vòng hạ khoá cũ, cả hai đều không tăng |𝒮_r|"), G + ":158"),
        (1, M(mr("u ← ")
              + ARGMAX("u ∈ A ∖ 𝒮_r,  used[c^u] < a_{c^u}",
                       brack(IF(msub(mr("𝒮"), mr("r")) + mr(" ∪ ")
                                   + brace(mr("u")))
                             + mr(" − ") + IF(msub(mr("𝒮"), mr("r"))))))
         + tx("   (hoà ⇒ id nút nhỏ nhất)")
         + cm("RÀNG BUỘC HẠN NGẠCH nằm dưới dấu argmax. Ứng viên thuộc cộng đồng "
              "đã đầy bị loại VĨNH VIỄN khỏi heap trước khi tính lợi ích (0 phép "
              "tính), và được đếm vào quota_blocks. Hoà ⇒ id nút nhỏ nhất"),
         G + ":159-176"),
        (1, M(mr("g ← ") + IF(msub(mr("𝒮"), mr("r")) + mr(" ∪ ")
                                 + brace(mr("u"))) + mr(" − ")
              + IF(msub(mr("𝒮"), mr("r"))))
         + cm("𝒮_r là tập DÙNG CHUNG trên mọi cộng đồng ⇒ ap(v|𝒮) chia sẻ ⇒ "
              "chồng lấn được đếm ĐÚNG MỘT LẦN. Đây là chỗ trả nợ cho hư cấu "
              "độc lập ở pha 1"), G + ":168-172"),
        (1, kw("if ") + M(mr("g ≤ 0")) + kw(" và ")
         + M(msub(mr("𝒮"), mr("r")) + mr(" ≠ ∅")) + kw(" then ")
         + tx("thoát vòng lặp") + kw(" end if")
         + cm("cùng quy tắc dừng với MIA.greedy_incremental ⇒ |𝒮| có thể < K"),
         G + ":178-179"),
        (1, M(msub(mr("𝒮"), mr("r")) + mr(" ← ") + msub(mr("𝒮"), mr("r"))
              + mr(" ∪ ") + brace(mr("u")) + mr(";   used")
              + brack(msup(mr("c"), mr("u"))) + mr(" += 1"))
         + cm("và xoá sạch bộ nhớ đệm lợi ích: 𝒮 đã đổi nên mọi giá trị cũ vô "
              "nghĩa — yêu cầu ĐÚNG ĐẮN, không phải dọn bộ nhớ"),
         G + ":181-187"),
        (0, kw("end while"), ""),
        (0, run("PHA 4 — nghiệm thu và khoảng cách độc lập",
                bold=True, size=9, color="1F4E79"), ""),
        (0, M(mr("realised ← ") + IF(msub(mr("𝒮"), mr("r"))) + mr(" = ")
              + SUM("v ∈ 𝒰", mfunc("ap", mr("v | ") + msub(mr("𝒮"), mr("r")))))
         + cm("một lần đánh giá Eq (18) THẬT, duyệt từng đỉnh; không cộng mảnh"),
         G + ":421"),
        (0, M(mr("concat ← ") + IF(msub(mr("𝒮"), up("cat"))) + mr(" = ")
              + SUM("v ∈ 𝒰", mfunc("ap", mr("v | ") + msub(mr("𝒮"), up("cat")))))
         + cm("𝒮_cat là cách GHÉP — đúng cách Thuật toán 2 lắp lời giải. Giữ lại "
              "làm đối chứng"), G + ":422"),
        (0, kw("if ") + M(mr("concat > realised + 10⁻¹²")) + kw(" then ")
         + M(mr("𝒮 ← ") + msub(mr("𝒮"), up("cat")) + mr(",  value ← concat"))
         + cm("so HAI GIÁ TRỊ THẬT, không bao giờ đụng dp. Lưới an toàn: lượt "
              "tham lam có hạn ngạch cũng chỉ là tham lam"), G + ":428-433"),
        (0, kw("else ") + M(mr("𝒮 ← ") + msub(mr("𝒮"), mr("r"))
                            + mr(",  value ← realised")) + kw(" end if"),
         G + ":435-436"),
        (0, M(mr("Γ ← dp − value"))
         + cm("CHẶN TRÊN của mức vi phạm giả thiết độc lập, không đúng bằng: nó "
              "hấp thụ cả phần đường cong lạc quan hơn các mảnh thật sự được "
              "chọn ở pha 3"), G + ":438-441"),
        (0, kw("return ") + M(mr("𝒮, value, ") + brace(msub(mr("a"), mr("m")))
                              + mr(", dp, Γ")), G + ":460-463"),
    ]


# ---------------------------------------------------------------------------
# Concise builds -- the paper's own notation, implementation mechanics dropped
#
# These are what the clean (paste-into-the-paper) document renders.  They follow
# the notation of the printed Algorithm 2, transcribed at ctim/ctim.py:9-55:
#   *  script letters  G = (U, E),  D,  S,  S_m,  c_m,  c^v_m
#   *  "=" for ordinary assignment, "<-" only for an arg max result
#   *  a side condition trails the expression after a comma,  ", u in c_m"
#   *  I[m,k] and s[m,k] for the two tables, dI_m for the per-community gain
# Heaps, CELF lazy evaluation, the incremental IncInf repair, cache
# invalidation and the diagnostic counters are deliberately absent: they are
# implementation, not algorithm, and the printed Algorithm 2 has none of them.
# The detailed build keeps every one of them, with its citation.
# ---------------------------------------------------------------------------

def _cvm():
    """The paper's community label for v:  c^v_m  (superscript v, subscript m)."""
    return msubsup(mr("c"), mr("m"), mr("v"))


def shared_lines_short():
    return [
        (0, kw("for ") + M(mr("c = 1..C,  c′ = 1..C,  z = 1..Z")) + kw(" do"),
         I + ":205-211"),
        (1, M(mfunc("P", mr("c | z, c′")) + mr(" = ")
              + msub(mr("η"), mr("c′c")) + mr(" · ") + msub(mr("θ"), mr("c,z")))
         + tx("      ▷ Eq (10)"), I + ":205-211"),
        (0, kw("end for"), ""),
        (0, kw("for ") + M(mdel(mr("u, v")) + mr(" ∈ ℰ")) + kw(" do"),
         I + ":261"),
        (1, M(mfunc("P", mr("v | z, u")) + mr(" = ")
              + SUM("c, c′", msub(mr("π"), mr("vc")) + mr(" · ")
                    + msub(mr("π"), mr("uc′")) + mr(" · ")
                    + mfunc("P", mr("c | z, c′"))))
         + tx("      ▷ Eq (11)"), I + ":188-192"),
        (1, M(mfunc("pp", mr("u, v")) + mr(" = ") + mfunc("P", mr("v | i, u"))
              + mr(" = ") + SUM("z = 1..Z", mfunc("P", mr("z | i")) + mr(" · ")
                                + mfunc("P", mr("v | z, u"))))
         + tx("      ▷ Eq (12)"), I + ":224-252"),
        (0, kw("end for"), ""),
        (0, kw("for ") + M(mr("v = 1..U")) + kw(" do"), C + ":178"),
        (1, M(_cvm() + mr(" ← ") + ARGMAX("c", msub(mr("π"), mr("v,c"))))
         + tx("      ▷ Eq (19),  hoà ⇒ c nhỏ nhất"), C + ":183-190"),
        (0, kw("end for"), ""),
        (0, M(msub(mr("c"), mr("m")) + mr(" = ")
              + brace(mr("v ∈ 𝒰 : ") + _cvm() + mr(" = m")) + mr(",   m = 1..C")),
         C + ":413-417"),
    ]


def ctim_lines_short():
    return [
        (0, kw("thực hiện ") + ref("Thuật toán 1")
         + tx("      ▷ pp(u,v),  ") + M(_cvm()) + tx(",  ") + M(msub(mr("c"), mr("m"))),
         C + ":405-417"),
        (0, M(msub(mr("𝒢"), mr("m")) + mr(" = ")
              + mdel(msub(mr("c"), mr("m")) + mr(", ") + msub(mr("ℰ"), mr("m")))
              + mr(",   ") + msub(mr("ℰ"), mr("m")) + mr(" = ")
              + brace(mdel(mr("u, v")) + mr(" ∈ ℰ : u ∈ ")
                      + msub(mr("c"), mr("m")) + mr(" ∧ v ∈ ")
                      + msub(mr("c"), mr("m")))), C + ":437"),
        (0, M(msub(mr("I"), mr("m")) + mdel(mr("·")) + mr(" = Eq (18) trên ")
              + msub(mr("𝒢"), mr("m"))) + tx("      ▷ đồ thị con cảm sinh"),
         C + ":437 → " + I + ":300-319"),
        (0, M(mr("𝒮 = ") + msub(mr("𝒮"), mr("1")) + mr(" = ")
              + msub(mr("𝒮"), mr("2")) + mr(" = … = ") + msub(mr("𝒮"), mr("C"))
              + mr(" = ∅")), C + ":422-423"),
        (0, kw("for ") + M(mr("k = 1..K")) + kw(" do  ")
         + M(mr("I") + brack(mr("0, k")) + mr(" = 0;   s") + brack(mr("0, k"))
             + mr(" = 0")) + kw("  end for"), C + ":443-446"),
        (0, kw("for ") + M(mr("m = 1..C")) + kw(" do  ")
         + M(mr("I") + brack(mr("m, 0")) + mr(" = 0")) + kw("  end for"),
         C + ":447-449"),
        (0, kw("for ") + M(mr("k = 1..K")) + kw(" do"), C + ":456"),
        (1, kw("for ") + M(mr("m = 1..C")) + kw(" do"), C + ":458"),
        (2, M(mr("Δ") + msub(mr("I"), mr("m")) + mr(" = ")
              + MAX("", mdel(msub(mr("I"), mr("m"))
                             + mdel(msub(mr("𝒮"), mr("m")) + mr(" ∪ u"))
                             + mr(" − ") + msub(mr("I"), mr("m"))
                             + mdel(msub(mr("𝒮"), mr("m")))))
              + mr(",   u ∈ ") + msub(mr("c"), mr("m")) + mr(" ∖ ")
              + msub(mr("𝒮"), mr("m")))
         + tx("      ▷ hoà ⇒ u nhỏ nhất;  ")
         + M(msub(mr("c"), mr("m")) + mr(" ∖ ") + msub(mr("𝒮"), mr("m"))
             + mr(" = ∅"))
         + tx(" ⇒ ") + M(mr("Δ") + msub(mr("I"), mr("m")) + mr(" = 0")),
         C + ":459-466 → :258-271"),
        (2, M(mr("I") + brack(mr("m, k")) + mr(" = ")
              + MAX("", mdel(mr("I") + brack(mr("m−1, k")) + mr(",  I")
                             + brack(mr("C, k−1")) + mr(" + Δ")
                             + msub(mr("I"), mr("m"))))), C + ":471-474"),
        (2, kw("if ") + M(mr("I") + brack(mr("C, k−1")) + mr(" + Δ")
                          + msub(mr("I"), mr("m")) + mr(" ≥ I")
                          + brack(mr("m−1, k"))) + kw(" then"), C + ":479"),
        (3, M(mr("s") + brack(mr("m, k")) + mr(" = m")), C + ":480-481"),
        (2, kw("else"), ""),
        (3, M(mr("s") + brack(mr("m, k")) + mr(" = s") + brack(mr("m−1, k"))),
         C + ":482-483"),
        (2, kw("end if"), ""),
        (1, kw("end for"), ""),
        (1, M(mr("j = s") + brack(mr("C, k"))), C + ":487"),
        (1, M(msub(mr("u"), mr("k")) + mr(" ← ")
              + ARGMAX(mr("u ∈ ") + msub(mr("c"), mr("j")) + mr(" ∖ ")
                       + msub(mr("𝒮"), mr("j")),
                       mdel(msub(mr("I"), mr("j"))
                            + mdel(msub(mr("𝒮"), mr("j")) + mr(" ∪ u"))
                            + mr(" − ") + msub(mr("I"), mr("j"))
                            + mdel(msub(mr("𝒮"), mr("j"))))))
         + tx("      ▷ hoà ⇒ u nhỏ nhất;  ")
         + M(msub(mr("c"), mr("j")) + mr(" ∖ ") + msub(mr("𝒮"), mr("j"))
             + mr(" = ∅")) + tx(" ⇒ ")
         + M(mr("j = ") + ARGMAX(mr("m : ") + msub(mr("c"), mr("m")) + mr(" ∖ ")
                                 + msub(mr("𝒮"), mr("m")) + mr(" ≠ ∅"),
                                 mr("Δ") + msub(mr("I"), mr("m"))))
         + tx(" (hoà ⇒ m nhỏ nhất);  không tồn tại m ⇒ thoát vòng lặp k"),
         C + ":488-507"),
        (1, M(msub(mr("𝒮"), mr("j")) + mr(" = ") + msub(mr("𝒮"), mr("j"))
              + mr(" ∪ ") + msub(mr("u"), mr("k")) + mr(";    𝒮 = 𝒮 ∪ ")
              + msub(mr("u"), mr("k")))
         + tx("      ▷ nối vào cuối: 𝒮 giữ thứ tự chọn"),
         C + ":302-303, 514-515"),
        (0, kw("end for"), ""),
        (0, kw("return ") + M(mr("𝒮")), C + ":544"),
    ]


def ctimg_lines_short():
    return [
        (0, kw("thực hiện ") + ref("Thuật toán 1")
         + tx("      ▷ pp(u,v),  ") + M(_cvm()) + tx(",  ") + M(msub(mr("c"), mr("m"))),
         G + ":375-388"),
        (0, M(IF(mr("·")) + mr(" = Eq (18) trên 𝒢"))
         + tx("      ▷ MỘT hàm mục tiêu; không có ") + M(msub(mr("𝒢"), mr("m"))),
         G + ":224"),
        (0, run("Pha 1:  đường cong lợi ích của từng cộng đồng",
                bold=True, size=9, color="1F4E79"), ""),
        (0, kw("for ") + M(mr("m = 1..C")) + kw(" do"), G + ":235-238"),
        (1, M(msubsup(mr("𝒮"), mr("0"), mr("m")) + mr(" = ∅;   ")
              + msub(mr("I"), mr("m")) + mdel(mr("0")) + mr(" = 0;   ")
              + msub(up("cap"), mr("m")) + mr(" = ")
              + MIN("", mdel(mr("K, |") + msub(mr("c"), mr("m")) + mr("|")))),
         G + ":240; " + E + ":85-86"),
        (1, kw("for ") + M(mr("j = 1..") + msub(up("cap"), mr("m"))) + kw(" do"),
         G + ":245"),
        (2, M(msubsup(mr("u"), mr("j"), mr("m")) + mr(" ← ")
              + ARGMAX("", mdel(IF(msubsup(mr("𝒮"), mr("j−1"), mr("m"))
                                      + mr(" ∪ u"))
                                + mr(" − ") + IF(msubsup(mr("𝒮"), mr("j−1"),
                                                                mr("m")))))
              + mr(",   u ∈ ") + msub(mr("c"), mr("m")) + mr(" ∖ ")
              + msubsup(mr("𝒮"), mr("j−1"), mr("m")))
         + tx("      ▷ hoà ⇒ u nhỏ nhất"), G + ":239-240"),
        (2, M(msubsup(mr("𝒮"), mr("j"), mr("m")) + mr(" = ")
              + msubsup(mr("𝒮"), mr("j−1"), mr("m")) + mr(" ∪ ")
              + msubsup(mr("u"), mr("j"), mr("m")) + mr(";    ")
              + msub(mr("I"), mr("m")) + mdel(mr("j")) + mr(" = ")
              + IF(msubsup(mr("𝒮"), mr("j"), mr("m")))), G + ":246-249"),
        (1, kw("end for"), ""),
        (0, kw("end for"), ""),
        (0, run("Pha 2:  phân bổ ngân sách", bold=True, size=9, color="1F4E79"), ""),
        (0, M(mr("I") + brack(mr("0, k")) + mr(" = 0,   k = 0..K")),
         E + ":346-347"),
        (0, kw("for ") + M(mr("m = 1..C")) + kw(" do"), E + ":349"),
        (1, kw("for ") + M(mr("k = 0..K")) + kw(" do"), E + ":353"),
        (2, M(mr("I") + brack(mr("m, k")) + mr(" = ")
              + MAX("", mdel(mr("I") + brack(mr("m−1, k−j")) + mr(" + ")
                             + msub(mr("I"), mr("m")) + mdel(mr("j"))))
              + mr(",   0 ≤ j ≤ ")
              + MIN("", mdel(msub(up("cap"), mr("m")) + mr(", k")))),
         E + ":354-365"),
        (2, M(mr("t") + brack(mr("m, k")) + mr(" = j"))
         + tx(" nhỏ nhất đạt cực đại"), E + ":362-366"),
        (1, kw("end for"), ""),
        (0, kw("end for"), ""),
        (0, M(mr("k = K;   ") + msub(mr("𝒮"), up("cat")) + mr(" = ∅;   dp = I")
              + brack(mr("C, K"))), E + ":368-370, 378"),
        (0, kw("for ") + M(mr("m = C..1")) + kw(" do  ")
         + M(msub(mr("a"), mr("m")) + mr(" = t") + brack(mr("m, k")) + mr(";   ")
             + msub(mr("𝒮"), up("cat")) + mr(" = ") + msub(mr("𝒮"), up("cat"))
             + mr(" ∪ ") + msubsup(mr("𝒮"), msub(mr("a"), mr("m")), mr("m"))
             + mr(";   k = k − ") + msub(mr("a"), mr("m"))) + kw("  end for"),
         E + ":371-376"),
        (0, run("Pha 3:  hiện thực hoá dưới ràng buộc hạn ngạch",
                bold=True, size=9, color="1F4E79"), ""),
        (0, M(msub(mr("𝒮"), mr("r")) + mr(" = ∅")), G + ":150-156"),
        (0, kw("while ") + M(mr("|") + msub(mr("𝒮"), mr("r")) + mr("| < ")
                             + SUM("m", msub(mr("a"), mr("m")))) + kw(" do"),
         G + ":158"),
        # The quota must sit INSIDE the arg max's domain.  Written as its own
        # line it does not mention u, so it constrains nothing -- it is already
        # a loop invariant, true at the top of every iteration.  The strict "<"
        # also subsumes the old "a_m > 0" index, since a_m > |S_r n c_m| >= 0.
        (1, M(mr("u ← ")
              + ARGMAX("", mdel(IF(msub(mr("𝒮"), mr("r")) + mr(" ∪ u"))
                                + mr(" − ") + IF(msub(mr("𝒮"), mr("r")))))
              + mr(",   u ∈ ")
              + mnary("⋃",
                      mr("m : |") + msub(mr("𝒮"), mr("r")) + mr(" ∩ ")
                      + msub(mr("c"), mr("m")) + mr("| < ")
                      + msub(mr("a"), mr("m")),
                      "", msub(mr("c"), mr("m")))
              + mr(" ∖ ") + msub(mr("𝒮"), mr("r")))
         + tx("      ▷ hạn ngạch nằm trong miền;  hoà ⇒ u nhỏ nhất"),
         G + ":159-176"),
        (1, kw("if ") + M(IF(msub(mr("𝒮"), mr("r")) + mr(" ∪ u"))
                          + mr(" − ") + IF(msub(mr("𝒮"), mr("r")))
                          + mr(" ≤ 0")) + kw(" và ")
         + M(msub(mr("𝒮"), mr("r")) + mr(" ≠ ∅")) + kw(" then ") + tx("thoát")
         + kw(" end if"),
         G + ":178-179"),
        (1, M(msub(mr("𝒮"), mr("r")) + mr(" = ") + msub(mr("𝒮"), mr("r"))
              + mr(" ∪ u")), G + ":181-186"),
        (0, kw("end while"), ""),
        (0, run("Pha 4:  chọn kết quả và đo khoảng cách độc lập",
                bold=True, size=9, color="1F4E79"), ""),
        (0, kw("if ") + M(IF(msub(mr("𝒮"), up("cat"))) + mr(" > ")
                          + IF(msub(mr("𝒮"), mr("r")))) + kw(" then ")
         + M(mr("𝒮 = ") + msub(mr("𝒮"), up("cat"))) + kw(" else ")
         + M(mr("𝒮 = ") + msub(mr("𝒮"), mr("r"))) + kw(" end if"),
         G + ":428-436"),
        (0, M(mr("Γ = dp − ") + IF(mr("𝒮"))), G + ":438"),
        (0, kw("return ") + M(mr("𝒮, Γ")), G + ":460-463"),
    ]


# ---------------------------------------------------------------------------

def build(clean=False):
    """clean=True drops the ▷ reading notes, the source column, and every
    section that is commentary rather than algorithm -- the paste-into-the-paper
    build.  Both builds share one definition of every pseudocode line, so they
    cannot drift apart."""
    _CLEAN[0] = clean
    b = []
    A = b.append
    # commentary: emitted by the annotated build only
    AN = (lambda _x: None) if clean else b.append

    A(para(run("Thuật toán CTIM và CTIM-G", bold=True), style="Title",
           spacing_after=60))
    A(para(run("Viết theo khuôn algorithm của bài báo, công thức đầy đủ. Phần "
               "dùng chung được tách thành Thuật toán 1 và được hai thuật toán "
               "còn lại gọi lại ở dòng 1."
               if clean else
               "Viết theo khuôn algorithm của bài báo, công thức đầy đủ. Phần "
               "dùng chung được tách thành Thuật toán 1 và được hai thuật toán "
               "còn lại gọi lại ở dòng 1. Cột phải ghi tệp:dòng của mã nguồn "
               "hiện thực từng dòng.", italic=True, color="5A6673")))

    # The two builds use different symbol sets: the concise algorithms never
    # mention MIP/MIIA/MIOA/ap/sigma (they only ever cite "Eq (18)"), while the
    # detailed ones do.  A table that lists symbols the reader will not meet is
    # as much of a defect as one that omits symbols they will.
    rows = [
        ["𝒢 = (𝒰, ℰ)", "đồ thị xã hội có hướng; U = |𝒰|"],
        ["C, Z", "số cộng đồng, số chủ đề"],
        ["π, η, θ", "phân phối người dùng–cộng đồng, cộng đồng–cộng đồng, cộng đồng–chủ đề"],
        ["P(z | i)", "độ liên quan chủ đề của sản phẩm i"],
        ["pp(u,v)", "xác suất lan truyền trên cạnh (u,v) cho sản phẩm i, Eq (12)"],
        ["c^v_m", "cộng đồng của v (Eq 19);  c_m = { v ∈ 𝒰 : c^v_m = m }"],
    ]
    if not clean:
        rows += [
            ["MIP(u,v)", "đường đi ảnh hưởng cực đại, Eq (14)"],
            ["MIIA(v,h), MIOA(u,h)", "cây ảnh hưởng vào / ra, Eq (15), Eq (16)"],
            ["ap(v | 𝒮)", "xác suất kích hoạt của v, Eq (17)"],
        ]
    rows += [
        ["I(𝒮)", "độ lan truyền Eq (18) trên TOÀN đồ thị 𝒢"],
        ["𝒢_m = (c_m, ℰ_m)", "đồ thị con CẢM SINH trên c_m: cạnh (u,v) tồn tại "
                              "khi CẢ HAI đầu mút ∈ c_m — chỉ Thuật toán 2"],
        ["I_m(𝒮)", "Eq (18) trên 𝒢_m — chỉ Thuật toán 2 dùng"],
        ["𝒮, 𝒮_m", "tập hạt giống toàn cục (giữ thứ tự chọn); tập hạt giống của cộng đồng m"],
        ["ΔI_m", "lợi ích biên lớn nhất mà một ứng viên trong c_m ∖ 𝒮_m mang lại "
                 "cho 𝒮_m — chỉ Thuật toán 2"],
        ["I[m,k], s[m,k]", "bảng quy hoạch động và bảng con trỏ cộng đồng — ngoặc "
                           "VUÔNG, khác hẳn hàm I(·) — chỉ Thuật toán 2"],
    ]
    if not clean:
        rows += [["σ(u)", "I({u}), độ lan truyền đơn lẻ — chỉ Thuật toán 3 dùng"]]
    rows += [
        ["I_m(j), cap_m", "đường cong lợi ích của cộng đồng m và trần "
                          "cap_m = min(K, |c_m|) — chỉ Thuật toán 3"],
        ["u^m_j", "hạt giống thứ j trên đường cong của cộng đồng m; 𝒮^m_j là "
                  "tiền tố j phần tử — chỉ Thuật toán 3"],
        ["t[m,k]", "số hạt giống cấp cho cộng đồng m tại ô (m,k); KHÁC nghĩa với "
                   "s[m,k] của Thuật toán 2 — chỉ Thuật toán 3"],
        ["a_m", "ngân sách (hạn ngạch) cộng đồng m nhận được — chỉ Thuật toán 3"],
        ["dp", "giá trị quy hoạch động dự báo, dp = I[C,K] = Σ_m I_m(a_m) — "
               "chỉ Thuật toán 3"],
        ["𝒮_cat, 𝒮_r", "nghiệm ghép từ các đường cong; nghiệm hiện thực hoá dưới "
                        "hạn ngạch — chỉ Thuật toán 3"],
        ["Γ", "khoảng cách độc lập, Γ = dp − I(𝒮) — chỉ Thuật toán 3"],
    ]
    A(heading("Ký hiệu", 1))
    A(table(["Ký hiệu", "Ý nghĩa"], rows, [2200, 6800], ["left", "left"]))

    # ---- Algorithm 1 -----------------------------------------------------
    A(heading("Thuật toán 1 — Phần dùng chung của CTIM và CTIM-G", 1))
    A(para("Hai phương pháp chạy giống hệt nhau cho tới đúng ba đầu ra dưới đây, "
           "rồi mới rẽ nhánh. Tách riêng ra để hai thuật toán sau không phải chép "
           "lại, và để thấy rõ phần khác biệt bắt đầu từ đâu."))
    A(algo(
        "Algorithm 1  Shared preamble — Eq (12) edge weights and Eq (19) communities",
        [("Input:", "Đồ thị có hướng 𝒢 = (𝒰, ℰ); mô hình đã học (π, η, θ, "
                    "P(z | i)); sản phẩm i; số cộng đồng C",
          I + ":180-186; " + C + ":170-177"),
         ("Output:", "pp(u,v) cho mọi (u,v) ∈ ℰ;  c^v_m cho mọi v ∈ 𝒰;  "
                     "c_m,  m = 1..C",
          C + ":405-417")],
        shared_lines_short() if clean else shared_lines()))
    AN(caption("Trong ba đầu ra, chỉ Eq (19) là dùng chung theo nghĩa BYTE-FOR-"
              "BYTE — cùng một hàm detect_communities, ctim_global.py:98 import "
              "thẳng từ ctim.ctim, gọi ở ctim.py:411 và ctim_global.py:388 với "
              "cùng đối số. Eq (12) dùng chung ở tầng tính toán nhưng khác ở "
              "tầng giao diện: CTIM luôn tự dựng pp, CTIM-G cho phép tiêm sẵn. "
              "{ c_m } cho cùng kết quả nhưng lệch chỉ số 1 giữa hai bản cài đặt."))

    # ---- Algorithm 2 -----------------------------------------------------
    A(heading("Thuật toán 2 — CTIM", 1))
    A(algo(
        "Algorithm 2  Community-based topic-aware influence maximization (CTIM)",
        [("Input:", "Đầu ra của Thuật toán 1; số hạt giống K; ngưỡng MIA h "
                    "(bài báo đặt h = 0.1); C = |η|", C + ":329-331, 375"),
         ("Output:", "Tập hạt giống 𝒮", C + ":544")],
        ctim_lines_short() if clean else ctim_lines()))
    AN(caption("Dòng 5 dựng ĐỒ THỊ CON CẢM SINH — đây là điểm rẽ nhánh so với "
              "Thuật toán 3 dòng 2. Dòng 16–20 là quy hoạch động của bài báo, "
              "phát đúng một hạt giống mỗi vòng k. Dòng 23–26 là nhánh dự phòng "
              "khi bảng trỏ vào cộng đồng rỗng — một khả năng bài báo không nêu. "
              "Dòng 27–34 là phép sửa chữa gia tăng bảng IncInf, tương đương "
              "Algorithm 2 dòng in 44 nhưng tránh tính lại từ đầu."))

    # ---- Algorithm 3 -----------------------------------------------------
    A(heading("Thuật toán 3 — CTIM-G", 1))
    A(algo(
        "Algorithm 3  CTIM with a single global objective (CTIM-G)",
        [("Input:", "Đầu ra của Thuật toán 1; số hạt giống K; ngưỡng MIA h", G + ":302-305"),
         ("Output:", "Tập hạt giống 𝒮;  khoảng cách độc lập Γ = dp − I(𝒮)",
          G + ":460-463")],
        ctimg_lines_short() if clean else ctimg_lines()))
    AN(caption("Bước tinh chỉnh 1-hoán vị (local search) KHÔNG có mặt ở đây: "
              "repair = False là mặc định tại ctim_global.py:303, nó được đăng "
              "ký thành một phương thức riêng tên CTIM-G+repair, và không lần "
              "chạy nào trong sáu nhật ký dùng nó. Đưa nó vào sẽ phá ràng buộc "
              "hạn ngạch — nó không biết cộng đồng là gì — nên không còn phân "
              "biệt được “sửa dòng 34 tại gốc” với “vá đầu ra sau khi đã sai”."))

    # ---- reading notes ---------------------------------------------------
    AN(heading("Năm chỗ dễ đọc sai", 1))
    AN(bullet(run("Thuật toán 2 dòng 5 dựng ")
             + run("𝒢_m", bold=True)
             + run(" — đồ thị con cảm sinh. Một cạnh chỉ tồn tại khi CẢ HAI đầu "
                   "mút nằm trong c_m, nên mọi cạnh bắc cầu giữa các cộng đồng "
                   "biến mất khỏi I_m trước khi bất kỳ ứng viên nào được chấm "
                   "điểm. Thuật toán 3 dòng 2 không có hạn chế đó. Đó là toàn bộ "
                   "khác biệt về mô hình; mọi thứ còn lại là hệ quả.")))
    AN(bullet("Thuật toán 2 dòng 16 đo lợi ích đối với 𝒮_m — hạt giống của RIÊNG "
             "cộng đồng đó. Thuật toán 3 dòng 8 cũng vậy ở pha 1 (cố ý, để các "
             "đường cong độc lập), nhưng dòng 31 ở pha 3 đo đối với 𝒮_r đầy đủ "
             "trên mọi cộng đồng. Vì thế phần ảnh hưởng chồng lấn được đếm hai "
             "lần ở dp và đúng một lần ở value."))
    AN(bullet("Đệ quy ở Thuật toán 2 dòng 18 khai triển thành I[C,k] = I[C,k−1] + "
             "max_m ΔI_m: tham lam trên cộng đồng, đúng một hạt giống mỗi vòng, "
             "không có vòng lặp j và không có số hạng cap. Quy hoạch động ở Thuật "
             "toán 3 dòng 18 có cả hai, nên cấp được nhiều hạt giống cho một "
             "cộng đồng trong một quyết định. Hai thứ này đều được gọi là “quy "
             "hoạch động” nhưng KHÔNG phải một."))
    AN(bullet("Cả hai thuật toán đều có thể trả về ít hơn K hạt giống, vì hai lý "
             "do khác nhau. Thuật toán 2: mọi cộng đồng cạn ứng viên (dòng 26). "
             "Thuật toán 3: dòng 19 phá thế hoà theo hướng j nhỏ hơn nên Σ a_m "
             "có thể < K, và dòng 32 còn dừng sớm khi lợi ích về 0."))
    AN(bullet("Γ ở dòng 41 là CHẶN TRÊN của mức vi phạm giả thiết độc lập, không "
             "phải bằng đúng. Nó còn hấp thụ phần chênh giữa các mảnh mà đường "
             "cong đã chọn và các mảnh mà pha 3 thật sự chọn. Γ = 0 khi và chỉ "
             "khi pha 3 chọn đúng những mảnh đó."))

    # ---- verification ----------------------------------------------------
    AN(heading("Cách kiểm chứng", 1))
    AN(para(run("Mỗi dòng có một trích dẫn tệp:dòng ở cột phải. Mọi đường dẫn đều "
               "nằm trong thư mục gói ")
           + run("ctim/", mono=True)
           + run(" — tiền tố đó được lược đi cho gọn cột, nên ")
           + run("ctim.py:437", mono=True)
           + run(" nghĩa là ")
           + run("ctim/ctim.py", mono=True)
           + run(" dòng 437. Mở đúng dòng đó là thấy phép toán tương ứng. Hai "
                 "dòng quyết định toàn bộ khác biệt giữa hai phương pháp:")))
    AN(para(run("sed -n '437p' ctim/ctim.py         → MIA(..., nodes=members)",
               mono=True, size=9), indent=340, spacing_after=0))
    AN(para(run("sed -n '224p' ctim/ctim_global.py  → MIA(...)  không có nodes=",
               mono=True, size=9), indent=340))
    AN(caption("Chú ý: các chú thích trong ctim/ctim.py ghi “line 34”, “line 35” "
              "là số dòng IN của Algorithm 2 như chép ở ctim/ctim.py:9-55, KHÔNG "
              "phải số dòng tệp. Số dòng tệp trong tài liệu này ứng với trạng "
              "thái mã nguồn tại thời điểm sinh tài liệu."))
    return "".join(b)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("results", "algorithms.docx"),
                   help="annotated build, with the source column, for checking")
    p.add_argument("--out-clean", dest="out_clean",
                   default=os.path.join("results", "algorithms_clean.docx"),
                   help="clean build, algorithm environment only, for the paper")
    p.add_argument("--only", choices=("annotated", "clean"),
                   help="build only one of the two")
    args = p.parse_args(argv)

    n  = (len(shared_lines()), len(ctim_lines()), len(ctimg_lines()))
    ns = (len(shared_lines_short()), len(ctim_lines_short()),
          len(ctimg_lines_short()))
    if args.only != "clean":
        write_docx(args.out, build(clean=False))
        print("[write] %s   (%d + %d + %d lines, annotated + source column)"
              % ((args.out,) + n))
    if args.only != "annotated":
        write_docx(args.out_clean, build(clean=True))
        print("[write] %s   (%d + %d + %d lines, clean/concise)"
              % ((args.out_clean,) + ns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
