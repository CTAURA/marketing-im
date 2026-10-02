"""CTIM transcribed verbatim, and CTIM-G as a marked-up edit of it.

Two listings:

  Algorithm 2  CTIM, transcribed from the printed figure exactly.  No symbol
               appears here that is not in the paper: G, U, E, D, pi, theta,
               eta, P(z|i), K, S, C, Z, M, U, c^v_m, dI_m, I[m,k], s[m,k],
               I_m, c_m, c_j, S_j, u_k, j.
  Algorithm 3  CTIM-G, written as the SAME listing with every line that
               differs from CTIM set in red.  Lines 1-25 are byte-identical
               and stay black.

One typesetting note.  Printed line 36 reads ``if I([C, k-1] + dI_m >= ...``
with a stray opening parenthesis after ``I``.  Lines 35 and 36 otherwise agree
that the reference is the table entry ``I[C, k-1]``, so the paren is read as an
artefact and not reproduced.  This also settles a disagreement inside this
repo: ctim/ctim.py:45 transcribes printed line 35 as ``I[m,k-1]``, while
ctim/ctim.py:344 states the paper prints ``I[C,k-1]`` on both lines and the
running code at ctim/ctim.py:471 uses ``I[C,k-1]``.  The figure agrees with the
code, not with that one transcription line.

Usage
-----
    python scripts/export_algo_diff_docx.py --out results/algorithms_diff.docx
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_docx import (caption, heading, mdel, minline, mlimlow,   # noqa: E402
                         mnary, mr, msub, msubsup, msup, para, run,
                         table, write_docx)

RED = "C00000"

# ---------------------------------------------------------------------------
# math shortcuts, kept to exactly the constructs the printed figure uses
# ---------------------------------------------------------------------------


def M(*parts):
    return minline("".join(parts))


def sub(base, s):
    return msub(mr(base), mr(s))


def SUM(idx, body):
    return mnary("∑", mr(idx), "", body)


def SUMTO(lo, hi, body):
    return mnary("∑", mr(lo), mr(hi), body)


def _omml_cond(x):
    """Marks a limit that is already OMML, so it is not wrapped in mr()."""
    return x


def _lim(name, cond, body):
    if not cond:
        return mr(name, plain=True) + body
    lim = cond if str(cond).startswith("<m:") else mr(cond)
    return mlimlow(mr(name, plain=True), lim) + body


def ARGMAX(cond, body):
    return _lim("arg max", cond, body)


def MAX(cond, body):
    return _lim("max", cond, body)


def MIN(body):
    return mr("min", plain=True) + body


def redden(omml):
    """Set every math run inside an OMML fragment in red.

    OOXML puts formatting for a math run in ``w:rPr`` INSIDE ``m:r``, after
    ``m:rPr`` when that is present.  Order matters, so the two cases are
    rewritten separately rather than with one pattern.
    """
    c = '<w:rPr><w:color w:val="%s"/></w:rPr>' % RED
    # Both branches must be idempotent: r() re-reddens a body that RM() already
    # reddened, and OOXML allows at most ONE w:rPr per m:r.
    out = re.sub(r'(<m:r><m:rPr><m:sty m:val="p"/></m:rPr>)(?!<w:rPr>)',
                 lambda mo: mo.group(1) + c, omml)
    out = re.sub(r'<m:r>(?!<m:rPr>)(?!<w:rPr>)', '<m:r>' + c, out)
    return out


def kw(s, red=False):
    return run(s, bold=True, size=9, color=(RED if red else None))


def tx(s, red=False):
    return run(s, size=9, color=(RED if red else None))


# ---------------------------------------------------------------------------
# renderer: the paper's algorithm environment, two columns
# ---------------------------------------------------------------------------

W_NUM, W_BODY = 520, 8480

_RULE = ('<w:tcBorders><w:bottom w:val="single" w:sz="10" w:color="000000"/>'
         "</w:tcBorders>")


def _tc(content, width, borders="", indent=0, align=None, span=1):
    pr = ['<w:tcW w:w="%d" w:type="dxa"/>' % width]
    if span > 1:
        pr.append('<w:gridSpan w:val="%d"/>' % span)
    pr += [borders, '<w:vAlign w:val="top"/>']
    ppr = ['<w:spacing w:before="12" w:after="12" w:line="240" w:lineRule="auto"/>']
    if indent:
        ppr.append('<w:ind w:left="%d"/>' % indent)
    if align:
        ppr.append('<w:jc w:val="%s"/>' % align)
    p = "<w:p><w:pPr>%s</w:pPr>%s</w:p>" % ("".join(ppr), content)
    return "<w:tc><w:tcPr>%s</w:tcPr>%s</w:tc>" % ("".join(pr), p)


def algo(title, io_rows, lines):
    """lines: list of (indent, body, changed).  changed only affects colour."""
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
           '<w:right w:w="50" w:type="dxa"/><w:top w:w="0" w:type="dxa"/>'
           '<w:bottom w:w="0" w:type="dxa"/></w:tblCellMar>',
           "</w:tblPr><w:tblGrid>",
           '<w:gridCol w:w="%d"/><w:gridCol w:w="%d"/>' % (W_NUM, W_BODY),
           "</w:tblGrid>"]
    out.append("<w:tr>")
    out.append(_tc(run(title, bold=True, size=11), W_NUM + W_BODY, _RULE, span=2))
    out.append("</w:tr>")
    for i, (label, body) in enumerate(io_rows):
        b = _RULE if i == len(io_rows) - 1 else ""
        out.append("<w:tr>")
        out.append(_tc(run(label + " ", bold=True, size=9) + body,
                       W_NUM + W_BODY, b, span=2))
        out.append("</w:tr>")
    n = 0
    for indent, body, changed in lines:
        n += 1
        out.append("<w:tr>")
        out.append(_tc(run("%d:" % n, size=8,
                           color=(RED if changed else "55606B")),
                       W_NUM, align="right"))
        out.append(_tc(body, W_BODY, indent=indent * 190))
        out.append("</w:tr>")
    out.append("</w:tbl>")
    out.append(para("", spacing_after=200))
    return "".join(out)


# ---------------------------------------------------------------------------
# Algorithm 2 -- CTIM, verbatim
# ---------------------------------------------------------------------------

def ctim_io():
    return [
        ("Input:", run("Directed graph ", size=9)
         + M(mr("𝒢 = ") + mdel(mr("𝒰, ℰ")))
         + run(", set of potential-influence logs ", size=9)
         + M(mr("𝒟, π, θ, η"))
         + run(", item-topic relevance ", size=9)
         + M(mr("P") + mdel(mr("z | i")))
         + run(", size of seed nodes ", size=9) + M(mr("K"))),
        ("Output:", run("Seed nodes set ", size=9) + M(mr("𝒮")) + run(".", size=9)),
    ]


def ctim_lines():
    L = []
    a = lambda ind, body: L.append((ind, body, False))          # noqa: E731

    a(0, kw("for ") + M(mr("c = 1 … C")) + kw(" do"))
    a(1, kw("for ") + M(mr("c′ = 1 … C")) + kw(" do"))
    a(2, kw("for ") + M(mr("z = 1 … Z")) + kw(" do"))
    a(3, M(mr("P") + mdel(mr("c | z, c′")) + mr(" = ")
           + sub("η", "c′c") + mr(" · ") + sub("θ", "c,z") + mr(";")))
    a(2, kw("end for"))
    a(1, kw("end for"))
    a(0, kw("end for"))
    a(0, kw("for ") + M(mr("d = ") + mdel(mr("u, v, i")) + mr(" ∈ 𝒟")) + kw(" do"))
    a(1, kw("for ") + M(mr("c = 1 … C")) + kw(" do"))
    a(2, kw("for ") + M(mr("c′ = 1 … C")) + kw(" do"))
    a(3, M(mr("P") + mdel(mr("v | z, u")) + mr(" = ")
           + SUM("c,c′", sub("π", "vc") + mr(" · ") + sub("π", "uc′")
                 + mr(" · ") + mr("P") + mdel(mr("c | z, c′"))) + mr(";")))
    a(2, kw("end for"))
    a(1, kw("end for"))
    a(0, kw("end for"))
    a(0, kw("for ") + M(mr("i = 1 … M")) + kw(" do"))
    a(1, kw("for ") + M(mr("d = ") + mdel(mr("u, v, i")) + mr(" ∈ 𝒟")) + kw(" do"))
    a(2, kw("for ") + M(mr("z = 1 … Z")) + kw(" do"))
    a(3, M(mr("P") + mdel(mr("v | i, u")) + mr(" = ")
           + SUMTO("z=1", "Z", mr("P") + mdel(mr("z | i")) + mr(" · ")
                   + mr("P") + mdel(mr("v | z, u"))) + mr(";")))
    a(2, kw("end for"))
    a(1, kw("end for"))
    a(0, kw("end for"))
    a(0, kw("for ") + M(mr("v = 1 … U")) + kw(" do"))
    a(1, M(msubsup(mr("c"), mr("m"), mr("v")) + mr(" ← ")
           + ARGMAX("c", sub("π", "v,c")) + mr(";")))
    a(0, kw("end for"))
    a(0, M(mr("𝒮 = ") + sub("𝒮", "1") + mr(" = ") + sub("𝒮", "2")
           + mr(" = ... = ") + sub("𝒮", "C") + mr(" = ∅")))
    a(0, kw("for ") + M(mr("k = 1 … K")) + kw(" do"))
    a(1, M(mr("I") + mdel(mr("0, k"), "[", "]") + mr(" = 0;   s")
           + mdel(mr("0, k"), "[", "]") + mr(" = 0;")))
    a(0, kw("end for"))
    a(0, kw("for ") + M(mr("m = 1 … C")) + kw(" do"))
    a(1, M(mr("I") + mdel(mr("m, 0"), "[", "]") + mr(" = 0;")))
    a(0, kw("end for"))
    a(0, kw("for ") + M(mr("k = 1 … K")) + kw(" do"))
    a(1, kw("for ") + M(mr("m = 1 … C")) + kw(" do"))
    a(2, M(mr("Δ") + sub("I", "m") + mr(" = ")
           + MAX("", mdel(sub("I", "m") + mdel(mr("𝒮 ∪ u")) + mr(" − ")
                          + sub("I", "m") + mdel(mr("𝒮"))))
           + mr(", u ∈ ") + sub("c", "m") + mr(";")))
    a(2, M(mr("I") + mdel(mr("m, k"), "[", "]") + mr(" = ")
           + MAX("", mdel(mr("I") + mdel(mr("m−1, k"), "[", "]") + mr(", I")
                          + mdel(mr("C, k−1"), "[", "]") + mr("+Δ")
                          + sub("I", "m"))) + mr(";")))
    a(2, kw("if ") + M(mr("I") + mdel(mr("C, k−1"), "[", "]") + mr(" + Δ")
                       + sub("I", "m") + mr(" ≥ I")
                       + mdel(mr("m−1, k"), "[", "]")) + kw(" then"))
    a(3, M(mr("s") + mdel(mr("m, k"), "[", "]") + mr(" = m;")))
    a(2, kw("else"))
    a(3, M(mr("s") + mdel(mr("m, k"), "[", "]") + mr(" = s")
           + mdel(mr("m−1, k"), "[", "]") + mr(";")))
    a(2, kw("end if"))
    a(1, kw("end for"))
    a(1, M(mr("j ← s") + mdel(mr("C, k"), "[", "]") + mr(";")))
    a(1, M(sub("u", "k") + mr(" ← ")
           + ARGMAX(_omml_cond(mr("u ∈ ") + sub("c", "j")),
                    mdel(mr("I") + mdel(sub("𝒮", "j") + mr(" ∪ u"))
                         + mr(" − ") + mr("I") + mdel(sub("𝒮", "j"))))
           + mr(";")))
    a(1, M(sub("𝒮", "j") + mr(" = ") + sub("𝒮", "j") + mr(" ∪ ") + sub("u", "k")
           + mr(";   𝒮 = 𝒮 ∪ ") + sub("u", "k")))
    a(0, kw("end for"))
    return L


# ---------------------------------------------------------------------------
# Algorithm 3 -- CTIM-G, the same listing with the edits in red
# ---------------------------------------------------------------------------

def ctimg_lines():
    L = ctim_lines()[:25]            # lines 1-25 are byte-identical
    a = lambda ind, body: L.append((ind, body, False))          # noqa: E731
    r = lambda ind, body: L.append((ind, redden(body)           # noqa: E731
                                    if body.startswith("<m:") else body, True))

    def RM(*parts):
        return redden(M(*parts))

    # -- new: build one benefit curve per community, on the GLOBAL objective
    r(0, kw("for ", True) + RM(mr("m = 1 … C")) + kw(" do", True))
    r(1, RM(sub("I", "m") + mdel(mr("0")) + mr(" = 0;   ")
            + msubsup(mr("𝒮"), mr("m"), mdel(mr("0"))) + mr(" = ∅;")))
    r(1, kw("for ", True) + RM(mr("j = 1 … ")
                               + MIN(mdel(mr("K, |") + sub("c", "m") + mr("|"))))
      + kw(" do", True))
    r(2, RM(mr("u ← ")
            + ARGMAX(_omml_cond(mr("u ∈ ") + sub("c", "m") + mr(" ∖ ")
                                + msubsup(mr("𝒮"), mr("m"), mdel(mr("j−1")))),
                     mdel(mr("I") + mdel(msubsup(mr("𝒮"), mr("m"), mdel(mr("j−1")))
                                         + mr(" ∪ u"))
                          + mr(" − ") + mr("I")
                          + mdel(msubsup(mr("𝒮"), mr("m"), mdel(mr("j−1"))))))
            + mr(";"))
      + tx("   ties broken by the smallest node index", True))
    r(2, RM(msubsup(mr("𝒮"), mr("m"), mdel(mr("j"))) + mr(" = ")
            + msubsup(mr("𝒮"), mr("m"), mdel(mr("j−1"))) + mr(" ∪ u;   ")
            + sub("I", "m") + mdel(mr("j")) + mr(" = I")
            + mdel(msubsup(mr("𝒮"), mr("m"), mdel(mr("j")))) + mr(";")))
    r(1, kw("end for", True))
    r(0, kw("end for", True))

    # -- initialisation: s[0,k] is gone, the table is now indexed from k = 0
    r(0, kw("for ", True) + RM(mr("k = 0 … K")) + kw(" do", True))
    r(1, RM(mr("I") + mdel(mr("0, k"), "[", "]") + mr(" = 0;")))
    a(0, kw("end for"))
    a(0, kw("for ") + M(mr("m = 1 … C")) + kw(" do"))
    a(1, M(mr("I") + mdel(mr("m, 0"), "[", "]") + mr(" = 0;")))
    a(0, kw("end for"))

    # -- the DP: a j-loop replaces the one-seed-per-round recurrence
    r(0, kw("for ", True) + RM(mr("m = 1 … C")) + kw(" do", True))
    r(1, kw("for ", True) + RM(mr("k = 0 … K")) + kw(" do", True))
    r(2, RM(mr("I") + mdel(mr("m, k"), "[", "]") + mr(" = ")
            + MAX(_omml_cond(mr("0 ≤ j ≤ ")
                             + MIN(mdel(mr("k, |") + sub("c", "m") + mr("|")))),
                  mdel(mr("I") + mdel(mr("m−1, k−j"), "[", "]") + mr(" + ")
                       + sub("I", "m") + mdel(mr("j")))) + mr(";")))
    r(2, RM(mr("t") + mdel(mr("m, k"), "[", "]") + mr(" = "))
      + tx("smallest ", True) + RM(mr("j")) + tx(" attaining the maximum;", True))
    a(1, kw("end for"))
    a(0, kw("end for"))

    # -- backtrack: recover the per-community budget
    r(0, RM(mr("k = K;   ") + sub("𝒮", "cat") + mr(" = ∅;")))
    r(0, kw("for ", True) + RM(mr("m = C … 1")) + kw(" do", True))
    r(1, RM(sub("a", "m") + mr(" = t") + mdel(mr("m, k"), "[", "]") + mr(";   ")
            + sub("𝒮", "cat") + mr(" = ") + sub("𝒮", "cat") + mr(" ∪ ")
            + msubsup(mr("𝒮"), mr("m"), mdel(sub("a", "m")))
            + mr(";   k = k − ") + sub("a", "m") + mr(";")))
    r(0, kw("end for", True))

    # -- realisation: ONE greedy over all funded communities, quota-capped
    r(0, RM(mr("𝒮 = ∅;")))
    r(0, kw("while ", True) + RM(mr("|𝒮| < ") + SUM("m", sub("a", "m")))
      + kw(" do", True))
    r(1, RM(sub("u", "k") + mr(" ← ")
            + ARGMAX(_omml_cond(mr("u ∈ ")
                                + mnary("⋃", mr("m : |𝒮 ∩ ") + sub("c", "m")
                                        + mr("| < ") + sub("a", "m"), "",
                                        sub("c", "m"))
                                + mr(" ∖ 𝒮")),
                     mdel(mr("I") + mdel(mr("𝒮 ∪ u")) + mr(" − ")
                          + mr("I") + mdel(mr("𝒮")))) + mr(";"))
      + tx("   ties broken by the smallest node index", True))
    r(1, kw("if ", True) + RM(mr("I") + mdel(mr("𝒮 ∪ u")) + mr(" − ") + mr("I")
                              + mdel(mr("𝒮")) + mr(" ≤ 0")) + kw(" and ", True)
      + RM(mr("𝒮 ≠ ∅")) + kw(" then break end if", True))
    r(1, RM(mr("𝒮 = 𝒮 ∪ ") + sub("u", "k") + mr(";")))
    r(0, kw("end while", True))

    # -- keep the better of the two assemblies, then report the gap
    r(0, kw("if ", True) + RM(mr("I") + mdel(sub("𝒮", "cat")) + mr(" > I")
                              + mdel(mr("𝒮"))) + kw(" then ", True)
      + RM(mr("𝒮 = ") + sub("𝒮", "cat")) + kw(" end if", True))
    r(0, RM(mr("Γ = I") + mdel(mr("C, K"), "[", "]") + mr(" − I") + mdel(mr("𝒮"))
            + mr(";")))
    return L


def ctimg_io():
    return [
        ("Input:", run("Directed graph ", size=9)
         + M(mr("𝒢 = ") + mdel(mr("𝒰, ℰ")))
         + run(", set of potential-influence logs ", size=9)
         + M(mr("𝒟, π, θ, η"))
         + run(", item-topic relevance ", size=9)
         + M(mr("P") + mdel(mr("z | i")))
         + run(", size of seed nodes ", size=9) + M(mr("K"))),
        ("Output:", run("Seed nodes set ", size=9) + M(mr("𝒮"))
         + run(", ", size=9)
         + run("independence gap ", size=9, color=RED)
         + redden(M(mr("Γ"))) + run(".", size=9)),
    ]


# ---------------------------------------------------------------------------

def build():
    b = []
    A = b.append

    A(para(run("CTIM và CTIM-G: bản chép nguyên văn và bản sửa", bold=True),
           style="Title", spacing_after=60))
    A(para(run("Thuật toán 2 là bản chép nguyên văn từ hình in của bài báo, "
               "không thêm một ký hiệu nào. Thuật toán 3 là chính bản đó, sửa "
               "lại thành CTIM-G, với mọi dòng khác biệt được ", italic=True,
               color="5A6673")
           + run("tô đỏ", italic=True, bold=True, color=RED)
           + run(".", italic=True, color="5A6673")))

    A(heading("Thuật toán 2 — CTIM, chép nguyên văn", 1))
    A(algo("Algorithm 2  Community-based influence maximization algorithm",
           ctim_io(), ctim_lines()))
    A(caption("Chép từ hình in. Ký hiệu dùng đúng bộ của bài báo: 𝒢, 𝒰, ℰ, 𝒟, "
              "π, θ, η, P(z | i), K, 𝒮, C, Z, M, U, c^v_m, ΔI_m, I[m, k], "
              "s[m, k], I_m, c_m, c_j, 𝒮_j, u_k, j. Một lưu ý về bản in: dòng "
              "36 hiện ra là “if I([C, k−1] + ΔI_m ≥ …” với một dấu ngoặc mở "
              "thừa sau I. Dòng 35 và 36 đều tham chiếu ô bảng I[C, k−1], nên "
              "chúng tôi đọc dấu ngoặc đó là lỗi trình bày và không chép lại."))

    A(heading("Thuật toán 3 — CTIM-G, sửa từ Thuật toán 2", 1))
    A(algo("Algorithm 3  Community-based influence maximization with a global "
           "objective", ctimg_io(), ctimg_lines()))
    A(caption("Dòng 1–25 giống Thuật toán 2 từng ký tự và để nguyên màu đen. "
              "Mọi dòng tô đỏ là chỗ khác. Số thứ tự dòng cũng đỏ theo để dò "
              "nhanh."))

    A(heading("Bảng đối chiếu", 1))
    A(table(["Thuật toán 2", "Thuật toán 3", "Thay đổi"], [
        ["1–24", "1–24", "không đổi — Eq (10)–(12) và Eq (19) dùng chung"],
        ["25", "25", "không đổi"],
        ["—", "26–32", "MỚI: dựng đường cong I_m(j) cho từng cộng đồng, "
                       "lợi ích đo bằng I trên toàn đồ thị chứ không phải I_m"],
        ["26–28", "33–35", "s[0, k] bị bỏ; bảng đánh chỉ số từ k = 0"],
        ["29–31", "36–38", "không đổi"],
        ["32–33", "39–40", "đảo thứ tự hai vòng: m ở ngoài, k ở trong"],
        ["34", "—", "BỎ: ΔI_m không còn, vì lợi ích biên đã nằm trong đường cong"],
        ["35", "41", "truy hồi có thêm vòng lặp j và số hạng cap"],
        ["36–40", "42", "BỎ bảng s[m, k]; thay bằng t[m, k] ghi SỐ SUẤT"],
        ["42", "45–48", "truy vết ngược ra vector ngân sách a thay vì một chỉ số j"],
        ["43", "49–52", "một lượt tham lam CHUNG dưới hạn ngạch thay vì một hạt "
                        "giống mỗi vòng k; kèm điều kiện dừng sớm khi lợi ích "
                        "biên về 0"],
        ["44", "53", "𝒮_j biến mất; chỉ còn 𝒮 toàn cục"],
        ["—", "55", "MỚI: bước phòng hờ, giữ tập tốt hơn trong hai"],
        ["—", "56", "MỚI: khoảng cách độc lập Γ"],
    ], [1500, 1500, 6000], ["center", "center", "left"]))

    A(heading("Ba khác biệt cốt lõi", 1))
    A(para(run("1.  Hàm mục tiêu. ", bold=True)
           + run("Thuật toán 2 dòng 34 đo lợi ích bằng ")
           + M(sub("I", "m"))
           + run(" — Eq (18) trên đồ thị con cảm sinh bởi ")
           + M(sub("c", "m"))
           + run(". Thuật toán 3 dòng 29 đo bằng ") + M(mr("I"))
           + run(" trên toàn đồ thị 𝒢. Đây là thay đổi duy nhất về mô hình; "
                 "hai khác biệt còn lại là hệ quả bắt buộc của nó.")))
    A(para(run("2.  Bảng quy hoạch động. ", bold=True)
           + run("Truy hồi ở dòng 35 của Thuật toán 2 phát đúng một hạt giống "
                 "mỗi vòng k và không có vòng lặp j. Dòng 41 của Thuật toán 3 "
                 "có vòng lặp j, nên cấp được nhiều hạt giống cho một cộng "
                 "đồng trong một quyết định. Bảng ")
           + M(mr("s") + mdel(mr("m, k"), "[", "]"))
           + run(" ghi CHỈ SỐ cộng đồng; bảng ")
           + M(mr("t") + mdel(mr("m, k"), "[", "]"))
           + run(" ghi SỐ SUẤT. Hai bảng khác nghĩa dù cùng kích thước.")))
    A(para(run("3.  Cách lắp lời giải. ", bold=True)
           + run("Thuật toán 2 dòng 43–44 chọn hạt giống trong riêng ")
           + M(sub("c", "j")) + run(" và đo lợi ích đối với riêng ")
           + M(sub("𝒮", "j"))
           + run(", rồi ghép các tập cộng đồng lại. Thuật toán 3 dòng 49–52 "
                 "chạy MỘT lượt tham lam trên hợp các cộng đồng được cấp vốn "
                 "và đo lợi ích đối với ") + M(mr("𝒮"))
           + run(" đầy đủ, nên phần ảnh hưởng chồng lấn giữa các cộng đồng chỉ "
                 "được tính một lần.")))
    return "".join(b)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default=os.path.join("results", "algorithms_diff.docx"))
    args = p.parse_args(argv)
    write_docx(args.out, build())
    c, g = ctim_lines(), ctimg_lines()
    assert len(c) == 45, "Algorithm 2 phai co dung 45 dong nhu hinh in, dang co %d" % len(c)
    print("[write] %s   (CTIM %d lines, CTIM-G %d lines, %d marked changed)"
          % (args.out, len(c), len(g), sum(1 for x in g if x[2])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
