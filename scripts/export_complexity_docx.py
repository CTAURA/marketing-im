"""Time-complexity analysis of CTIM and CTIM-G, in the paper's own style.

Section 4.3 of Huang et al. analyses CTIM by attributing a cost to each band of
lines in Algorithm 2, in the symbols U, E, D, M, Z, C, K, |c_p|, n_ip, n_op and
T_p, and closes with a dominance statement.  This document restates that
analysis, checks it against the implementation, then carries the same symbols
and the same treatment through CTIM-G's (G1)-(G7) so the two can be compared
term by term rather than through two different vocabularies.

Measured figures are read from the six run logs at build time.

Two claims this document deliberately does NOT make:

  * that ``n_i / n_ip`` predicts CTIM-G's QUALITY gain.  It is a cost ratio.
    Across the five datasets with a probe block, the sign of the quality
    difference does not follow it (Ciao 2.08 loses, Digg 1.18 wins).
  * a corrected Yelp timing ratio.  ``yelp_paper_h01.log`` predates the Eq (12)
    subtraction, so its CTIM column is inflated and its ratio is a lower bound.

Usage
-----
    python scripts/export_complexity_docx.py --out results/complexity.docx
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_docx import (bullet, caption, code_block, heading, para,   # noqa: E402
                         run, table, write_docx)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DATASETS = [
    ("Yelp", "yelp_paper_h01.log"),
    ("Digg", "digg_paper_h01.log"),
    ("Epinions", "epinions_paper_h01.log"),
    ("Ciao", "ciao_paper_h01.log"),
    ("Last.fm", "lastfm_paper_h01.log"),
    ("Delicious", "delicious_paper_h01.log"),
]


def harvest():
    out = []
    for name, log in _DATASETS:
        with open(os.path.join(_REPO, log), encoding="utf-8",
                  errors="replace") as fh:
            t = fh.read()
        nm = re.search(r"n=\|V\|=(\d+)  m=\|E\|=(\d+)  K=(\d+)  C=(\d+)", t)
        arb = re.search(r"MEASURED: (\d+) arborescences built, mean size ([\d.]+)", t)
        ev = re.search(r"MEASURED: (\d+) curve gain evals, (\d+) realisation gain evals", t)
        ph = re.findall(r"phases: mia ([\d.]+)s  solo ([\d.]+)s  curves ([\d.]+)s"
                        r"  dp ([\d.]+)s  realise ([\d.]+)s", t)
        fu = re.search(r"mean \|MIIA\| on the FULL graph\s*:\s*([\d.]+)", t)
        ind = re.search(r"mean \|MIIA\| on the INDUCED subgraph:\s*([\d.]+)", t)
        used = re.search(r"communities used: (\d+) / (\d+)", t)
        out.append({
            "name": name,
            "U": int(nm.group(1)), "E": int(nm.group(2)),
            "K": int(nm.group(3)), "C": int(nm.group(4)),
            "arb": int(arb.group(1)) if arb else None,
            "t": float(arb.group(2)) if arb else None,
            "curve_ev": int(ev.group(1)) if ev else None,
            "real_ev": int(ev.group(2)) if ev else None,
            "n_i": float(fu.group(1)) if fu else None,
            "n_ip": float(ind.group(1)) if ind else None,
            "used": int(used.group(1)) if used else None,
            "phases": [[float(x) for x in row] for row in ph],
            "t_ctim": [float(x) for x in
                       re.findall(r"CTIM\s+[\d.]+\s+[+-]?[\d.]+%\s+--\s+([\d.]+)\s+\d+", t)],
            "t_ctimg": [float(x) for x in
                        re.findall(r"CTIM-G\s+[\d.]+\s+[+-]?[\d.]+%\s+--\s+([\d.]+)\s+\d+", t)],
            "eq12_ok": "rebuilt inside CTIM" in t,
        })
    return out


def _mean(v):
    return sum(v) / len(v) if v else float("nan")


def _pm(d, i):
    return _mean([p[i] for p in d["phases"]]) if d["phases"] else float("nan")


def build():
    D = harvest()
    by = {d["name"]: d for d in D}
    b = []
    A = b.append

    A(para(run("Phân tích độ phức tạp thời gian: CTIM và CTIM-G", bold=True),
           style="Title", spacing_after=60))
    A(para(run("Trình bày theo đúng lối Mục 4.3 của bài báo gốc: quy chi phí về "
               "từng dải dòng của thuật toán, dùng nguyên bộ ký hiệu của bài "
               "báo, và kết bằng một phát biểu chi phối. CTIM-G được phân tích "
               "bằng cùng bộ ký hiệu đó để hai bên so được từng số hạng.",
               italic=True, color="5A6673")))

    # ------------------------------------------------------------------ 1
    A(heading("1.  Ký hiệu", 1))
    A(para("Chúng tôi giữ nguyên ký hiệu của bài báo gốc và bổ sung ba ký hiệu "
           "cho CTIM-G. Ngưỡng cắt đường đi được bài báo viết là θ; bản cài đặt "
           "viết là h. Hai ký hiệu chỉ cùng một đại lượng."))
    A(table(["Ký hiệu", "Ý nghĩa", "Của ai"], [
        ["U, E, D", "số người dùng, số liên kết, số bản ghi lan truyền tiềm năng",
         "bài báo"],
        ["M, Z, C, K", "số sản phẩm, số chủ đề, số cộng đồng, số hạt giống",
         "bài báo"],
        ["T₁, T₂", "số vòng lấy mẫu chủ đề tiềm ẩn; số vòng lấy mẫu s, s′, c, c′",
         "bài báo"],
        ["|c_p|", "số nút của cộng đồng LỚN NHẤT", "bài báo"],
        ["t_ip", "thời gian lớn nhất để dựng MIIA(v, θ) với v ∈ c_p", "bài báo"],
        ["n_ip, n_op",
         "max_{v∈c_p} |MIIA(v, θ)| và max_{v∈c_p} |MIOA(v, θ)|, trên ĐỒ THỊ CON",
         "bài báo"],
        ["T_p", "thời gian tính độ lan truyền của một nút trong cộng đồng c_p",
         "bài báo"],
        ["t_i", "thời gian lớn nhất để dựng MIIA(v, θ) với v ∈ 𝒰, trên TOÀN đồ thị",
         "bổ sung"],
        ["n_i, n_o",
         "max_{v∈𝒰} |MIIA(v, θ)| và max_{v∈𝒰} |MIOA(v, θ)|, trên TOÀN đồ thị",
         "bổ sung"],
        ["T_g", "thời gian tính độ lan truyền của một nút trên toàn đồ thị 𝒢",
         "bổ sung"],
    ], [1500, 5900, 1600], ["center", "left", "center"]))
    A(para(run("Quan hệ then chốt: ", bold=True)
           + run("vì 𝒢_m là đồ thị con của 𝒢 nên mọi đường đi trong 𝒢_m đều tồn "
                 "tại trong 𝒢. Do đó n_ip ≤ n_i, n_op ≤ n_o và t_ip ≤ t_i, với "
                 "mọi p. Toàn bộ khoảng cách chi phí giữa hai thuật toán nằm ở "
                 "ba bất đẳng thức này.")))

    # ------------------------------------------------------------------ 2
    A(heading("2.  Quá trình sinh (Thuật toán 1 của bài báo)", 1))
    A(para("Cho C và Z cố định, thời gian của quá trình sinh tuyến tính theo cỡ "
           "dữ liệu vào và theo số vòng lặp:"))
    A(code_block(["O( T1·M  +  T2·(E + D) )"]))
    A(para("Mỗi vòng lấy mẫu chủ đề tiềm ẩn tốn O(Z·F) với F là số thuộc tính "
           "của sản phẩm. Mỗi vòng lấy mẫu s, s′, c, c′ tốn O(C²) trên mỗi liên "
           "kết và trên mỗi bản ghi lan truyền tiềm năng, cho O(C²·(E + D)) mỗi "
           "vòng."))
    A(para(run("Phần này CHUNG cho cả hai thuật toán. ", bold=True)
           + run("CTIM-G không thay đổi mô hình sinh, không thay đổi bộ lấy "
                 "mẫu, và dùng lại nguyên vẹn π, η, θ đã học. Vì thế số hạng "
                 "này triệt tiêu trong mọi phép so sánh dưới đây và chúng tôi "
                 "không nhắc lại nó.")))

    # ------------------------------------------------------------------ 3
    A(heading("3.  CTIM (Thuật toán 2 của bài báo)", 1))
    A(para(run("Dòng 1–24.", bold=True)
           + run(" Tính cường độ ảnh hưởng giữa người dùng trên mọi sản phẩm "
                 "(Eq (10)–(12)) và giá trị thuộc cộng đồng (Eq (19)):")))
    A(code_block(["O( C²·Z  +  C²·D  +  M·D·Z  +  U·C )"]))
    A(para(run("Dòng 25–31.", bold=True)
           + run(" Khởi tạo. O(K) cho hai mảng I[0, k] và s[0, k]; O(C) cho "
                 "𝒮, 𝒮₁, …, 𝒮_C và mảng I[m, 0].")))
    A(para(run("Dòng 32–41.", bold=True)
           + run(" Vòng kép trên k = 1..K và m = 1..C. Mỗi ô cần độ lan truyền "
                 "của một nút trong cộng đồng, tốn T_p. Vậy dải này tốn "
                 "O(K·C·T_p), với")))
    A(code_block(["T_p  =  O( |c_p|·t_ip  +  n_ip·n_op·log|c_p| )"]))
    A(para("Số hạng thứ nhất là chi phí dựng cây MIIA cho mọi nút của cộng đồng "
           "lớn nhất; số hạng thứ hai là chi phí cập nhật gia tăng sau mỗi hạt "
           "giống, có thêm log|c_p| do thao tác trên hàng đợi ưu tiên."))
    A(para(run("Dòng 42–45.", bold=True)
           + run(" Khai thác hạt giống bằng mô hình MIA: O(K·|c_p|·T_p).")))
    A(para(run("Tổng cộng.", bold=True)
           + run(" Độ phức tạp của CTIM là")))
    A(code_block([
        "O( K·C·T_p  +  K·|c_p|·T_p )",
        "",
        "voi dieu kien   K·C·T_p + K·|c_p|·T_p  >>  C²Z + C²D + MDZ + UC",
    ]))
    A(caption("Đây là phát biểu của bài báo gốc, được chúng tôi kiểm lại trên "
              "bản cài đặt. Một điểm bản cài đặt làm khác: bảng quy hoạch động "
              "được hợp nhất vào vòng chọn hạt giống thay vì chạy riêng, nên "
              "dải 32–41 và dải 42–45 chia sẻ cùng một lần dựng cây; điều này "
              "không đổi bậc tiệm cận nhưng làm hằng số nhỏ đi."))

    # ------------------------------------------------------------------ 4
    A(heading("4.  CTIM-G", 1))
    A(para("Chúng tôi phân tích CTIM-G theo đúng cách trên, quy chi phí về từng "
           "công thức (G1)–(G7). Dải dòng 1–24 và 25–31 không đổi, vì CTIM-G "
           "dùng chung Eq (10)–(12) và Eq (19); ba pha sau thì khác."))
    A(para(run("(G1) — bộ đánh giá toàn cục.", bold=True)
           + run(" Thay vì C đồ thị con cảm sinh, CTIM-G dựng MỘT đối tượng MIA "
                 "trên 𝒢. Mọi cây MIIA và MIOA được dựng lười và đệm lại, nên "
                 "mỗi nút chạy Dijkstra nhiều nhất hai lần trong cả lần chạy. "
                 "Chi phí O(U·t_i). Đây là chỗ t_ip bị thay bằng t_i.")))
    A(para(run("(G2) — đường cong lợi ích, dải tương ứng dòng 32–41.", bold=True)
           + run(" Với mỗi cộng đồng m, một lượt tham lam lười chọn tới "
                 "min(K, |c_m|) hạt giống, mỗi lần tính lợi ích tốn O(n_i·n_o). "
                 "Đặt T_g là thời gian tính độ lan truyền của một nút trên 𝒢:")))
    A(code_block(["T_g  =  O( U·t_i  +  n_i·n_o·log U )"]))
    A(para("Dải này tốn O(K·C·T_g), cùng dạng với O(K·C·T_p) của CTIM nhưng với "
           "T_g thay cho T_p. Vì n_ip ≤ n_i và n_op ≤ n_o nên T_p ≤ T_g luôn "
           "đúng: CTIM-G không bao giờ rẻ hơn CTIM ở dải này."))
    A(para(run("(G4) — phân bổ ngân sách.", bold=True)
           + run(" Bảng (C+1)×(K+1), mỗi ô quét j ∈ [0, min(cap_m, k)]. Chi phí "
                 "O(C·K²), chặt hơn là O(K·Σ_m min(K, |c_m|)). So với O(C·K) "
                 "của CTIM, đây là số hạng DUY NHẤT mà CTIM-G đổi bậc tiệm cận. "
                 "Mục 6 cho thấy nó cũng là số hạng rẻ nhất.")))
    A(para(run("(G5) — hiện thực hoá có hạn ngạch, dải tương ứng dòng 42–45.",
               bold=True)
           + run(" Một lượt tham lam duy nhất chọn K hạt giống trên hợp các "
                 "cộng đồng được cấp vốn, mỗi lần tính lợi ích tốn O(n_i·n_o). "
                 "Chi phí O(K·n_i·n_o), so với O(K·|c_p|·T_p) của CTIM. Số hạng "
                 "này RẺ HƠN của CTIM vì nó không dựng lại cây nào — mọi cây đã "
                 "có sẵn từ (G1).")))
    A(para(run("(G6)(G7) — kết sổ.", bold=True)
           + run(" Hai lần tính Eq (18) trên tập K hạt giống, mỗi lần "
                 "O(K·n_o·n_i), cộng một phép trừ. CTIM không có dải này.")))
    A(para(run("Tổng cộng.", bold=True)
           + run(" Độ phức tạp của CTIM-G là")))
    A(code_block([
        "O( K·C·T_g  +  K·n_i·n_o  +  C·K² )",
        "",
        "voi dieu kien   K·C·T_g  >>  C²Z + C²D + MDZ + UC",
    ]))

    # ------------------------------------------------------------------ 5
    A(heading("5.  So sánh từng số hạng", 1))
    A(table(["Dải", "CTIM", "CTIM-G", "Quan hệ"], [
        ["Quá trình sinh", "O(T₁M + T₂(E+D))", "giống hệt", "triệt tiêu"],
        ["Dòng 1–24", "O(C²Z + C²D + MDZ + UC)", "giống hệt", "triệt tiêu"],
        ["Dòng 25–31", "O(K) + O(C)", "O(K) + O(C)", "triệt tiêu"],
        ["Dựng cây", "nằm trong T_p, dùng t_ip",
         "O(U·t_i), tách riêng", "CTIM-G ĐẮT HƠN, hệ số t_i / t_ip"],
        ["Dòng 32–41", "O(K·C·T_p)", "O(K·C·T_g)",
         "CTIM-G đắt hơn vì T_p ≤ T_g"],
        ["Phân bổ ngân sách", "O(C·K)", "O(C·K²)",
         "CTIM-G đổi BẬC, nhưng là số hạng rẻ nhất"],
        ["Dòng 42–45", "O(K·|c_p|·T_p)", "O(K·n_i·n_o)",
         "CTIM-G RẺ HƠN: cây đã dựng sẵn"],
        ["Kết sổ", "không có", "O(K·n_i·n_o)", "chỉ CTIM-G"],
    ], [1600, 2300, 2200, 2900], ["left", "left", "left", "left"]))
    A(para(run("Kết luận hình thức. ", bold=True)
           + run("CTIM-G không đổi bậc tiệm cận của bất kỳ dải nào ngoại trừ "
                 "bảng phân bổ ngân sách, và bảng đó là dải rẻ nhất trong cả "
                 "hai thuật toán. Chênh lệch chi phí THỰC TẾ đến từ đúng một "
                 "thay đổi: t_ip, n_ip, n_op được thay bằng t_i, n_i, n_o. "
                 "Nghĩa là khoảng cách chi phí bằng tỉ số kích thước cây trên "
                 "toàn đồ thị so với trên đồ thị con — không phụ thuộc K, không "
                 "phụ thuộc C.")))

    # ------------------------------------------------------------------ 6
    A(heading("6.  Số đo", 1))
    rows = []
    for d in D:
        rows.append([
            d["name"],
            "{:,}".format(d["U"]).replace(",", "."),
            ("%.2f" % d["n_i"]) if d["n_i"] else "không đo",
            ("%.2f" % d["n_ip"]) if d["n_ip"] else "không đo",
            ("%.2f" % (d["n_i"] / d["n_ip"])) if d["n_ip"] else "—",
            ("%.2f×" % (_mean(d["t_ctimg"]) / _mean(d["t_ctim"])))
            + ("" if d["eq12_ok"] else " (*)"),
        ])
    A(table(["Tập dữ liệu", "U", "n_i (toàn đồ thị)", "n_ip (cảm sinh)",
             "n_i / n_ip", "Thời gian CTIM-G / CTIM"],
            rows, [1400, 1300, 1700, 1600, 1200, 1800],
            ["left", "right", "right", "right", "right", "right"]))
    A(caption("Bảng 1. n_i và n_ip đo bằng phép thăm dò 200 nút trong mỗi nhật "
              "ký; nhật ký Yelp không có khối này. (*) Yelp chạy trên bản công "
              "cụ trước khi khoản trừ Eq (12) tồn tại, nên mẫu số bị phóng to "
              "và tỉ số in ra là chặn dưới."))
    A(para(run("Lý thuyết dự báo đúng chiều. ", bold=True)
           + run("Trên Delicious n_i / n_ip = 1.00 và thời gian gần như bằng "
                 "nhau; trên Epinions tỉ số 4.09 đi cùng 3.34×. Nhưng đây là "
                 "quan hệ về CHI PHÍ. Tỉ số này KHÔNG dự báo được chênh lệch "
                 "CHẤT LƯỢNG: Ciao có 2.08 và CTIM-G thua, Digg có 1.18 và "
                 "CTIM-G thắng.")))

    rows = []
    for d in D:
        rows.append([
            d["name"],
            "{:,}".format(d["arb"]).replace(",", ".") if d["arb"] else "—",
            "%.2f U" % (d["arb"] / d["U"]) if d["arb"] else "—",
            "{:,}".format(d["curve_ev"]).replace(",", ".") if d["curve_ev"] else "—",
            str(d["real_ev"]) if d["real_ev"] is not None else "—",
            str(d["used"]) if d["used"] is not None else "—",
        ])
    A(table(["Tập dữ liệu", "Số cây dựng", "so với U",
             "Lần tính lợi ích (G2)", "(G5)", "Cộng đồng được cấp vốn"],
            rows, [1400, 1500, 1200, 1900, 1000, 2000],
            ["left", "right", "right", "right", "right", "right"]))
    A(caption("Bảng 2. Bộ đếm do CTIM-G tự ghi. Số cây luôn nằm giữa U và 2U "
              "đúng như dự đoán ở Mục 4, vì MIIA và MIOA được đệm riêng và mỗi "
              "nút chạy Dijkstra nhiều nhất một lần cho mỗi chiều."))
    A(para("Số lần tính lợi ích ở (G5) nhỏ hơn ở (G2) một tới hai bậc — %d so "
           "với %d trên Digg. Đó là hàng đợi lười làm việc: phần lớn ứng viên "
           "không bao giờ lên tới đỉnh nên không tốn phép tính nào. Điều này "
           "giải thích vì sao số hạng O(K·n_i·n_o) của (G5) nhỏ hơn nhiều so "
           "với O(K·|c_p|·T_p) của CTIM trên thực tế, chứ không chỉ trên giấy."
           % (by["Digg"]["real_ev"], by["Digg"]["curve_ev"])))

    # ------------------------------------------------------------------ 7
    A(heading("7.  Điều phân tích tiệm cận không thấy", 1))
    rows = []
    for d in D:
        tot = sum(_pm(d, i) for i in range(5))
        rows.append([
            d["name"], "%.2f" % _pm(d, 1), "%.2f" % _pm(d, 2),
            "%.2f" % _pm(d, 3), "%.2f" % _pm(d, 4),
            ("%.0f%%" % (100 * _pm(d, 2) / tot)) if tot > 0 else "—",
        ])
    A(table(["Tập dữ liệu", "solo (s)", "(G2) đường cong (s)", "(G4) QHĐ (s)",
             "(G5) hiện thực hoá (s)", "phần của (G2)"],
            rows, [1400, 1300, 1900, 1400, 1900, 1300],
            ["left", "right", "right", "right", "right", "right"]))
    A(caption("Bảng 3. Phân rã thời gian CTIM-G theo pha, trung bình ba sản "
              "phẩm. Pha dựng MIA đọc 0.00 s vì bộ chạy tự dựng đối tượng MIA "
              "rồi tiêm vào để đọc lại bộ đếm (scripts/run_ctim_global.py:"
              "356-358); chi phí O(U·t_i) vì thế hiện ra trong hai cột solo và "
              "(G2), không phải trong một cột riêng."))
    A(bullet(run("Số hạng O(C·K²) không bao giờ là nút cổ chai. ", bold=True)
             + run("Bảng phân bổ tốn 0.00–0.01 s trên mọi tập, kể cả Yelp với "
                   "%s nút. Nó đúng về tiệm cận nhưng với C = %d và K = %d thì "
                   "hằng số quá nhỏ để quan sát được. Tối ưu bước này là vô ích."
                   % ("{:,}".format(by["Yelp"]["U"]).replace(",", "."),
                      by["Yelp"]["C"], by["Yelp"]["K"]))))
    A(bullet(run("Phân tích tiệm cận dùng |c_p| — cộng đồng LỚN NHẤT — nhưng "
                 "phần lớn cộng đồng không được cấp vốn. ", bold=True)
             + run("Trên năm trong sáu tập, bảng phân bổ chỉ cấp cho 2 tới 5 "
                   "cộng đồng trong số gần 100. Số hạng O(K·C·T_g) vì thế là "
                   "chặn trên rất lỏng: pha (G5) chỉ duyệt hợp các cộng đồng có "
                   "ngân sách, không phải cả C cộng đồng.")))
    A(bullet(run("Bảng solo là chi phí bắt buộc, không phải chi phí phụ. ",
                 bold=True)
             + run("Nó tốn %.2f s trên Yelp — bậc thứ hai sau (G2). Nó tính "
                   "I({u}) cho MỌI nút nên đắt tuyến tính theo U, dù chỉ cần "
                   "K = %d hạt giống. Phân tích tiệm cận gộp nó vào O(U·t_i) và "
                   "vì thế không làm nó nổi lên như một mục tiêu tối ưu riêng."
                   % (_pm(by["Yelp"], 1), by["Yelp"]["K"]))))
    A(bullet(run("Pha (G2) chi phối, nhưng không ở mọi tập. ", bold=True)
             + run("Nó chiếm %.0f%% thời gian trên Yelp và %.0f%% trên Epinions, "
                   "nhưng chỉ %.0f%% trên Delicious, nơi pha solo lớn gấp đôi "
                   "nó. Khi cây ảnh hưởng suy biến về nút đơn lẻ, dựng đường "
                   "cong gần như không tốn gì."
                   % (100 * _pm(by["Yelp"], 2) / sum(_pm(by["Yelp"], i) for i in range(5)),
                      100 * _pm(by["Epinions"], 2) / sum(_pm(by["Epinions"], i) for i in range(5)),
                      100 * _pm(by["Delicious"], 2) / sum(_pm(by["Delicious"], i) for i in range(5))))))
    return "".join(b)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("results", "complexity.docx"))
    args = p.parse_args(argv)
    D = harvest()
    write_docx(args.out, build())
    print("[write] %s   (%d datasets, %d with an n_i / n_ip probe)"
          % (args.out, len(D), sum(1 for d in D if d["n_ip"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
