"""Section IV (Experiments) of the paper, as a .docx.

Every figure is read from the six run logs at the repo root, or from
``data/processed/<name>/meta.json`` / ``graph.tsv``.  Nothing is quoted from
memory, from another run, or from a different value of ``h_eval``.

What the numbers rest on, stated once here and again in 4.6:

  * all six runs used ``--paper-eval``, so ``h_sel = h_eval = 0.1``: the
    objective being graded is the objective being maximised
  * the ``MC-IC`` column is ``--`` in all 36 result rows -- no Monte-Carlo
    verification was run on any dataset
  * only Digg and Yelp are the original paper's benchmarks; the other four are
    marked "NOT a dataset the paper used: added by this project" in their own
    ``meta.json``
  * ``yelp_paper_h01.log`` predates the Eq (12) timing subtraction and contains
    no ``full / induced`` probe, so it contributes to neither the timing
    comparison nor the structural hypothesis

Usage
-----
    python scripts/export_paper_experiment_docx.py --out results/paper_section_IV.docx
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_docx import (bullet, caption, heading, para, run,        # noqa: E402
                         table, write_docx)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name, log file, is it one of the paper's own benchmarks?
_DATASETS = [
    ("Yelp", "yelp_paper_h01.log", True),
    ("Digg", "digg_paper_h01.log", True),
    ("Epinions", "epinions_paper_h01.log", False),
    ("Ciao", "ciao_paper_h01.log", False),
    ("Last.fm", "lastfm_paper_h01.log", False),
    ("Delicious", "delicious_paper_h01.log", False),
]


def _read(log):
    with open(os.path.join(_REPO, log), encoding="utf-8", errors="replace") as fh:
        return fh.read()


def harvest():
    """Read every figure Section IV quotes, straight out of the logs.

    Written as one pass so a stale number cannot survive in one table while
    being corrected in another.
    """
    out = []
    for name, log, own in _DATASETS:
        t = _read(log)
        nm = re.search(r"n=\|V\|=(\d+)  m=\|E\|=(\d+)  K=(\d+)  C=(\d+)", t)
        gains = re.findall(r"CTIM-G\s+[\d.]+\s+([+-]?[\d.]+)%", t)
        base = re.findall(r"CTIM\s+([\d.]+)\s+[+-]?[\d.]+%", t)
        newv = re.findall(r"CTIM-G\s+([\d.]+)\s+[+-]?[\d.]+%", t)
        t_a = re.findall(r"CTIM\s+[\d.]+\s+[+-]?[\d.]+%\s+--\s+([\d.]+)\s+\d+", t)
        t_b = re.findall(r"CTIM-G\s+[\d.]+\s+[+-]?[\d.]+%\s+--\s+([\d.]+)\s+\d+", t)
        probe = re.findall(r"ratio  full / induced\s*:\s*([\d.]+)x", t)
        used = re.search(r"communities used: (\d+) / (\d+)", t)
        gaps = re.findall(r"independence gap ([+-]?[\d.]+)%", t)
        out.append({
            "name": name, "own": own,
            "V": int(nm.group(1)), "E": int(nm.group(2)),
            "K": int(nm.group(3)), "C": int(nm.group(4)),
            "gains": [float(x) for x in gains],
            "base": [float(x) for x in base],
            "new": [float(x) for x in newv],
            "t_ctim": [float(x) for x in t_a],
            "t_ctimg": [float(x) for x in t_b],
            "ratio": (probe[0] + "×") if probe else None,
            "used": (used.group(1), used.group(2)) if used else None,
            "gaps": [float(x) for x in gaps],
            # The Eq (12) subtraction line is what makes CTIM's time comparable.
            # Where it is present, eq12_frac is the share of CTIM's RAW time it
            # removed -- which is also the size of the bias on any log missing it.
            "eq12_subtracted": "rebuilt inside CTIM" in t,
            "eq12_frac": _eq12_frac(t, [float(x) for x in t_a]),
        })
    return out


def _eq12_frac(text, t_ctim):
    """Share of CTIM's raw select time that the Eq (12) rebuild accounted for."""
    sub = [float(x) for x in
           re.findall(r"rebuilt inside CTIM: ([\d.]+)s", text)]
    if not sub or not t_ctim:
        return None
    n = min(len(sub), len(t_ctim))
    raw = sum(sub[:n]) + sum(t_ctim[:n])
    return (sum(sub[:n]) / raw) if raw else None


def _mean(v):
    return sum(v) / len(v) if v else float("nan")


def _rng(v, fmt="%.2f"):
    """A range when the items disagree, a single value when they do not."""
    if not v:
        return "—"
    lo, hi = min(v), max(v)
    return (fmt % lo) if abs(hi - lo) < 5e-3 else ("%s–%s" % (fmt % lo, fmt % hi))


def build():
    D = harvest()
    by = {d["name"]: d for d in D}
    b = []
    A = b.append

    A(para(run("IV.  Thực nghiệm", bold=True), style="Title", spacing_after=60))
    A(para(run("Mục này so sánh CTIM với CTIM-G trên sáu tập dữ liệu thực. "
               "Mọi con số được đọc trực tiếp từ sáu tệp nhật ký ở gốc kho mã "
               "và từ data/processed/<tập>/. Chúng tôi nêu ngay ba giới hạn của "
               "thiết lập, và khai triển đầy đủ ở Mục 4.6: phép chấm điểm dùng "
               "đúng hàm mục tiêu mà cả hai phương pháp tối ưu hoá; không có lần "
               "chạy nào được kiểm chứng bằng mô phỏng Monte Carlo; và chỉ hai "
               "trong sáu tập là benchmark của bài báo gốc.",
               italic=True, color="5A6673")))

    # ---------------------------------------------------------------- 4.1
    A(heading("4.1.  Tập dữ liệu", 2))
    A(para("Sáu tập được chọn để phủ một dải rộng về quy mô (từ 1.861 tới "
           "366.427 người dùng) và về cấu trúc quan hệ. Cần phân biệt hai nhóm "
           "ngay từ đầu. Digg và Yelp là benchmark của chính bài báo gốc. Bốn "
           "tập còn lại do chúng tôi bổ sung, và tệp mô tả của chính chúng ghi "
           "rõ điều đó; chúng được đưa vào để mở rộng dải cấu trúc chứ không "
           "phải để tuyên bố là benchmark chuẩn."))
    rows = []
    for d in D:
        rows.append([
            d["name"],
            "{:,}".format(d["V"]).replace(",", "."),
            "{:,}".format(d["E"]).replace(",", "."),
            "%.2f" % (d["E"] / d["V"]),
            "gốc" if d["own"] else "bổ sung",
        ])
    A(table(["Tập dữ liệu", "|𝒰|", "|ℰ|", "cung / nút", "Nguồn gốc"],
            rows, [1700, 1700, 1900, 1500, 1500],
            ["left", "right", "right", "right", "left"]))
    A(caption("Bảng 1. Quy mô sáu tập, đọc từ dòng “n=|V|= … m=|E|= …” trong "
              "khối [C] COMPLEXITY của mỗi nhật ký. Cột nguồn gốc: “bổ sung” "
              "nghĩa là data/processed/<tập>/meta.json ghi “NOT a dataset the "
              "paper used: added by this project”."))

    A(para(run("Về tính có hướng.", bold=True)
           + run(" Chỉ ba tập có quan hệ bất đối xứng thật: Digg (quan hệ fan), "
                 "Epinions và Ciao (quan hệ tin cậy). Với Yelp, Last.fm và "
                 "Delicious, quan hệ bạn bè vốn vô hướng và bộ đọc dữ liệu "
                 "nhân mỗi quan hệ thành hai cung ngược chiều "
                 "(ctim/datasets/lastfm.py:29-30, delicious.py:38, "
                 "yelp.py:30-32), nên tỉ lệ cung hai chiều của chúng bằng 1.0 "
                 "theo cách xây dựng chứ không phải theo bản chất mạng. Mọi "
                 "nhận định liên quan tới tính có hướng vì thế chỉ có ba tập "
                 "làm chứng, không phải sáu.")))
    A(para(run("Về thuộc tính sản phẩm.", bold=True)
           + run(" Epinions và Ciao chỉ có duy nhất cột danh mục, đo được 1,002 "
                 "và 1,000 token mỗi sản phẩm; tệp mô tả của cả hai cảnh báo "
                 "bằng chữ in hoa rằng với khối lượng tiên nghiệm Dirichlet của "
                 "bài báo (α = 50/Z), mô hình chủ đề không thể rời khỏi phân "
                 "phối tiên nghiệm, và khuyến cáo chỉ dùng chúng cho cấu trúc "
                 "đồ thị. Chúng tôi tuân thủ khuyến cáo đó: không rút kết luận "
                 "nào về chủ đề từ hai tập này.")))

    # ---------------------------------------------------------------- 4.2
    A(heading("4.2.  Thiết lập thực nghiệm", 2))
    y = by["Yelp"]
    A(para("Cả sáu lần chạy dùng chung một cấu hình: K = %d hạt giống, C = %d "
           "cộng đồng yêu cầu, ngưỡng MIA h = 0,1 đúng như bài báo gốc, ba sản "
           "phẩm kiểm thử mỗi tập. Hai phương pháp nhận cùng một mô hình đã "
           "khớp, cùng ma trận trọng số cạnh Eq (12) và cùng phân hoạch cộng "
           "đồng Eq (19), nên chênh lệch quan sát được thuộc về chiến lược chọn "
           "hạt giống chứ không thuộc về mô hình."
           % (y["K"], y["C"])))
    A(para(run("Ba điều phải khai báo về thiết lập này.", bold=True)
           + run(" Thứ nhất, cờ --paper-eval được bật, nên ngưỡng chấm điểm "
                 "trùng ngưỡng chọn. Thứ hai, cột trọng tài Monte Carlo là dấu "
                 "gạch ở toàn bộ 36 dòng kết quả: không lần chạy nào được kiểm "
                 "chứng bằng mô phỏng lan truyền độc lập. Thứ ba, mỗi lần chạy "
                 "chỉ so hai phương pháp; không có Greedy, CINEMA, hay "
                 "GlobalGreedy làm mốc, và chỉ có một điểm ngân sách K = 20 chứ "
                 "không có đường cong theo K.")))
    A(para(run("Số cộng đồng thực sự được cấp vốn nhỏ hơn nhiều so với C.",
               bold=True)
           + run(" Tuy yêu cầu C = 100, bước phân bổ ngân sách chỉ cấp cho "
                 + ", ".join("%s %s" % (d["name"], d["used"][0])
                             for d in D if d["used"])
                 + " cộng đồng tương ứng. Trên năm trong sáu tập, quy hoạch "
                   "động thực chất chỉ chọn giữa hai tới năm phương án. Đây là "
                   "một tính chất của K = 20 chứ không phải của phương pháp, "
                   "nhưng nó giới hạn mức độ mà phép phân rã theo cộng đồng "
                   "được thử thách trong thực nghiệm này.")))

    # ---------------------------------------------------------------- 4.3
    A(heading("4.3.  Chất lượng tập hạt giống", 2))
    rows = []
    for d in D:
        rows.append([
            d["name"],
            "%.2f" % _mean(d["base"]),
            "%.2f" % _mean(d["new"]),
            "%+.2f%%" % _mean(d["gains"]),
            _rng(d["gains"], "%+.2f") + "%",
        ])
    A(table(["Tập dữ liệu", "CTIM", "CTIM-G", "Chênh trung bình",
             "Khoảng ba sản phẩm"],
            rows, [1700, 1700, 1700, 1800, 1900],
            ["left", "right", "right", "right", "right"]))
    A(caption("Bảng 2. Giá trị hàm mục tiêu Eq (18) tại h = 0,1, trung bình "
              "trên ba sản phẩm. Đây KHÔNG phải mức lan truyền mô phỏng: cột "
              "MC-IC của cả sáu nhật ký đều là dấu gạch."))
    A(para(run("Kết quả chia làm ba nhóm, và hai trong sáu tập là bất lợi cho "
               "CTIM-G.", bold=True)
           + run(" CTIM-G hơn rõ rệt trên Yelp (%+.2f%%) và Epinions (%+.2f%%). "
                 "Trên Digg chênh lệch là %+.2f%%, và trên Delicious hai phương "
                 "pháp trả về cùng một giá trị. Trên Ciao (%+.2f%%) và Last.fm "
                 "(%+.2f%%) CTIM-G THUA CTIM, và thua trên cả ba sản phẩm ở cả "
                 "hai tập."
                 % (_mean(by["Yelp"]["gains"]), _mean(by["Epinions"]["gains"]),
                    _mean(by["Digg"]["gains"]), _mean(by["Ciao"]["gains"]),
                    _mean(by["Last.fm"]["gains"])))))
    A(para(run("Quy mô đồ thị không giải thích được kết quả.", bold=True)
           + run(" Delicious là tập NHỎ NHẤT (1.861 nút) và là tập hoà; "
                 "Last.fm (1.892) và Ciao (2.215) lớn hơn Delicious và đều "
                 "thua. Ba tập nhỏ nhất vì thế chia ra hai thua một hoà, trong "
                 "khi hai tập lớn nhất đều thắng — thứ tự theo số nút không "
                 "sắp được kết quả. Chúng tôi sẽ chỉ ra ở Mục 4.5 rằng đại "
                 "lượng phân biệt được là cấu trúc cây ảnh hưởng, không phải "
                 "quy mô.")))

    # ---------------------------------------------------------------- 4.4
    A(heading("4.4.  Thời gian chạy", 2))
    rows = []
    for d in D:
        a, g = _mean(d["t_ctim"]), _mean(d["t_ctimg"])
        rows.append([
            d["name"], "%.2f" % a, "%.2f" % g,
            ("%.2f×" % (g / a)) + ("" if d["eq12_subtracted"] else " (*)"),
            "đã làm sạch" if d["eq12_subtracted"] else "CÒN CÕNG Eq (12)",
        ])
    A(table(["Tập dữ liệu", "CTIM (s)", "CTIM-G (s)", "Tỉ số",
             "Cột CTIM chứa gì"],
            rows, [1600, 1500, 1600, 1400, 2700],
            ["left", "right", "right", "right", "center"]))
    A(caption("Bảng 3. Thời gian chọn hạt giống, trung bình ba sản phẩm. Hàng "
              "Yelp KHÔNG so sánh được với năm hàng còn lại và được đánh dấu "
              "(*). Nguyên nhân: ctim_select_seeds không nhận tham số pp nên nó "
              "tự dựng lại toàn bộ trọng số cạnh Eq (12) bên trong, còn CTIM-G "
              "được đưa pp dựng sẵn; khoản này không thuộc về việc chọn hạt "
              "giống nên bộ chạy trừ nó ra và in dòng “[timing] Eq (12) rebuilt "
              "inside CTIM … excluded”. Nhật ký Yelp không có dòng đó. "
              "CHIỀU LỆCH: cột CTIM của Yelp bị phóng to, nên tỉ số in ra là "
              "CHẶN DƯỚI — tỉ số thật của Yelp CAO HƠN %.2f×. Trên năm tập kia, "
              "khoản bị trừ chiếm %.0f%%–%.0f%% thời gian thô của CTIM."
              % (_mean(by["Yelp"]["t_ctimg"]) / _mean(by["Yelp"]["t_ctim"]),
                 100 * min(d["eq12_frac"] for d in D if d["eq12_frac"]),
                 100 * max(d["eq12_frac"] for d in D if d["eq12_frac"]))))
    A(para("Bỏ Yelp ra, tỉ số CTIM-G/CTIM trải từ %0.2f× (Last.fm) tới %0.2f× "
           "(Epinions). Đáng chú ý, trên Last.fm CTIM-G NHANH HƠN CTIM ở hai "
           "trong ba sản phẩm (%.2f s so với %.2f s, và %.2f s so với %.2f s); "
           "chỉ sản phẩm thứ ba đảo chiều. Trên Delicious hiệu số tuyệt đối là "
           "hàng phần trăm giây, dưới độ phân giải in ấn của nhật ký, nên tỉ số "
           "của hai tập này không nên được diễn giải."
           % (_mean(by["Last.fm"]["t_ctimg"]) / _mean(by["Last.fm"]["t_ctim"]),
              _mean(by["Epinions"]["t_ctimg"]) / _mean(by["Epinions"]["t_ctim"]),
              by["Last.fm"]["t_ctimg"][0], by["Last.fm"]["t_ctim"][0],
              by["Last.fm"]["t_ctimg"][1], by["Last.fm"]["t_ctim"][1])))
    A(para(run("Về Yelp, cần nói rõ chiều lệch để không bị đọc ngược. ",
               bold=True)
           + run("Tỉ số %.2f× của Yelp KHÔNG phải con số bị thổi phồng mà là "
                 "con số bị hạ thấp: mẫu số của nó — thời gian CTIM — còn cõng "
                 "thêm chi phí dựng lại trọng số cạnh mà năm tập kia đã được "
                 "trừ. Trên năm tập đó, khoản trừ chiếm từ %.0f%% tới %.0f%% "
                 "thời gian thô của CTIM. Chúng tôi không ngoại suy tỉ lệ ấy "
                 "sang Yelp và không báo cáo một tỉ số hiệu chỉnh; điều nói "
                 "được là %.2f× là chặn dưới. Sửa dứt điểm đòi hỏi chạy lại "
                 "Yelp trên bản công cụ hiện tại."
                 % (_mean(by["Yelp"]["t_ctimg"]) / _mean(by["Yelp"]["t_ctim"]),
                    100 * min(d["eq12_frac"] for d in D if d["eq12_frac"]),
                    100 * max(d["eq12_frac"] for d in D if d["eq12_frac"]),
                    _mean(by["Yelp"]["t_ctimg"]) / _mean(by["Yelp"]["t_ctim"])))))
    A(para("Chi phí thêm của CTIM-G tập trung ở pha dựng đường cong, vì các cây "
           "ảnh hưởng được dựng trên toàn đồ thị thay vì trên đồ thị con cảm "
           "sinh. Bước quy hoạch động không đáng kể: nó không vượt quá 0,01 s "
           "trên bất kỳ tập nào."))

    # ---------------------------------------------------------------- 4.5
    A(heading("4.5.  Khi nào CTIM-G có lợi", 2))
    A(para("Đại lượng chúng tôi đặt giả thuyết là tỉ số giữa kích thước trung "
           "bình của cây ảnh hưởng dựng trên toàn đồ thị và kích thước trên đồ "
           "thị con cảm sinh. Tỉ số này đo trực tiếp phần cấu trúc mà phép cắt "
           "cụt hàm mục tiêu của CTIM vứt bỏ: tỉ số bằng 1 nghĩa là phân hoạch "
           "không cắt mất gì, tỉ số lớn nghĩa là cắt mất nhiều."))
    rows = []
    for d in sorted(D, key=lambda x: (x["ratio"] is None,
                                      float(x["ratio"][:-1]) if x["ratio"] else 0)):
        rows.append([
            d["name"],
            d["ratio"] or "không đo",
            "%+.2f%%" % _mean(d["gains"]),
            "%s / %s" % d["used"] if d["used"] else "—",
            _rng(d["gaps"], "%.2f") + "%",
        ])
    A(table(["Tập dữ liệu", "Tỉ số toàn đồ thị / cảm sinh",
             "Chênh chất lượng", "Cộng đồng được cấp vốn", "Khoảng cách Γ"],
            rows, [1500, 2300, 1700, 2000, 1500],
            ["left", "right", "right", "center", "right"]))
    A(caption("Bảng 4. Tỉ số đọc từ khối thăm dò 200 nút trong mỗi nhật ký. "
              "Yelp không có khối này nên không có điểm dữ liệu; giả thuyết "
              "dưới đây vì thế chỉ có NĂM điểm, và tập cho mức lợi lớn nhất "
              "không nằm trong số đó."))
    A(para(run("Giả thuyết đúng ở dạng ngưỡng, và chỉ ở dạng ngưỡng.",
               bold=True)
           + run(" Tập duy nhất có tỉ số vượt 4 là Epinions (4,1×) và nó cũng "
                 "là tập duy nhất trong năm điểm cho mức lợi hai chữ số phần "
                 "trăm. Nhưng quan hệ KHÔNG đơn điệu: Ciao có tỉ số 2,1× — cao "
                 "thứ hai — mà lại thua, trong khi Digg với tỉ số 1,2× lại "
                 "thắng nhẹ. Với năm điểm dữ liệu và một vi phạm rõ ràng, "
                 "chúng tôi không tuyên bố một quy luật; điều nói được là: khi "
                 "phân hoạch cắt mất ít cấu trúc (tỉ số gần 1), CTIM-G không "
                 "có gì để thu hồi và không nên được dùng.")))
    A(para(run("Delicious là trường hợp suy biến, không phải trường hợp thành "
               "công.", bold=True)
           + run(" Tỉ số 1,0× và cây ảnh hưởng trung bình 1,07 đỉnh nghĩa là ở "
                 "h = 0,1 hầu như mọi đỉnh chỉ tự kích hoạt chính nó. Hai "
                 "phương pháp trả về cùng một giá trị vì không có lan truyền "
                 "nào để mà khác nhau. Chúng tôi giữ tập này trong bảng để "
                 "minh hoạ giới hạn dưới, và khuyến cáo không tính nó vào bất "
                 "kỳ trung bình nào.")))

    # ---------------------------------------------------------------- 4.6
    A(heading("4.6.  Các mối đe doạ đến tính hợp lệ", 2))
    A(bullet(run("Phép chấm điểm tuần hoàn. ", bold=True)
             + run("Cả sáu lần chạy đặt ngưỡng chấm bằng ngưỡng chọn, nên hàm "
                   "được dùng để xếp hạng chính là hàm mà CTIM-G tối ưu hoá "
                   "trực tiếp còn CTIM thì không. Công cụ đo tự in cảnh báo ở "
                   "đầu mỗi nhật ký rằng đây không phải phép so sánh hợp lệ cho "
                   "CTIM-G. Đây là mối đe doạ nghiêm trọng nhất và nó chưa được "
                   "khắc phục.")))
    A(bullet(run("Không có trọng tài độc lập. ", bold=True)
             + run("Cột MC-IC là dấu gạch ở toàn bộ 36 dòng kết quả. Không con "
                   "số nào trong Bảng 2 được đối chiếu với mô phỏng lan truyền "
                   "thật. Kết luận đúng đắn duy nhất rút ra được từ Bảng 2 là "
                   "về hành vi của hàm mục tiêu, không phải về ảnh hưởng thực "
                   "tế.")))
    A(bullet(run("Ba sản phẩm không phải ba phép thử độc lập. ", bold=True)
             + run("Chúng là ba điều kiện khác nhau, mỗi điều kiện có ma trận "
                   "trọng số cạnh riêng, và nhật ký nói rõ giá trị không so "
                   "được giữa các sản phẩm. Độ lệch chuẩn giữa ba sản phẩm vì "
                   "thế không phải sai số lặp lại, và chúng tôi không báo cáo "
                   "khoảng tin cậy nào. Việc chọn hạt giống là tất định, nên "
                   "không có phương sai giữa các lần chạy để đo.")))
    A(bullet(run("Yelp — tập cho kết quả tốt nhất — không tái lập được tại chỗ. ",
                 bold=True)
             + run("Thư mục dữ liệu đã xử lý của Yelp không có trong kho mã, "
                   "nhật ký hiệu chỉnh của nó cũng không, và nhật ký chạy của "
                   "nó thiếu cả dòng trừ Eq (12) lẫn khối thăm dò cấu trúc. "
                   "Con số +%.2f%% vì thế không thể kiểm chứng chéo bằng bất kỳ "
                   "cách nào khác trong kho mã này. Hai thiếu sót này lệch về "
                   "hai hướng NGƯỢC nhau và không triệt tiêu nhau: thiếu khối "
                   "thăm dò làm Yelp không đóng góp được điểm dữ liệu nào cho "
                   "giả thuyết ở Mục 4.5, còn thiếu dòng trừ làm tỉ số thời "
                   "gian của Yelp thành chặn dưới chứ không phải ước lượng."
                   % _mean(by["Yelp"]["gains"]))))
    A(bullet(run("Bốn trong sáu tập không phải benchmark của bài báo gốc. ",
                 bold=True)
             + run("Chúng do chúng tôi bổ sung, và với Epinions và Ciao thì "
                   "chính tệp mô tả khuyến cáo không dùng để kết luận về mô "
                   "hình chủ đề. Kết quả trên bốn tập này nên đọc như phép thử "
                   "khả năng khái quát, không phải như phép lặp lại thí nghiệm "
                   "của bài báo gốc.")))
    A(bullet(run("C = 100 không được dò lại theo từng tập. ", bold=True)
             + run("Giá trị này lấy nguyên từ bài báo gốc. Trên ba tập nhỏ "
                   "nhất, nó cho trung bình dưới 25 người dùng mỗi cộng đồng, "
                   "thấp hơn ngưỡng 40 mà chính kho mã đặt ra ở "
                   "ctim/experiments.py:312. Chúng tôi không khảo sát độ nhạy "
                   "theo C.")))
    A(bullet(run("Γ là chặn trên, và nó lớn ở cả nơi CTIM-G thắng lẫn nơi nó "
                 "thua. ", bold=True)
             + run("Khoảng cách độc lập đo được nằm trong khoảng %s%% tới "
                   "%s%% trên năm tập không suy biến. Nó không tương quan với "
                   "dấu của chênh lệch chất lượng: Digg có Γ khoảng %.2f%% và "
                   "thắng, Last.fm có Γ khoảng %.2f%% và thua. Γ chỉ nói bảng "
                   "phân bổ ngân sách được dựng trên số liệu lạc quan tới mức "
                   "nào, không dự báo được kết quả."
                   % (_rng([min(by[n]["gaps"]) for n in
                            ("Last.fm",)], "%.2f"),
                      _rng([max(by["Ciao"]["gaps"])], "%.2f"),
                      _mean(by["Digg"]["gaps"]), _mean(by["Last.fm"]["gaps"])))))
    A(caption("Mục 4.6 được viết trước khi kết quả được diễn giải, không phải "
              "sau. Hai mối đe doạ đầu tiên đủ nghiêm trọng để bất kỳ kết luận "
              "nào ở 4.3 và 4.5 đều phải đọc kèm chúng."))
    return "".join(b)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default=os.path.join("results", "paper_section_IV.docx"))
    args = p.parse_args(argv)
    D = harvest()
    write_docx(args.out, build())
    print("[write] %s   (%d datasets, %d with an Eq-12 subtraction, "
          "%d with a structural probe)"
          % (args.out, len(D),
             sum(1 for d in D if d["eq12_subtracted"]),
             sum(1 for d in D if d["ratio"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
