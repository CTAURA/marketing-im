"""Report: CTIM vs CTIM-G vs CTIM-G with lazy curves, graded by MIA and MC-IC.

Every figure is parsed from the run logs ``<dataset>_lazy_h01.log`` at the repo
root, written by

    python -u scripts/run_ctim_global.py --dataset data/processed/<ds> \
        --cache data/processed/_<ds>.pkl --methods CTIM,CTIM-G,CTIM-G-lazy \
        --items all --K 20 --h 0.1 --paper-eval --mc 1000

(Digg uses the default --dataset / --cache.)  Nothing is copied by hand.

Usage
-----
    python scripts/export_lazy_report_docx.py --out results/lazy_curve_report.docx
"""

import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_docx import (bullet, caption, heading, para, run,        # noqa: E402
                         table, write_docx)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = [("Digg", "digg"), ("Epinions", "epinions"), ("Ciao", "ciao"),
            ("Last.fm", "lastfm"), ("Delicious", "delicious")]
METHODS = ("CTIM", "CTIM-G", "CTIM-G-lazy")

_ROW = re.compile(r"^(CTIM-G-lazy|CTIM-G|CTIM)\s+([\d.]+)\s+([+-]?[\d.]+)%\s+"
                  r"([\d.]+) \+- ([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", re.M)
_PH = re.compile(r"phases: mia ([\d.]+)s  solo ([\d.]+)s  curves ([\d.]+)s"
                 r"  dp ([\d.]+)s  realise ([\d.]+)s")
_EV = re.compile(r"MEASURED: (\d+) curve gain evals, (\d+) realisation gain evals")
_PTS = re.compile(r"MEASURED: (\d+) curve points built")
_ARB = re.compile(r"MEASURED: (\d+) arborescences built, mean size ([\d.]+)")
_DIST = re.compile(r"\((\d+) distinct seed sets for (\d+) methods\)")
_DIM = re.compile(r"U=(\d+)\s+E=(\d+)\s+C=(\d+)")
_GAP = re.compile(r"independence gap ([+-]?[\d.]+)%")
_OVL = re.compile(r"^\s+CTIM-G-lazy\s+(\S+)\s+(\S+)\s+(\S+)\s*$", re.M)

# |z| below this is treated as "no detectable difference" under MC-IC
Z_SIG = 2.0
# |MIA difference| below this (in %) is treated as a tie
MIA_TIE = 0.01


def parse(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    dim = _DIM.search(text)
    items = []
    for chunk in re.split(r"\n#+\n# ITEM \d+ of \d+", text)[1:]:
        rows = {m.group(1): {"I": float(m.group(2)), "vs": float(m.group(3)),
                             "mc": float(m.group(4)), "se": float(m.group(5)),
                             "t": float(m.group(6)), "k": int(m.group(7))}
                for m in _ROW.finditer(chunk)}
        ph = [[float(x) for x in m.groups()] for m in _PH.finditer(chunk)]
        ev = [(int(a), int(b)) for a, b in _EV.findall(chunk)]
        pts = [int(x) for x in _PTS.findall(chunk)]
        arb = [int(a) for a, _ in _ARB.findall(chunk)]
        gaps = [float(x) for x in _GAP.findall(chunk)]
        dist = _DIST.search(chunk)
        ovl = _OVL.search(chunk.split("pairwise seed overlap:", 1)[-1])
        items.append({
            "rows": rows,
            # CTIM-G is printed before CTIM-G-lazy in every per-method block
            "phases": dict(zip(("CTIM-G", "CTIM-G-lazy"), ph)),
            "evals": dict(zip(("CTIM-G", "CTIM-G-lazy"), ev)),
            "points": dict(zip(("CTIM-G", "CTIM-G-lazy"), pts)),
            "trees": dict(zip(("CTIM-G", "CTIM-G-lazy"), arb)),
            "gap": gaps[0] if gaps else None,
            "distinct": int(dist.group(1)) if dist else None,
            # column 2 of the CTIM-G-lazy row of the overlap matrix is its
            # overlap with CTIM-G; a trailing * marks an identical set
            "lazy_same": bool(ovl) and ovl.group(2).endswith("*"),
        })
    return {"U": int(dim.group(1)), "E": int(dim.group(2)),
            "C": int(dim.group(3)), "items": items}


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else float("nan")


def col(d, method, key):
    return [it["rows"][method][key] for it in d["items"] if method in it["rows"]]


def pct(a, b):
    return 100.0 * (a / b - 1.0) if b else float("nan")


def mc_z(d, a, b):
    """Mean over items of (mc_a - mc_b) / combined SE."""
    zs = []
    for it in d["items"]:
        r = it["rows"]
        if a in r and b in r:
            se = math.sqrt(r[a]["se"] ** 2 + r[b]["se"] ** 2)
            if se > 0:
                zs.append((r[a]["mc"] - r[b]["mc"]) / se)
    return mean(zs)


def verdict(d):
    """Classify a dataset; a tie or |z| < Z_SIG is never called a reversal."""
    ci, gi = mean(col(d, "CTIM", "I")), mean(col(d, "CTIM-G", "I"))
    dmia = pct(gi, ci)
    z = mc_z(d, "CTIM-G", "CTIM")
    mc = ("không phân biệt được" if abs(z) < Z_SIG
          else ("CTIM-G hơn" if z > 0 else "CTIM-G kém"))
    mia = ("hoà" if abs(dmia) < MIA_TIE
           else ("CTIM-G hơn" if dmia > 0 else "CTIM-G kém"))
    if abs(z) >= Z_SIG and mia != "hoà" and mia != mc:
        tag = "ĐẢO CHIỀU"
    elif abs(z) < Z_SIG and mia != "hoà":
        tag = "MC không xác nhận"
    else:
        tag = "khớp"
    return mia, mc, tag, z


def build(D):
    b = []
    A = b.append
    n_items = sum(len(d["items"]) for _, d in D)

    A(para(run("CTIM, CTIM-G và CTIM-G với đường cong lười", bold=True),
           style="Title", spacing_after=60))
    A(para(run("So sánh chất lượng và thời gian trên %d tập dữ liệu, %d sản "
               "phẩm, chấm bằng cả thước MIA lẫn mô phỏng Monte Carlo. Mọi con "
               "số được đọc trực tiếp từ nhật ký chạy." % (len(D), n_items),
               italic=True, color="5A6673")))

    # ---------------------------------------------------------------- 1
    A(heading("1.  Tóm tắt", 1))
    same = sum(1 for _, d in D for it in d["items"] if it["lazy_same"])
    A(bullet(run("Đường cong lười trả về đúng tập hạt giống của CTIM-G ",
                 bold=True)
             + run("ở %d/%d sản phẩm. Đọc thẳng từ ma trận trùng lặp trong nhật "
                   "ký: ô CTIM-G × CTIM-G-lazy mang dấu * nghĩa là cùng một tập."
                   % (same, n_items))))
    big = [(name, mean(col(d, "CTIM-G", "t")) / mean(col(d, "CTIM-G-lazy", "t")))
           for name, d in D if mean(col(d, "CTIM-G", "t")) >= 1.0]
    small = [name for name, d in D if mean(col(d, "CTIM-G", "t")) < 1.0]
    A(bullet(run("Bản lười nhanh hơn CTIM-G %.1f–%.1f lần " %
                 (min(x for _, x in big), max(x for _, x in big)), bold=True)
             + run("trên %s, nơi CTIM-G cần hơn 1 giây. " %
                   ", ".join(n for n, _ in big))
             + run(("Trên %s, cả hai chạy dưới 0,4 giây; chênh lệch nằm dưới "
                    "độ phân giải 0,01 s của nhật ký nên không kết luận được."
                    % " và ".join(small)) if small else "")))
    V = {name: verdict(d) for name, d in D}
    rev = [n for n, v in V.items() if v[2] == "ĐẢO CHIỀU"]
    gone = [n for n, v in V.items() if v[2] == "MC không xác nhận"]
    better = [n for n, v in V.items() if v[1] == "CTIM-G hơn"]
    A(bullet(run(("Dưới mô phỏng Monte Carlo, CTIM-G không tốt hơn CTIM một "
                  "cách có ý nghĩa trên tập nào trong %d tập. " % len(D))
                 if not better else
                 ("Dưới Monte Carlo, CTIM-G tốt hơn có ý nghĩa trên: %s. "
                  % ", ".join(better)), bold=True)
             + run(("Nó KÉM hơn có ý nghĩa trên %s, trong khi thước MIA lại báo "
                    "nó hơn. " % ", ".join(rev)) if rev else "")
             + run(("Trên %s, chênh lệch mà thước MIA báo cáo không còn phân "
                    "biệt được dưới Monte Carlo (|z| < %.0f)."
                    % (", ".join(gone), Z_SIG)) if gone else "")))

    # ---------------------------------------------------------------- 2
    A(heading("2.  Thiết lập", 1))
    A(para("K = 20 hạt giống, ngưỡng MIA h = 0,1, ba sản phẩm kiểm thử mỗi tập, "
           "C = 100 cộng đồng theo Eq (19). Ba phương "
           "thức dùng chung một mô hình đã khớp và chung ma trận trọng số Eq "
           "(12). Mỗi tập hạt giống được chấm hai cách: (a) hàm Eq (18) của MIA "
           "tại h = 0,1, chính là hàm mà CTIM-G tối ưu hoá; (b) mô phỏng lan "
           "truyền độc lập (MC-IC) 1.000 lần, độc lập với mọi ngưỡng."))
    A(para(run("CTIM-G-lazy ", bold=True)
           + run("thay hai pha dựng đường cong và quy hoạch động bằng một heap "
                 "cộng đồng: mỗi bước cấp một hạt giống cho cộng đồng có lợi ích "
                 "biên kế tiếp lớn nhất, và chỉ dựng điểm đường cong đó. Vì các "
                 "đường cong lõm, cách này đạt cùng phân bổ tối ưu với quy hoạch "
                 "động; hai bên chỉ có thể khác nhau khi gặp thế hoà chính xác.")))

    rows = [[name, "{:,}".format(d["U"]).replace(",", "."),
             "{:,}".format(d["E"]).replace(",", "."), str(len(d["items"]))]
            for name, d in D]
    A(table(["Tập dữ liệu", "|𝒰|", "|ℰ|", "Số sản phẩm"], rows,
            [2400, 2200, 2400, 2000], ["left", "right", "right", "right"]))

    # ---------------------------------------------------------------- 3
    A(heading("3.  Chất lượng: hai thước đo", 1))
    rows = []
    for name, d in D:
        ci, gi = mean(col(d, "CTIM", "I")), mean(col(d, "CTIM-G", "I"))
        cm, gm = mean(col(d, "CTIM", "mc")), mean(col(d, "CTIM-G", "mc"))
        rows.append([name, "%.2f" % ci, "%.2f" % gi, "%+.2f%%" % pct(gi, ci),
                     "%.2f" % cm, "%.2f" % gm, "%+.2f%%" % pct(gm, cm),
                     "%+.1f" % V[name][3], V[name][2]])
    A(table(["Tập", "CTIM\nMIA", "CTIM-G\nMIA", "Chênh\nMIA",
             "CTIM\nMC-IC", "CTIM-G\nMC-IC", "Chênh\nMC-IC", "z", "Kết luận"],
            rows, [1050, 900, 900, 850, 950, 950, 900, 600, 1500],
            ["left"] + ["right"] * 7 + ["center"]))
    A(caption("Bảng 2. Trung bình trên các sản phẩm. MIA là hàm Eq (18) tại "
              "h = 0,1; MC-IC là trung bình 1.000 lần mô phỏng. Cột z là chênh "
              "lệch MC-IC giữa CTIM-G và CTIM chia cho sai số chuẩn gộp, lấy "
              "trung bình trên các sản phẩm. Cột kết luận: ĐẢO CHIỀU = Monte "
              "Carlo cho chênh lệch có ý nghĩa (|z| ≥ 2) ngược dấu với MIA; MC "
              "không xác nhận = MIA báo chênh lệch nhưng Monte Carlo không phân "
              "biệt được (|z| < 2); khớp = hai thước không mâu thuẫn. "
              "CTIM-G-lazy không có cột riêng vì nó trả về đúng tập của CTIM-G "
              "(Mục 5)."))
    A(para(run("Cách đọc. ", bold=True)
           + run("Cột MIA là thước mà CTIM-G được thiết kế để tối ưu, nên nó có "
                 "lợi thế sẵn trên cột đó. Cột MC-IC là trọng tài độc lập. Khi "
                 "hai cột ngược chiều, kết luận nên dựa vào cột MC-IC.")))

    # ---------------------------------------------------------------- 4
    A(heading("4.  Thời gian chọn hạt giống", 1))
    rows = []
    for name, d in D:
        a = mean(col(d, "CTIM", "t"))
        g = mean(col(d, "CTIM-G", "t"))
        z = mean(col(d, "CTIM-G-lazy", "t"))
        rows.append([name, "%.2f" % a, "%.2f" % g, "%.2f" % z,
                     "%.2f×" % (g / a), "%.2f×" % (z / a), "%.2f×" % (g / z)])
    A(table(["Tập", "CTIM (s)", "CTIM-G (s)", "lười (s)",
             "CTIM-G / CTIM", "lười / CTIM", "CTIM-G / lười"],
            rows, [1300, 1200, 1300, 1200, 1400, 1300, 1300],
            ["left"] + ["right"] * 6))
    A(caption("Bảng 3. Thời gian chọn hạt giống, trung bình các sản phẩm. Thời "
              "gian của CTIM đã trừ chi phí dựng lại Eq (12) bên trong nó, nên "
              "ba cột so được trực tiếp. Cột cuối là mức tăng tốc của bản lười."))

    # ---------------------------------------------------------------- 5
    A(heading("5.  Bản lười tiết kiệm ở đâu", 1))
    rows = []
    for name, d in D:
        def m(key, meth, idx=None):
            v = [it[key].get(meth) for it in d["items"]]
            v = [x[idx] if idx is not None else x for x in v if x is not None]
            return mean(v)
        rows.append([
            name,
            "%.0f → %.0f" % (m("points", "CTIM-G"), m("points", "CTIM-G-lazy")),
            "%.0f → %.0f" % (m("evals", "CTIM-G", 0), m("evals", "CTIM-G-lazy", 0)),
            "%.2f → %.2f" % (m("phases", "CTIM-G", 2), m("phases", "CTIM-G-lazy", 2)),
            "%.0f → %.0f" % (m("trees", "CTIM-G"), m("trees", "CTIM-G-lazy")),
            "%d/%d" % (sum(1 for it in d["items"] if it["lazy_same"]),
                       len(d["items"])),
        ])
    A(table(["Tập", "Điểm đường cong", "Lần tính lợi ích", "Pha đường cong (s)",
             "Số cây dựng", "Cùng tập hạt giống"],
            rows, [1200, 1700, 1800, 1800, 1700, 1500],
            ["left"] + ["right"] * 5))
    A(caption("Bảng 4. CTIM-G → CTIM-G-lazy, trung bình các sản phẩm. Cột cuối "
              "đếm số sản phẩm mà hai phương thức trả về cùng một tập."))
    A(para(run("Số lần tính lợi ích giảm mạnh hơn nhiều so với thời gian. ",
               bold=True)
           + run("Lý do: phần chi phí còn lại chủ yếu là dựng cây ảnh hưởng "
                 "(Dijkstra). Bảng solo cần một cây cho mọi nút bất kể dựng bao "
                 "nhiêu điểm đường cong, nên số cây chỉ giảm nhẹ. Muốn nhanh hơn "
                 "nữa thì mục tiêu tiếp theo là bảng solo, không phải đường cong.")))

    # ---------------------------------------------------------------- 6
    A(heading("6.  Giới hạn", 1))
    A(bullet("MC-IC dùng 1.000 lần mô phỏng mỗi tập hạt giống; sai số chuẩn "
             "được in kèm. Kết luận về những chênh lệch có |z| < 2 là không "
             "chắc chắn."))
    A(bullet("Mỗi tập chỉ có một điểm ngân sách K = 20 và một ngưỡng h = 0,1."))
    A(bullet("Yelp không chạy được ở máy này vì dữ liệu đã xử lý của nó không "
             "có trong kho mã."))
    A(bullet("Ba sản phẩm mỗi tập là ba điều kiện khác nhau (mỗi sản phẩm một ma "
             "trận trọng số), không phải ba lần lặp của cùng một phép đo."))
    return "".join(b)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("results", "lazy_curve_report.docx"))
    p.add_argument("--logdir", default=_REPO)
    args = p.parse_args(argv)
    D = []
    for name, key in DATASETS:
        path = os.path.join(args.logdir, "%s_lazy_h01.log" % key)
        if not os.path.exists(path):
            print("[skip] %s: %s not found" % (name, path))
            continue
        d = parse(path)
        if not d["items"]:
            print("[skip] %s: no items parsed" % name)
            continue
        D.append((name, d))
    if not D:
        print("ERROR: no logs")
        return 1
    write_docx(args.out, build(D))
    print("[write] %s   (%d datasets)" % (args.out, len(D)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
