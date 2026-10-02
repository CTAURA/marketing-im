"""Short results table: CTIM vs CTIM-G vs CTIM-GL (CTIM-G with lazy curves).

Influence spread and seed-selection time only, averaged over the test items,
read from ``<dataset>_lazy_h01.log`` at the repo root.  In those logs the lazy
variant is labelled ``CTIM-G-lazy``; this report calls it CTIM-GL.

Usage
-----
    python scripts/export_short_report_docx.py --out results/ctim_ctimg_ctimgl.docx
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_docx import caption, heading, para, run, table, write_docx   # noqa: E402
from export_lazy_report_docx import DATASETS, col, mean, parse, pct      # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# report name -> label used in the run logs
NAMES = (("CTIM", "CTIM"), ("CTIM-G", "CTIM-G"), ("CTIM-GL", "CTIM-G-lazy"))


def build(D):
    b = []
    A = b.append
    A(para(run("So sánh CTIM, CTIM-G và CTIM-GL", bold=True), style="Title",
           spacing_after=60))
    A(para(run("K = 20 hạt giống, h = 0,1, trung bình trên 3 sản phẩm mỗi tập. "
               "CTIM-GL là CTIM-G với đường cong lười.", italic=True,
               color="5A6673")))

    A(heading("Độ lan truyền ảnh hưởng", 1))
    rows = []
    for name, d in D:
        base = mean(col(d, "CTIM", "I"))
        r = [name]
        for _, key in NAMES:
            r.append("%.2f" % mean(col(d, key, "I")))
        for _, key in NAMES[1:]:
            r.append("%+.2f%%" % pct(mean(col(d, key, "I")), base))
        rows.append(r)
    A(table(["Tập dữ liệu", "CTIM", "CTIM-G", "CTIM-GL",
             "CTIM-G so với CTIM", "CTIM-GL so với CTIM"],
            rows, [1500, 1300, 1300, 1300, 1800, 1800],
            ["left"] + ["right"] * 5))
    A(caption("Độ lan truyền I(S) theo Eq (18) của mô hình MIA tại h = 0,1."))

    A(heading("Thời gian chọn hạt giống (giây)", 1))
    rows = []
    for name, d in D:
        rows.append([name] + ["%.2f" % mean(col(d, key, "t")) for _, key in NAMES])
    A(table(["Tập dữ liệu", "CTIM", "CTIM-G", "CTIM-GL"],
            rows, [2400, 2200, 2200, 2200], ["left", "right", "right", "right"]))
    return "".join(b)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("results", "ctim_ctimg_ctimgl.docx"))
    p.add_argument("--logdir", default=_REPO)
    args = p.parse_args(argv)
    D = []
    for name, key in DATASETS:
        path = os.path.join(args.logdir, "%s_lazy_h01.log" % key)
        if os.path.exists(path):
            d = parse(path)
            if d["items"]:
                D.append((name, d))
    if not D:
        print("ERROR: no logs")
        return 1
    write_docx(args.out, build(D))
    print("[write] %s   (%d datasets)" % (args.out, len(D)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
