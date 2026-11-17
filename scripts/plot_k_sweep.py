#!/usr/bin/env python3
"""Paper-style Fig. 2 plots (spread vs K, running time vs K) for every dataset.

Reads the K-sweep workbook (sheet ``ChiTiet``: one row per dataset x K x test
item x method), averages over the test items exactly as the workbook's own
AVERAGEIFS formulas do, and draws one figure per dataset in the layout of the
CTIM paper's Fig. 2:

    (a) Influence spread   -- I_h_eval, the MIA spread of Eq (18), linear y
    (b) Running time       -- seed-selection time in seconds, log y

MC-IC spread is deliberately not plotted.

The xlsx is parsed with the standard library (no openpyxl); only the drawing
needs matplotlib.

    python scripts/plot_k_sweep.py --xlsx path/to/sheet.xlsx --out results/figures/k_sweep
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

DATASETS = [("lastfm", "Last.fm"), ("delicious", "Delicious"), ("ciao", "Ciao"),
            ("digg", "Digg"), ("epinions", "Epinions"), ("yelp", "Yelp")]
KS = [5, 10, 15, 20]

# Fixed method -> style.  Colours validated for colour-vision deficiency
# (dataviz validate_palette.js: all checks pass, worst adjacent CVD dE 25).
# CTIM-G and CTIM-G-lazy return the same seeds, so their spread lines coincide;
# the lazy line is dashed with hollow markers so both stay visible.
METHODS = [
    ("CTIM",        dict(color="#4472C4", marker="D", linestyle="-",  markersize=8.5,
                         markerfacecolor="#4472C4", zorder=3)),
    ("CTIM-G",      dict(color="#ED7D31", marker="s", linestyle="-",  markersize=7,
                         markerfacecolor="#ED7D31", zorder=4)),
    ("CTIM-G-lazy", dict(color="#7B3FA0", marker="^", linestyle="--", markersize=8,
                         markerfacecolor="white", markeredgewidth=1.6, zorder=5)),
]

SANS = ["Arial", "DejaVu Sans"]
SERIF = ["Times New Roman", "DejaVu Serif"]
AXIS_GREY = "#8C8C8C"

# ---------------------------------------------------------------------------
# xlsx -> rows (stdlib only)
# ---------------------------------------------------------------------------

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def _col_idx(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_sheet(path, name):
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % _NS["m"])))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = {r.get("Id"): r.get("Target")
            for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    for sh in wb.find("m:sheets", _NS):
        if sh.get("name") != name:
            continue
        target = rels[sh.get("{%s}id" % _NS["r"])].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        rows = []
        for row in ET.fromstring(z.read(target)).iter("{%s}row" % _NS["m"]):
            cells = {}
            for c in row.findall("m:c", _NS):
                v = c.find("m:v", _NS)
                if c.get("t") == "s" and v is not None:
                    val = shared[int(v.text)]
                elif c.get("t") == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter("{%s}t" % _NS["m"]))
                else:
                    val = v.text if v is not None else None
                cells[_col_idx(c.get("r"))] = val
            if cells:
                rows.append([cells.get(i) for i in range(max(cells) + 1)])
        return rows
    raise SystemExit("sheet %r not found in %s" % (name, path))


def load_means(path):
    """{(dataset, K, method): (mean I_h_eval, mean select seconds, n_items)}"""
    rows = read_sheet(path, "ChiTiet")
    hdr = rows[0]
    col = {h: i for i, h in enumerate(hdr)}
    need = ["Dataset", "K", "Method", "I_h_eval (Eq 18)", "Select time (s)"]
    missing = [h for h in need if h not in col]
    if missing:
        raise SystemExit("ChiTiet is missing columns: %s" % missing)
    acc = defaultdict(list)
    for r in rows[1:]:
        if not r or not r[col["Dataset"]]:
            continue
        key = (r[col["Dataset"]], int(float(r[col["K"]])), r[col["Method"]])
        acc[key].append((float(r[col["I_h_eval (Eq 18)"]]), float(r[col["Select time (s)"]])))
    return {k: (sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v), len(v))
            for k, v in acc.items()}


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_GREY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="#404040", labelsize=10, length=3, width=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set_xlim(KS[0] - 0.6, KS[-1] + 0.6)
    ax.xaxis.set_major_locator(FixedLocator(KS))
    ax.set_xlabel("K", fontsize=11, fontfamily=SANS)


def _log_limits(values):
    lo = 10 ** math.floor(math.log10(min(values) * 0.8))
    hi = 10 ** math.ceil(math.log10(max(values) * 1.25))
    if hi / lo < 100:          # at least two decades, like the paper's panel
        lo /= 10
    return lo, hi


def _caption(ax, text):
    ax.text(0.5, -0.24, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=13, fontfamily=SERIF)


def plot_dataset(means, key, name, out_dir):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.3))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.27, wspace=0.26)

    times = []
    for method, style in METHODS:
        spread = [means[(key, k, method)][0] for k in KS]
        secs = [means[(key, k, method)][1] for k in KS]
        times += secs
        common = dict(label=method, linewidth=1.8, markeredgecolor=style["color"], **style)
        ax_a.plot(KS, spread, **common)
        ax_b.plot(KS, secs, **common)

    # (a) influence spread -- linear, from 0 like the paper
    _style_axes(ax_a)
    ax_a.set_ylim(bottom=0)
    ax_a.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
    ax_a.set_ylabel("Influence spread", fontsize=11, fontfamily=SANS)
    ax_a.legend(loc="lower right", frameon=False, fontsize=10, handlelength=2.6)
    _caption(ax_a, "(a) Influence spread")
    spreads = [[means[(key, k, m)][0] for m, _ in METHODS] for k in KS]
    if all((max(s) - min(s)) <= 1e-3 * max(s) for s in spreads):
        ax_a.text(0.03, 0.97, "All three methods give the same spread at every K",
                  transform=ax_a.transAxes, ha="left", va="top", fontsize=9.5,
                  color="#595959", fontfamily=SANS)

    # (b) running time -- log scale, decade ticks as 10^n
    _style_axes(ax_b)
    ax_b.set_yscale("log")
    ax_b.set_ylim(*_log_limits(times))
    ax_b.yaxis.set_major_locator(LogLocator(base=10))
    ax_b.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax_b.yaxis.set_minor_formatter(NullFormatter())
    ax_b.set_ylabel("Time (Sec)", fontsize=11, fontfamily=SANS)
    ax_b.legend(loc="best", frameon=False, fontsize=10, handlelength=2.6)
    _caption(ax_b, "(b) Running time")

    fig.text(0.01, 0.015, "Performance comparison experiments on %s dataset" % name,
             fontsize=11, fontfamily=SERIF, ha="left", va="bottom")

    path = os.path.join(out_dir, "%s_spread_time.png" % key)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def write_table(means, out_dir):
    path = os.path.join(out_dir, "k_sweep_means.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "K", "method", "mean_I_MIA", "mean_select_s", "n_items"])
        for key, _ in DATASETS:
            for k in KS:
                for method, _ in METHODS:
                    i, t, n = means[(key, k, method)]
                    w.writerow([key, k, method, "%.4f" % i, "%.4f" % t, n])
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--xlsx", required=True, help="K-sweep workbook with a ChiTiet sheet")
    p.add_argument("--out", default=os.path.join("results", "figures", "k_sweep"))
    args = p.parse_args(argv)

    means = load_means(args.xlsx)
    missing = [(d, k, m) for d, _ in DATASETS for k in KS for m, _ in METHODS
               if (d, k, m) not in means]
    if missing:
        raise SystemExit("no ChiTiet rows for: %s" % missing)

    os.makedirs(args.out, exist_ok=True)
    for key, name in DATASETS:
        print(plot_dataset(means, key, name, args.out))
    print(write_table(means, args.out))


if __name__ == "__main__":
    main()
