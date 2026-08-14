"""Export run_ctim_global.py logs to a real .xlsx workbook.

Standard library only (Python 3.9): an .xlsx file is a ZIP of XML parts, so
``zipfile`` plus hand-written XML is enough -- no openpyxl, no pandas.

Usage
-----
    python scripts/export_xlsx.py \
        --log yelp2.txt --log digg_items_K20.log \
        --methods CTIM,CTIM-G \
        --out results/ctim_vs_ctimg.xlsx

Each ``--log`` is a stdout capture of ``scripts/run_ctim_global.py``.  The
parser keys off that driver's exact print formats; if those change, this
breaks loudly (a sheet comes out empty) rather than silently.

Sheets produced
---------------
    <dataset>      one per log: spread, gap, timing and CTIM-G phases per item
    Complexity     the textbook cost of each method + the primitives measured
    Notes          the caveats that must travel with these numbers
"""

import argparse
import os
import re
import zipfile

# ---------------------------------------------------------------------------
# 1.  Parsing run_ctim_global.py stdout
# ---------------------------------------------------------------------------

_RE_HEADER = re.compile(
    r"K=(?P<K>\d+)\s+h_sel=(?P<h>[\d.eE+-]+)\s+h_eval=(?P<he>[\d.eE+-]+)\s+seed=(?P<seed>\d+)")
_RE_GRAPH = re.compile(r"U=(?P<n>\d+)\s+E=(?P<m>\d+)\s+C=(?P<C>\d+)")
_RE_ITEMS = re.compile(r"test_items=\[(?P<items>[^\]]*)\]")
_RE_ITEM_HDR = re.compile(r"^# ITEM (?P<i>\d+) of (?P<n>\d+) -- test_items\[(?P<idx>\d+)\] = (?P<item>\d+)")
# anchored to the [setup] prefix: the "item N done in ..." line also names
# Eq (12), and an unanchored pattern would swallow it.
_RE_EQ12 = re.compile(r"\[setup\] Eq \(12\) weights\s+(?P<s>[\d.]+)s")
_RE_METHOD = re.compile(r"^--- (?P<name>.+?) ---\s*$")
_RE_PHASES = re.compile(
    r"phases:\s+mia (?P<mia>[\d.]+)s\s+solo (?P<solo>[\d.]+)s\s+curves (?P<curves>[\d.]+)s"
    r"\s+dp (?P<dp>[\d.]+)s\s+realise (?P<realise>[\d.]+)s\s+repair (?P<repair>[\d.]+)s")
_RE_COMM = re.compile(r"communities used:\s+(?P<used>\d+)\s*/\s*(?P<tot>\d+)")
_RE_DPVAL = re.compile(
    r"dp_value (?P<dp>[\d.]+) is an UPPER BOUND.*?realised (?P<real>[\d.]+), "
    r"independence gap (?P<gap>[\d.]+)%")
_RE_SCORING = re.compile(r"\[scoring\] done in (?P<s>[\d.]+)s")
_RE_ROW = re.compile(
    r"^(?P<m>\S+)\s+(?P<I>[-\d.]+)\s+(?P<gap>[-\d.]+)%\s+"
    r"(?P<mc>--|[-\d.]+\s*\+-\s*[-\d.]+)\s+"
    r"(?P<sec>[\d.]+|\(injected\))\s+(?P<n>\d+)\s*$")
_RE_HIST_ROW = re.compile(r"^\s{4,}(?P<m>\S+)\s+(?P<I>[-\d.]+)\s*$")
_RE_ARB = re.compile(r"MEASURED: (?P<n>\d+) arborescences built, mean size (?P<sz>[\d.]+)")
_RE_EVALS = re.compile(r"MEASURED: (?P<c>\d+) curve gain evals, (?P<r>\d+) realisation gain evals")
_RE_COST = re.compile(r"^\s+cost : (?P<cost>.+?)\s*$")
_RE_ITEM_DONE = re.compile(r"item (?P<item>\d+) done in (?P<s>[\d.]+)s")
_RE_TOTAL = re.compile(r"^TOTAL (?P<s>[\d.]+)s")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _read_text(path):
    """Read a log whatever its encoding.

    PowerShell's ``*>`` redirection writes UTF-16 LE with a BOM, while ``tee``
    on macOS/Linux writes UTF-8; both show up in this project.  Sniff the BOM
    rather than guessing from the platform.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    for bom, enc in ((b"\xff\xfe\x00\x00", "utf-32-le"), (b"\x00\x00\xfe\xff", "utf-32-be"),
                     (b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"),
                     (b"\xef\xbb\xbf", "utf-8-sig")):
        if raw.startswith(bom):
            return raw.decode(enc, "replace")
    return raw.decode("utf-8", "replace")


def parse_run_log(path):
    """Return {'meta': {...}, 'items': [ {...}, ... ]} for one driver log."""
    lines = _read_text(path).splitlines()

    meta = {"log": os.path.basename(path)}
    items = []
    cur = None            # current item dict
    cur_method = None     # method whose "--- X ---" block we are inside
    in_hist = False       # inside the historical-column block
    in_results = False    # inside the [R] results table
    cost_for = None       # method whose complexity block we are inside

    for ln in lines:
        m = _RE_HEADER.search(ln)
        if m and "K" not in meta:
            meta.update(K=int(m.group("K")), h_sel=_f(m.group("h")),
                        h_eval=_f(m.group("he")), seed=int(m.group("seed")))
        m = _RE_GRAPH.search(ln)
        if m and "n" not in meta:
            meta.update(n=int(m.group("n")), m=int(m.group("m")), C=int(m.group("C")))
        m = _RE_ITEMS.search(ln)
        if m and "test_items" not in meta:
            meta["test_items"] = [int(x) for x in m.group("items").replace(" ", "").split(",") if x]
        m = _RE_TOTAL.search(ln)
        if m:
            meta["total_seconds"] = _f(m.group("s"))

        m = _RE_ITEM_HDR.search(ln)
        if m:
            cur = {"run": int(m.group("i")), "index": int(m.group("idx")),
                   "item": int(m.group("item")), "methods": {}, "cost": {}}
            items.append(cur)
            cur_method = None
            in_hist = in_results = False
            cost_for = None
            continue

        if cur is None:
            # single-item logs have no "# ITEM" banner; synthesise one lazily
            if ln.startswith("--- ") or _RE_EQ12.search(ln):
                cur = {"run": 1, "index": 0, "item": (meta.get("test_items") or [0])[0],
                       "methods": {}, "cost": {}}
                items.append(cur)
            else:
                continue

        m = _RE_EQ12.search(ln)
        if m:
            cur["eq12_seconds"] = _f(m.group("s"))
            continue
        m = _RE_SCORING.search(ln)
        if m:
            cur["scoring_seconds"] = _f(m.group("s"))
            continue
        m = _RE_ITEM_DONE.search(ln)
        if m:
            cur["item_seconds"] = _f(m.group("s"))
            continue

        m = _RE_METHOD.match(ln)
        if m:
            cur_method = m.group("name")
            cur["methods"].setdefault(cur_method, {})
            continue

        if cur_method:
            d = cur["methods"][cur_method]
            m = _RE_PHASES.search(ln)
            if m:
                for k in ("mia", "solo", "curves", "dp", "realise", "repair"):
                    d["phase_" + k] = _f(m.group(k))
                continue
            m = _RE_COMM.search(ln)
            if m:
                d["communities_used"] = int(m.group("used"))
                d["communities_total"] = int(m.group("tot"))
                continue
            m = _RE_DPVAL.search(ln)
            if m:
                d["dp_value"] = _f(m.group("dp"))
                d["realised"] = _f(m.group("real"))
                d["independence_gap_pct"] = _f(m.group("gap"))
                continue

        if "[R] RESULTS" in ln:
            in_results, in_hist, cur_method = True, False, None
            continue
        if "historical column" in ln:
            in_results, in_hist = False, True
            continue
        if "seed overlap" in ln:
            in_hist = False
            continue
        if "[C] COMPLEXITY" in ln:
            in_results = in_hist = False
            continue

        if in_results:
            m = _RE_ROW.match(ln.strip())
            if m:
                d = cur["methods"].setdefault(m.group("m"), {})
                d["I"] = _f(m.group("I"))
                d["gap_pct"] = _f(m.group("gap"))
                d["select_seconds"] = _f(m.group("sec"))
                d["n_seeds"] = int(m.group("n"))
                mc = m.group("mc")
                if mc != "--":
                    a, b = mc.split("+-")
                    d["mc_ic"], d["mc_se"] = _f(a), _f(b)
            continue

        if in_hist:
            m = _RE_HIST_ROW.match(ln)
            if m and m.group("m") in cur["methods"]:
                cur["methods"][m.group("m")]["I_h_sel"] = _f(m.group("I"))
            continue

        # complexity block: "  CTIM-G" then indented cost/MEASURED lines
        stripped = ln.strip()
        if stripped and ln.startswith("  ") and not ln.startswith("   ") \
                and stripped in cur["methods"]:
            cost_for = stripped
            continue
        if cost_for:
            m = _RE_COST.match(ln)
            if m:
                cur["cost"][cost_for] = m.group("cost")
                continue
            m = _RE_ARB.search(ln)
            if m:
                cur["methods"][cost_for]["arborescences"] = int(m.group("n"))
                cur["methods"][cost_for]["mean_arb_size"] = _f(m.group("sz"))
                continue
            m = _RE_EVALS.search(ln)
            if m:
                cur["methods"][cost_for]["curve_gain_evals"] = int(m.group("c"))
                cur["methods"][cost_for]["realise_gain_evals"] = int(m.group("r"))
                continue

    return {"meta": meta, "items": items}


# ---------------------------------------------------------------------------
# 2.  Minimal .xlsx writer
# ---------------------------------------------------------------------------
#
# Style indices used below (see _STYLES):
#   0 text   1 header   2 0.0000   3 seconds   4 +0.000%   5 bold
#   6 0.0    7 #,##0    8 wrapped text

TEXT, HEAD, NUM4, SECS, PCT, BOLD, NUM1, INT, WRAP = range(9)

_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<numFmts count="4">
<numFmt numFmtId="164" formatCode="0.0000"/>
<numFmt numFmtId="165" formatCode="0.00&quot;s&quot;"/>
<numFmt numFmtId="166" formatCode="+0.000&quot;%&quot;;-0.000&quot;%&quot;;0"/>
<numFmt numFmtId="167" formatCode="0.0"/>
</numFmts>
<fonts count="2">
<font><sz val="11"/><name val="Calibri"/></font>
<font><b/><sz val="11"/><name val="Calibri"/></font>
</fonts>
<fills count="3">
<fill><patternFill patternType="none"/></fill>
<fill><patternFill patternType="gray125"/></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFE8EEF7"/><bgColor indexed="64"/></patternFill></fill>
</fills>
<borders count="2">
<border><left/><right/><top/><bottom/><diagonal/></border>
<border><left/><right/><top/><bottom style="thin"><color rgb="FF9BA7B4"/></bottom><diagonal/></border>
</borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="9">
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf>
<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
<xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
<xf numFmtId="166" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
<xf numFmtId="167" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
<xf numFmtId="3" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
</cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _col(i):
    """0 -> A, 25 -> Z, 26 -> AA."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _sheet_xml(rows, widths, freeze_row=0):
    """rows: list of list of (value, style).  value None -> blank cell."""
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">']
    if widths:
        out.append("<cols>")
        for i, w in enumerate(widths):
            out.append('<col min="%d" max="%d" width="%g" customWidth="1"/>' % (i + 1, i + 1, w))
        out.append("</cols>")
    if freeze_row:
        out.append('<sheetViews><sheetView workbookViewId="0">'
                   '<pane ySplit="%d" topLeftCell="A%d" activePane="bottomLeft" state="frozen"/>'
                   '</sheetView></sheetViews>' % (freeze_row, freeze_row + 1))
    out.append("<sheetData>")
    for r, row in enumerate(rows, start=1):
        out.append('<row r="%d">' % r)
        for c, cell in enumerate(row):
            val, style = cell if isinstance(cell, tuple) else (cell, TEXT)
            if val is None or val == "":
                continue
            ref = "%s%d" % (_col(c), r)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out.append('<c r="%s" s="%d"><v>%s</v></c>' % (ref, style, repr(val)))
            else:
                out.append('<c r="%s" s="%d" t="inlineStr"><is><t xml:space="preserve">%s</t></is></c>'
                           % (ref, style, _esc(val)))
        out.append("</row>")
    out.append("</sheetData></worksheet>")
    return "".join(out)


def write_xlsx(path, sheets):
    """sheets: list of (name, rows, widths, freeze_row)."""
    n = len(sheets)
    ct = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
          '<Default Extension="xml" ContentType="application/xml"/>',
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
          '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    for i in range(n):
        ct.append('<Override PartName="/xl/worksheets/sheet%d.xml" '
                  'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' % (i + 1))
    ct.append("</Types>")

    wb = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>']
    for i, (name, _r, _w, _f) in enumerate(sheets):
        wb.append('<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (_esc(name[:31]), i + 1, i + 1))
    wb.append("</sheets></workbook>")

    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
    for i in range(n):
        rels.append('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/'
                    '2006/relationships/worksheet" Target="worksheets/sheet%d.xml"/>' % (i + 1, i + 1))
    rels.append('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/'
                '2006/relationships/styles" Target="styles.xml"/>' % (n + 1))
    rels.append("</Relationships>")

    root = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')

    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(ct))
        z.writestr("_rels/.rels", root)
        z.writestr("xl/workbook.xml", "".join(wb))
        z.writestr("xl/_rels/workbook.xml.rels", "".join(rels))
        z.writestr("xl/styles.xml", _STYLES)
        for i, (_n, rows, widths, freeze) in enumerate(sheets):
            z.writestr("xl/worksheets/sheet%d.xml" % (i + 1), _sheet_xml(rows, widths, freeze))


# ---------------------------------------------------------------------------
# 3.  Sheet builders
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def dataset_sheet(run, methods):
    """One sheet per log: per-item spread, gap, timing, CTIM-G phases."""
    meta, items = run["meta"], run["items"]
    rows = [
        [("%s -- K=%s, h_sel=%s, h_eval=%s, seed=%s"
          % (meta.get("log"), meta.get("K"), meta.get("h_sel"),
             meta.get("h_eval"), meta.get("seed")), BOLD)],
        [("U=%s  E=%s  C=%s   test_items=%s"
          % (meta.get("n"), meta.get("m"), meta.get("C"),
             meta.get("test_items", [i["item"] for i in items])), TEXT)],
        [],
    ]
    if meta.get("h_sel") == meta.get("h_eval"):
        rows.append([("CIRCULAR: graded at the same threshold used to select "
                      "(the paper's protocol). CTIM-G maximises this objective "
                      "directly, so its gap is an upper bound.", TEXT)])
        rows.append([])

    hdr = ["#", "item", "method", "I @ h_eval", "gap", "I @ h_sel", "select",
           "Eq (12)", "scoring", "item total", "|S|", "MC-IC", "SE"]
    rows.append([(h, HEAD) for h in hdr])
    freeze = len(rows)

    for it in items:
        base_I = it["methods"].get(methods[0], {}).get("I")
        for name in methods:
            d = it["methods"].get(name)
            if not d:
                continue
            # recompute the gap from the spreads: the driver prints it to 2
            # decimals, which hides differences smaller than 0.01 pp.
            gap = (100.0 * (d["I"] / base_I - 1.0)
                   if (base_I and d.get("I") is not None) else d.get("gap_pct"))
            rows.append([
                (it["run"], INT), (it["item"], INT), (name, TEXT),
                (d.get("I"), NUM4), (gap, PCT), (d.get("I_h_sel"), NUM4),
                (d.get("select_seconds"), SECS), (it.get("eq12_seconds"), SECS),
                (it.get("scoring_seconds"), SECS), (it.get("item_seconds"), SECS),
                (d.get("n_seeds"), INT), (d.get("mc_ic"), NUM4), (d.get("mc_se"), NUM4),
            ])

    rows.append([])
    rows.append([("mean", BOLD)])
    for name in methods:
        sel = _mean([it["methods"].get(name, {}).get("select_seconds") for it in items])
        spr = _mean([it["methods"].get(name, {}).get("I") for it in items])
        gaps = []
        for it in items:
            b = it["methods"].get(methods[0], {}).get("I")
            v = it["methods"].get(name, {}).get("I")
            if b and v is not None:
                gaps.append(100.0 * (v / b - 1.0))
        gap = _mean(gaps)
        rows.append([None, None, (name, TEXT), (spr, NUM4), (gap, PCT), None, (sel, SECS)])

    base = methods[0]
    b = _mean([it["methods"].get(base, {}).get("select_seconds") for it in items])
    for name in methods[1:]:
        s = _mean([it["methods"].get(name, {}).get("select_seconds") for it in items])
        if b and s:
            rows.append([None, None, ("%s / %s wall clock" % (name, base), TEXT),
                         (s / b, NUM1), ("x", TEXT)])

    # ---- CTIM-G phase breakdown -------------------------------------------
    phased = [n for n in methods if any(it["methods"].get(n, {}).get("phase_curves")
                                        is not None for it in items)]
    if phased:
        rows.append([])
        rows.append([("Phase breakdown (seconds, and % of that method's select time)", BOLD)])
        ph = ["#", "item", "method", "mia", "solo", "curves", "dp", "realise",
              "repair", "select", "unattributed", "curves %"]
        rows.append([(h, HEAD) for h in ph])
        for it in items:
            for name in phased:
                d = it["methods"].get(name, {})
                if d.get("phase_curves") is None:
                    continue
                parts = [d.get("phase_" + k) or 0.0
                         for k in ("mia", "solo", "curves", "dp", "realise", "repair")]
                sel = d.get("select_seconds")
                rows.append([
                    (it["run"], INT), (it["item"], INT), (name, TEXT),
                    (parts[0], SECS), (parts[1], SECS), (parts[2], SECS),
                    (parts[3], SECS), (parts[4], SECS), (parts[5], SECS),
                    (sel, SECS),
                    ((sel - sum(parts)) if sel else None, SECS),
                    ((100.0 * parts[2] / sel) if sel else None, NUM1),
                ])

    rows.append([])
    rows.append([("TOTAL for the whole log", TEXT), (meta.get("total_seconds"), SECS)])

    widths = [4, 9, 16, 13, 10, 13, 10, 10, 10, 11, 6, 11, 8]
    return (rows, widths, freeze)


def complexity_sheet(runs, methods):
    rows = [
        [("Complexity -- textbook cost and the primitives actually measured", BOLD)],
        [],
    ]
    hdr = ["dataset", "n", "m", "C", "K", "method", "cost", "arborescences",
           "mean size", "curve evals", "realise evals", "mean select", "ms / arb"]
    rows.append([(h, HEAD) for h in hdr])
    freeze = len(rows)

    for run in runs:
        meta, items = run["meta"], run["items"]
        tag = os.path.splitext(meta.get("log", "?"))[0]
        for name in methods:
            costs = [it["cost"].get(name) for it in items if it["cost"].get(name)]
            arb = _mean([it["methods"].get(name, {}).get("arborescences") for it in items])
            sz = _mean([it["methods"].get(name, {}).get("mean_arb_size") for it in items])
            ce = _mean([it["methods"].get(name, {}).get("curve_gain_evals") for it in items])
            re_ = _mean([it["methods"].get(name, {}).get("realise_gain_evals") for it in items])
            sel = _mean([it["methods"].get(name, {}).get("select_seconds") for it in items])
            rows.append([
                (tag, TEXT), (meta.get("n"), INT), (meta.get("m"), INT),
                (meta.get("C"), INT), (meta.get("K"), INT), (name, TEXT),
                (costs[0] if costs else "", WRAP),
                (arb, INT), (sz, NUM1), (ce, INT), (re_, INT), (sel, SECS),
                ((1000.0 * sel / arb) if (sel and arb) else None, NUM4),
            ])

    rows.append([])
    for line in [
        "CTIM has no measured row: ctim_select_seeds builds its MIA objects "
        "internally, so the driver cannot read the memo dicts back as counters. "
        "Every CTIM/CTIM-G ratio here is wall clock with no primitive count behind it.",
        "Arborescence count is NOT a reliable cost proxy. Measured separately, "
        "CTIM-G builds ~17% more arborescences than GlobalGreedy, of larger mean "
        "size, yet runs 4.8x faster. The [C] section of run_ctim_global.py claims "
        "\"(count, mean size) IS the empirical cost\"; that claim is refuted.",
    ]:
        rows.append([(line, WRAP)])

    widths = [18, 10, 11, 6, 5, 15, 62, 14, 10, 12, 13, 12, 10]
    return (rows, widths, freeze)


_NOTES = [
    ("Grading protocol", BOLD),
    ("The paper uses ONE h = 0.1, for both selection (inside MIA) and for the "
     "reported Eq (18) spread. It does not separate h_sel from h_eval. Any sheet "
     "here whose h_sel equals h_eval reproduces that protocol exactly.", WRAP),
    ("That protocol is valid for the paper's own baselines (CTIM, CTIM_CGA, "
     "AIR+CGA, CINEMA, Greedy) -- none of them maximises Eq (18) directly. It "
     "stops being a comparison once a method that DOES maximise it is added. "
     "CTIM-G and GlobalGreedy are such methods, and they were added by this "
     "project, not by the paper. So the circularity is ours to declare.", WRAP),
    ("", TEXT),
    ("Why a finer h_eval does not help on Yelp", BOLD),
    ("Measured: h_eval = 0.05 returns byte-identical spreads to h = 0.1 on Yelp. "
     "All edges are live at h = 0.1, while 2-hop paths sit at pp <= 1/Z^2 ~ 0.0156, "
     "so the band [0.05, 0.1) is empty and lowering the threshold admits nothing. "
     "Only MC-IC, or h_eval <= 0.01, can referee Yelp independently. Neither has "
     "been run.", WRAP),
    ("On Digg the two thresholds DO differ, because only ~23.7% of edges are live "
     "at h = 0.1, so many 1-hop edges fall in [0.05, 0.1). There CTIM-G wins by "
     "+0.33% at h = 0.1 and loses by -1.00% at h = 0.05.", WRAP),
    ("", TEXT),
    ("What the multi-item runs do and do not establish", BOLD),
    ("Across items the gap has sd = 0.00 on both datasets. That is reproducibility, "
     "NOT robustness: the items produce nearly the same diffusion graph. Evidence "
     "from the logs -- identical allocation, identical arborescence count, dp_value "
     "differing only in the 4th significant digit.", WRAP),
    ("Cause, measured on data/processed/_calib_model.pkl: theta_bar_i[c] = "
     "sum_z P(z|i) * theta[c][z] is the only item-dependent term in Eq (12), and "
     "theta is near-uniform, so theta_bar_i[c] ~ 1/Z = 0.125 for every community "
     "and every item. Median spread across items 0.052%, max 0.268%.", WRAP),
    ("Consequence: any results table that averages over items to raise confidence "
     "is empty for the same reason.", WRAP),
    ("", TEXT),
    ("Reproduce", BOLD),
    ("python3 -u scripts/run_ctim_global.py --dataset data/processed/yelp "
     "--cache data/processed/_yelp_model.pkl --K 20 --h 0.1 --paper-eval "
     "--items all --methods \"CTIM,CTIM-G\" --mc 0 --probe-sample 0", WRAP),
    ("python scripts/export_xlsx.py --log yelp2.txt --log digg_items_K20.log "
     "--methods CTIM,CTIM-G --out results/ctim_vs_ctimg.xlsx", WRAP),
]


def notes_sheet():
    return ([[c] for c in _NOTES], [118], 0)


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--log", action="append", default=[],
                   help="run_ctim_global.py stdout capture; repeatable")
    p.add_argument("--methods", default="CTIM,CTIM-G",
                   help="comma list, in the order they should appear")
    p.add_argument("--out", default=os.path.join("results", "ctim_vs_ctimg.xlsx"))
    args = p.parse_args(argv)

    if not args.log:
        p.error("give at least one --log")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    runs = []
    for path in args.log:
        run = parse_run_log(path)
        n_rows = sum(1 for it in run["items"] for m in methods
                     if it["methods"].get(m, {}).get("I") is not None)
        print("[parse] %-24s  %d item(s), %d data row(s)"
              % (os.path.basename(path), len(run["items"]), n_rows))
        if not n_rows:
            print("        WARNING: nothing parsed -- is this a run_ctim_global.py log?")
        runs.append(run)

    sheets = []
    for run in runs:
        name = os.path.splitext(run["meta"].get("log", "run"))[0][:31]
        rows, widths, freeze = dataset_sheet(run, methods)
        sheets.append((name, rows, widths, freeze))
    rows, widths, freeze = complexity_sheet(runs, methods)
    sheets.append(("Complexity", rows, widths, freeze))
    rows, widths, freeze = notes_sheet()
    sheets.append(("Notes", rows, widths, freeze))

    write_xlsx(args.out, sheets)
    print("[write] %s   (%d sheets)" % (args.out, len(sheets)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
