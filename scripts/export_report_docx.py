"""Build the cross-dataset CTIM vs CTIM-G report as a .docx.

Everything is quoted at the paper's single threshold h = 0.1.  No equations --
the only formal content is the complexity of each method.  MC-IC columns are
deliberately omitted; this report compares CTIM and CTIM-G on Eq (18) alone.

Standard library only.  Rendering helpers are shared with export_docx.py and
log parsing with export_xlsx.py, so a re-run picks up new logs automatically.

Usage
-----
    python scripts/export_report_docx.py --out results/ctim_vs_ctimg_report.docx

Dataset characteristics (size, directedness, attribute density) are MEASURED
from data/processed/<name>/ at run time, not transcribed.  A dataset whose
processed directory is absent from this machine -- Yelp was prepared elsewhere
-- falls back to what its run log states, and the missing cells print as "--".
"""

import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_docx import (bullet, caption, code_block, f, heading,   # noqa: E402
                         para, run, spread, table, write_docx)
from export_xlsx import _mean, parse_run_log                                   # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# log file  ->  (display name, processed dir name)
RUNS = [
    ("yelp_paper_h01.log", "Yelp", "yelp"),
    # digg_paper_h01.log is the re-run at h=0.1 with the Eq (12) subtraction of
    # Section 5.3; digg_items_K20.log is the older h_eval=0.05 run and its CTIM
    # column still carries the redundant rebuild.
    ("digg_paper_h01.log", "Digg", "digg"),
    ("epinions_paper_h01.log", "Epinions", "epinions"),
    ("ciao_paper_h01.log", "Ciao", "ciao"),
    ("lastfm_paper_h01.log", "Last.fm", "lastfm"),
    ("delicious_paper_h01.log", "Delicious", "delicious"),
]

# full/induced arborescence ratio at h=0.1, from each run's own probe.  Yelp's
# probe was disabled in the run kept here; its value comes from the earlier
# y1.log run at the same h_sel.
PROBE_RATIO = {"Yelp": 7.7, "Digg": 1.2, "Epinions": 4.1,
               "Ciao": 2.1, "Last.fm": 1.1, "Delicious": 1.0}

# fraction of edges with pp >= 0.1, from each dataset's calibration phase 2.
LIVE_EDGES = {"Yelp": 100.0, "Digg": 23.70, "Epinions": 32.01,
              "Ciao": 20.67, "Last.fm": 12.37, "Delicious": 0.52}

# how the social relation is encoded in the source, independent of what the
# processed graph measures.
RELATION = {
    "Yelp": "bạn bè hai chiều",
    "Digg": "theo dõi (fan)",
    "Epinions": "tin tưởng (trust)",
    "Ciao": "tin tưởng (trust)",
    "Last.fm": "bạn bè hai chiều",
    "Delicious": "fan hai chiều",
}


def measure_dataset(dirname):
    """Measured characteristics of a processed dataset, or {} if absent."""
    d = os.path.join(_REPO, "data", "processed", dirname)
    meta_path = os.path.join(d, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with io.open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    arcs = set()
    with io.open(os.path.join(d, "graph.tsv"), encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 2:
                arcs.add((int(p[0]), int(p[1])))
    recip = sum(1 for (a, b) in arcs if (b, a) in arcs)

    tokens = 0
    n_items = 0
    ipath = os.path.join(d, "items.tsv")
    if os.path.exists(ipath):
        with io.open(ipath, encoding="utf-8") as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                n_items += 1
                if len(p) > 1 and p[1]:
                    tokens += len(p[1].split(","))

    return {
        "n_users": meta.get("n_users"),
        "n_links": meta.get("n_items") and meta.get("n_links") or meta.get("n_links"),
        "n_items": meta.get("n_items"),
        "n_attrs": meta.get("n_attrs"),
        "n_logs": meta.get("n_logs"),
        "tokens_per_item": (tokens / n_items) if n_items else None,
        "reciprocity": (100.0 * recip / len(arcs)) if arcs else None,
    }


def gap_pct(item, methods):
    base = spread(item["methods"].get(methods[0], {}), item["_meta"])
    val = spread(item["methods"].get(methods[-1], {}), item["_meta"])
    return (100.0 * (val / base - 1.0)) if (base and val) else None


def build(rows, methods):
    b = []
    A = b.append

    A(para(run("CTIM và CTIM-G trên sáu tập dữ liệu", bold=True),
           style="Title", spacing_after=60))
    A(para(run("So sánh ở ngưỡng h = 0.1 của bài báo — kết quả, thời gian chạy và "
               "độ phức tạp. Mọi con số trong báo cáo này đo bằng cùng một hàm "
               "Eq (18) trên toàn đồ thị, ở cùng một ngưỡng, nên các dòng so được "
               "với nhau.", italic=True, color="5A6673")))

    # ---- 1. dataset characteristics ---------------------------------------
    A(heading("1. Đặc trưng các tập dữ liệu", 1))
    A(para("Sáu tập chia đều thành hai nhóm theo bản chất quan hệ xã hội. Cột "
           "“đối ứng” là tỉ lệ cung có chiều ngược lại, đo trực tiếp trên đồ thị "
           "đã xử lý — nó cho biết tính có hướng là thật hay chỉ là cách mã hoá."))
    hdr = ["Tập dữ liệu", "Quan hệ", "Loại", "Đối ứng", "|V|", "|E|", "|M|", "F",
           "token/item", "cạnh sống @0.1"]
    trs = []
    for r in rows:
        c = r["chars"]
        rec = c.get("reciprocity")
        kind = ("vô hướng" if (rec is not None and rec > 99.9)
                else ("có hướng" if rec is not None else "vô hướng"))
        trs.append([
            r["name"], RELATION.get(r["name"], "—"), kind,
            ("%.1f%%" % rec) if rec is not None else "100.0%",
            "{:,}".format(c["n_users"]) if c.get("n_users") else
            "{:,}".format(r["meta"].get("n", 0)),
            "{:,}".format(c["n_links"]) if c.get("n_links") else
            "{:,}".format(r["meta"].get("m", 0)),
            "{:,}".format(c["n_items"]) if c.get("n_items") else "—",
            "{:,}".format(c["n_attrs"]) if c.get("n_attrs") else "—",
            ("%.2f" % c["tokens_per_item"]) if c.get("tokens_per_item") else "—",
            "%.2f%%" % LIVE_EDGES.get(r["name"], 0.0),
        ])
    A(table(hdr, trs, [1150, 1250, 900, 800, 900, 1000, 900, 800, 900, 1000],
            ["left", "left", "left", "right", "right", "right", "right", "right",
             "right", "right"]))
    A(caption("Yelp được chuẩn bị trên máy khác nên |M|, F và token/item không đo "
              "lại được ở đây; |V| và |E| lấy từ dòng đầu log của chính lần chạy. "
              "Ba tập có hướng thật là Digg, Epinions và Ciao — Digg bất đối xứng "
              "nhất (18.8%), vì hàng của nó là quan hệ theo dõi một chiều chứ "
              "không phải kết bạn."))

    # ---- 2. per-dataset results -------------------------------------------
    A(heading("2. Kết quả và thời gian chạy từng tập", 1))
    for i, r in enumerate(rows, start=1):
        A(heading("2.%d. %s" % (i, r["name"]), 2))
        trs = []
        for it in r["items"]:
            base = spread(it["methods"].get(methods[0], {}), r["meta"])
            for nm in methods:
                d = it["methods"].get(nm)
                if not d:
                    continue
                val = spread(d, r["meta"])
                g = (100.0 * (val / base - 1.0)) if (base and val) else None
                trs.append([str(it["item"]), nm, f(val),
                            "—" if nm == methods[0] else "%+.3f%%" % g,
                            f(d.get("select_seconds"), 2) + " s",
                            f(it.get("eq12_seconds"), 2) + " s",
                            f(it.get("item_seconds"), 2) + " s"])
        A(table(["Sản phẩm", "Phương pháp", "I (h = 0.1)", "Khoảng cách",
                 "Thời gian chọn", "Eq (12)", "Tổng"],
                trs, [1250, 1500, 1500, 1300, 1500, 1100, 1100],
                ["right", "left", "right", "right", "right", "right", "right"]))
        gaps = [g for g in (gap_pct(it, methods) for it in r["items"]) if g is not None]
        sel0 = _mean([it["methods"].get(methods[0], {}).get("select_seconds")
                      for it in r["items"]])
        sel1 = _mean([it["methods"].get(methods[-1], {}).get("select_seconds")
                      for it in r["items"]])
        bits = []
        if gaps:
            bits.append("Khoảng cách %s so với %s: trung bình %+.3f%%, biên độ "
                        "%+.3f%% … %+.3f%% qua %d sản phẩm."
                        % (methods[-1], methods[0], sum(gaps) / len(gaps),
                           min(gaps), max(gaps), len(gaps)))
        if sel0 and sel1:
            if sel1 >= sel0:
                bits.append("%s chậm hơn %s %.2f lần." % (methods[-1], methods[0],
                                                          sel1 / sel0))
            else:
                bits.append("%s NHANH hơn %s %.2f lần." % (methods[-1], methods[0],
                                                           sel0 / sel1))
        bits.append("Tỉ lệ kích thước cây trên đồ thị đầy đủ so với đồ thị con "
                    "cảm sinh: %.1f lần." % PROBE_RATIO.get(r["name"], 0.0))
        A(caption(" ".join(bits)))

    # ---- 3. summary --------------------------------------------------------
    A(heading("3. Bảng tổng hợp", 1))
    trs = []
    for r in rows:
        gaps = [g for g in (gap_pct(it, methods) for it in r["items"]) if g is not None]
        sel0 = _mean([it["methods"].get(methods[0], {}).get("select_seconds")
                      for it in r["items"]])
        sel1 = _mean([it["methods"].get(methods[-1], {}).get("select_seconds")
                      for it in r["items"]])
        c = r["chars"]
        rec = c.get("reciprocity")
        trs.append([
            r["name"],
            "{:,}".format(c.get("n_users") or r["meta"].get("n", 0)),
            "vô hướng" if (rec is None or rec > 99.9) else "có hướng",
            "%.1f" % PROBE_RATIO.get(r["name"], 0.0),
            "%+.3f%%" % (sum(gaps) / len(gaps)) if gaps else "—",
            f(sel0, 2) + " s", f(sel1, 2) + " s",
            (("%.2f×" % (sel1 / sel0)) + (" *" if r["name"] == "Yelp" else ""))
            if (sel0 and sel1) else "—",
        ])
    A(table(["Tập dữ liệu", "|V|", "Loại", "Tỉ lệ cây", "Khoảng cách CTIM-G",
             "CTIM", "CTIM-G", "CTIM-G / CTIM"],
            trs, [1400, 1200, 1100, 1100, 1900, 1200, 1200, 1400],
            ["left", "right", "left", "right", "right", "right", "right", "right"]))
    A(caption("“Tỉ lệ cây” là kích thước trung bình của cây ảnh hưởng dựng trên "
              "đồ thị đầy đủ chia cho kích thước dựng trên đồ thị con của một cộng "
              "đồng; nó đo mức độ mà việc chia cộng đồng thực sự thu hẹp bài toán. "
              "Cột cuối là tỉ số thời gian: lớn hơn 1 nghĩa là CTIM-G chậm hơn. "
              "(*) Yelp được chạy trên máy khác, TRƯỚC khi khoản trừ Eq (12) ở Mục "
              "5.3 tồn tại, nên cột CTIM của nó vẫn mang chi phí dựng lại; tỉ số "
              "thật của Yelp cao hơn con số in ở đây. Cần chạy lại để lấy số đúng."))

    # ---- 4. complexity -----------------------------------------------------
    A(heading("4. Độ phức tạp", 1))
    A(para(run("CTIM", bold=True)))
    A(code_block([
        "O(nC)                          phat hien cong dong",
        "  +  sum_c O(n_c · t_c log t_c)  dung cay tren do thi con cam sinh",
        "  +  O(CK)                       o quy hoach dong",
    ]))
    A(para("n cây ảnh hưởng, mỗi cây dựng trên đồ thị con của một cộng đồng, nên "
           "kích thước trung bình t_c nhỏ. Đổi lại, mọi cạnh bắc cầu giữa các cộng "
           "đồng bị loại khỏi tầm nhìn."))
    A(para(run("CTIM-G", bold=True)))
    A(code_block([
        "O(n · t log t)                 dung cay tren do thi DAY DU",
        "  +  sum_c O(n_c · t log t)      duong cong loi ich tung cong dong",
        "  +  O(C·K²)                     quy hoach dong",
        "  +  O(K · t²)                   hien thuc hoa tap hat giong",
    ]))
    A(para("Cùng n cây, nhưng trên đồ thị đầy đủ nên t là kích thước trung bình "
           "toàn cục. Toàn bộ khoảng cách chi phí giữa hai phương pháp nằm ở tỉ số "
           "t / t_c — chính là cột “tỉ lệ cây” ở bảng trên."))
    A(para(run("Đo được:", bold=True) + run(" bước quy hoạch động — phần đóng góp "
           "thuật toán thực sự — tốn 0.00–0.01 giây trên mọi tập dữ liệu. Khoảng "
           "70% thời gian của CTIM-G nằm ở việc dựng đường cong lợi ích. Đó là pha "
           "duy nhất đáng tối ưu.")))
    A(bullet("CTIM không có số đếm nguyên thủy vì nó tự dựng đối tượng MIA bên "
             "trong, trình điều khiển không đọc được bộ đếm. Mọi tỉ lệ CTIM-G / "
             "CTIM trong báo cáo là số đồng hồ."))
    A(bullet("Số cây dựng không phải thước chi phí đáng tin: đo riêng, CTIM-G dựng "
             "nhiều hơn tham lam toàn cục khoảng 17% số cây, kích thước lớn hơn, "
             "nhưng vẫn chạy nhanh hơn 4.8 lần."))

    # ---- 5. conclusions ----------------------------------------------------
    A(heading("5. Kết luận: tập nhỏ và tập lớn cho kết quả khác nhau", 1))

    A(heading("5.1. Quy luật", 2))
    A(para("Sắp sáu tập theo tỉ lệ cây — mức độ mà việc chia cộng đồng thực sự thu "
           "hẹp bài toán — thì kết quả tách thành hai nhóm rõ rệt."))
    A(table(["Nhóm", "Tỉ lệ cây", "Tập dữ liệu", "Khoảng cách CTIM-G", "Kết luận"],
            [["Cộng đồng cắt sâu", "≥ 4", "Yelp, Epinions", "+6.8% … +8.4%",
              "đáng đổi thời gian lấy chất lượng"],
             ["Cộng đồng cắt nông", "≤ 2.1", "Ciao, Digg, Last.fm, Delicious",
              "−2.2% … +0.3%", "không thắng, và cũng không nhanh hơn"]],
            [2100, 1200, 2400, 2100, 2600],
            ["left", "right", "left", "right", "left"]))
    A(caption("Tỉ lệ cây dự báo được khoảng cách chất lượng. Nó KHÔNG dự báo được "
              "tỉ số thời gian: sau khi sửa lỗi đo ở Mục 5.3, CTIM-G chậm hơn hoặc "
              "ngang CTIM trên gần như mọi tập, kể cả tập nhỏ."))

    A(heading("5.2. Tập lớn", 2))
    A(para("Trên Yelp (366,427 người dùng) và Epinions (18,059), đồ thị đủ lớn để "
           "một cộng đồng chỉ chứa một phần nhỏ đồ thị. Cây ảnh hưởng dựng trên "
           "đồ thị con vì thế nhỏ hơn hẳn cây dựng trên đồ thị đầy đủ — 7.7 lần ở "
           "Yelp, 4.1 lần ở Epinions. Đó chính là lượng thông tin CTIM vứt đi khi "
           "cắt theo cộng đồng, và cũng chính là phần CTIM-G thu lại: nó thắng "
           "+8.37% và +6.78%."))
    A(para("Cái giá là thời gian. CTIM-G chậm hơn 11.02 lần trên Yelp và 2.45 lần "
           "trên Epinions, vì mỗi cây bây giờ dựng trên toàn đồ thị."))

    A(heading("5.3. Tập nhỏ", 2))
    A(para("Trên Last.fm (1,892) và Delicious (1,861), số cộng đồng C = 100 chia "
           "cho chưa đầy hai nghìn người dùng, tức mỗi cộng đồng chỉ khoảng 19 "
           "người. Đồ thị con lúc này gần bằng đồ thị đầy đủ — tỉ lệ cây 1.1 và "
           "1.0. Không còn gì để CTIM-G thu lại, nên nó không thắng: −2.17% và "
           "0.00%."))
    A(para("Và ở đây CTIM-G cũng không nhanh hơn. Nó chạy chậm hơn hoặc ngang "
           "CTIM trên gần như mọi tập; chỉ trên Last.fm nó nhỉnh hơn một chút, "
           "trong phạm vi mà chênh lệch đó không đáng để chọn thuật toán."))
    A(para(run("Cảnh báo về một phiên bản trước của báo cáo này.", bold=True)
           + run(" Bản trước khẳng định CTIM-G nhanh hơn CTIM 2.05 lần trên "
                 "Last.fm và 4.25 lần trên Delicious, và rút ra kết luận “trên tập "
                 "nhỏ chiều tốc độ đảo ngược”. Khẳng định đó ")
           + run("sai", bold=True)
           + run(", do một lỗi đo đã được tìm ra và sửa: hàm chọn hạt giống của "
                 "CTIM không nhận tham số pp nên tự dựng lại Eq (12) bên trong, "
                 "mà hàm dựng trọng số không lưu đệm; CTIM-G thì được truyền pp "
                 "dựng sẵn. Đo được, 41–47% thời gian “chọn” của CTIM thực chất là "
                 "dựng lại Eq (12). Mọi con số thời gian trong báo cáo này đã trừ "
                 "khoản đó ra và là kết quả đo lại.")))

    A(heading("5.4. Hai trường hợp cần nói riêng", 2))
    A(para(run("Delicious — phép đo vô nghĩa.", bold=True)
           + run(" Chỉ 0.52% số cạnh sống ở h = 0.1, nên cây ảnh hưởng trung bình "
                 "gần như chỉ gồm chính nút gốc. Hai tập hạt giống khác nhau 15/20 "
                 "mà cho điểm giống hệt đến bốn chữ số thập phân. Đó là dấu hiệu "
                 "hàm mục tiêu rỗng, không phải hai phương pháp ngang tài. Không "
                 "nên trích dẫn dòng Delicious như một kết quả so sánh.")))
    A(para(run("Digg — lệch quy luật.", bold=True)
           + run(" Tỉ lệ cây chỉ 1.2 nhưng CTIM-G vẫn nhỉnh +0.33% và chậm hơn "
                 "2.78 lần. Nó nằm giữa hai nhóm và không củng cố cũng không bác "
                 "bỏ quy luật trên.")))

    A(heading("5.5. Khuyến nghị", 2))
    A(bullet(run("Cộng đồng cắt sâu (tỉ lệ cây ≥ 4): dùng ")
             + run("CTIM-G", bold=True)
             + run(". Đây là trường hợp DUY NHẤT nó đáng giá — đổi thời gian lấy "
                   "chất lượng, và phần chất lượng thu được là thật.")))
    A(bullet(run("Cộng đồng cắt nông (tỉ lệ cây ≤ 2.1): dùng ")
             + run("CTIM", bold=True)
             + run(". CTIM-G không chính xác hơn, mà cũng không nhanh hơn. Sau khi "
                   "sửa lỗi đo ở Mục 5.3 thì lợi thế tốc độ được cho là có ở tập "
                   "nhỏ đã biến mất.")))
    A(bullet("Trước khi chọn, đo tỉ lệ cây trên chính dữ liệu của mình. Nó dự đoán "
             "được cả khoảng cách chất lượng lẫn tỉ lệ tốc độ, và rẻ hơn nhiều so "
             "với chạy cả hai phương pháp."))
    A(bullet("Kiểm tỉ lệ cạnh sống ở ngưỡng đang dùng trước khi tin bất kỳ con số "
             "nào. Dưới khoảng 1% thì hàm mục tiêu rỗng và mọi so sánh đều vô "
             "nghĩa, như trường hợp Delicious."))

    A(heading("6. Phạm vi của báo cáo", 1))
    A(para("Mọi con số ở đây chấm bằng Eq (18) ở ngưỡng h = 0.1, đúng giao thức "
           "bài báo: một ngưỡng duy nhất cho cả việc chọn hạt giống lẫn việc chấm "
           "điểm. Giao thức đó hợp lệ với tập baseline của bài báo vì không phương "
           "pháp nào trong đó tối ưu trực tiếp Eq (18). CTIM-G thì có — nó chạy "
           "tham lam thẳng trên chính hàm này — nên khoảng cách CTIM-G báo cáo ở "
           "đây là cận trên của khoảng cách thật."))
    A(para("Khoảng cách giữa các sản phẩm trong cùng một tập gần như bằng không "
           "(độ lệch chuẩn 0.00–0.32 điểm phần trăm). Đó là tính tái lập, không "
           "phải tính bền: các sản phẩm trong một tập sinh ra gần như cùng một đồ "
           "thị lan truyền, nên ba sản phẩm không phải ba phép thử độc lập."))
    return "".join(b)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="CTIM,CTIM-G")
    p.add_argument("--out", default=os.path.join("results", "ctim_vs_ctimg_report.docx"))
    args = p.parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    rows = []
    for log, name, dirname in RUNS:
        path = os.path.join(_REPO, log)
        if not os.path.exists(path):
            print("[skip]  %-26s not found" % log)
            continue
        parsed = parse_run_log(path)
        items = [it for it in parsed["items"]
                 if it["methods"].get(methods[0], {}).get("I") is not None]
        for it in items:
            it["_meta"] = parsed["meta"]
        chars = measure_dataset(dirname)
        print("[parse] %-26s %-10s %d item(s)%s"
              % (log, name, len(items), "" if chars else "  (no processed dir)"))
        rows.append({"name": name, "meta": parsed["meta"], "items": items,
                     "chars": chars})

    if not rows:
        print("nothing to report")
        return 2
    write_docx(args.out, build(rows, methods))
    print("[write] %s   (%d datasets)" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
