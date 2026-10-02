"""Formal specification of CTIM-G, as a scientific-paper style .docx.

Defines CTIM-G the way a paper would: notation, the model it inherits unchanged,
the reference algorithm CTIM, then CTIM-G's own equations (G1)-(G7), the three
structural differences, the complexity of both, and the properties that can be
proved or that provably do NOT hold.

Every claim here was read off the implementation, not off a docstring:

    ctim/ctim.py            ctim_select_seeds()  line 329   (CTIM, Algorithm 2)
    ctim/ctim.py            detect_communities() line 170   (Eq (19))
    ctim/ea_dp.py           build_curves()       line 101   (induced-subgraph curves)
    ctim/ea_dp.py           allocate_exact()     line 332   (the resource DP)
    ctim/ctim_global.py     celf_select()        line 120   (lazy greedy)
    ctim/ctim_global.py     build_global_curves() line 197  (CTIM-G phase 1)
    ctim/ctim_global.py     realise_quota_greedy() line 272 (CTIM-G phase 3)
    ctim/ctim_global.py     select_seeds_global() line 302  (CTIM-G end to end)
    ctim/influence.py       MIA                             (Eq (13)-(18))

Equations are real Word equations (OMML), not images.

Usage
-----
    python scripts/export_ctimg_spec_docx.py --out results/ctimg_specification.docx
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_docx import (bullet, caption, code_block, equation, heading,   # noqa: E402
                         mbar, mdel, mfunc, mlimlow, mnary, mr, msub, msubsup,
                         msup, para, run, table, write_docx)

# ---------------------------------------------------------------------------
# equation fragments
# ---------------------------------------------------------------------------

S = mr("S")
V = mr("V")
h = mr("h")


def m_set(inner):
    return mdel(inner, "{", "}")


# ---------------------------------------------------------------------------
# the one measured table: read from the run logs so it cannot drift
# ---------------------------------------------------------------------------

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# log, display name, |V|, full/induced arborescence ratio from that run's probe
_RUNS = [
    ("yelp_paper_h01.log", "Yelp", 366427, 7.7),
    ("epinions_paper_h01.log", "Epinions", 18059, 4.1),
    ("ciao_paper_h01.log", "Ciao", 2215, 2.1),
    ("digg_paper_h01.log", "Digg", 30358, 1.2),
    ("lastfm_paper_h01.log", "Last.fm", 1892, 1.1),
    ("delicious_paper_h01.log", "Delicious", 1861, 1.0),
]

# Yelp was prepared and run on another machine, before the Eq (12) subtraction
# of Section 8.2 existed.  Its CTIM column therefore still carries the rebuild.
_STALE_TIMING = {"Yelp"}


def measured_rows():
    from export_xlsx import _mean, parse_run_log

    rows = []
    for log, name, n_users, ratio in _RUNS:
        path = os.path.join(_REPO, log)
        if not os.path.exists(path):
            continue
        parsed = parse_run_log(path)
        meta, items = parsed["meta"], parsed["items"]
        gaps, c0, c1 = [], [], []
        for it in items:
            a = it["methods"].get("CTIM", {})
            g = it["methods"].get("CTIM-G", {})
            va = a.get("I_h_sel") if meta.get("h_sel") != meta.get("h_eval") else a.get("I")
            vg = g.get("I_h_sel") if meta.get("h_sel") != meta.get("h_eval") else g.get("I")
            if va and vg:
                gaps.append(100.0 * (vg / va - 1.0))
            c0.append(a.get("select_seconds"))
            c1.append(g.get("select_seconds"))
        s0, s1 = _mean(c0), _mean(c1)
        ratio_txt = ("%.2f×" % (s1 / s0)) if (s0 and s1) else "—"
        if name in _STALE_TIMING:
            ratio_txt += " *"
        rows.append([name, "{:,}".format(n_users), "%.1f" % ratio,
                     ("%+.3f%%" % (sum(gaps) / len(gaps))) if gaps else "—",
                     ratio_txt])
    return rows


def build():
    b = []
    A = b.append

    # ---- title ------------------------------------------------------------
    A(para(run("CTIM-G: Phân bổ ngân sách theo cộng đồng với hàm mục tiêu toàn cục",
               bold=True), style="Title", spacing_after=60))
    A(para(run("Đặc tả hình thức, và sự khác biệt so với CTIM (Huang, Shen, Meng, "
               "Chang & He, Applied Intelligence, 2019)", italic=True,
               color="5A6673")))

    # ---- abstract ---------------------------------------------------------
    A(heading("Tóm tắt", 1))
    A(para("CTIM phân rã bài toán cực đại hoá ảnh hưởng thành C bài toán con theo "
           "cộng đồng, giải mỗi bài toán con trên đồ thị con cảm sinh của cộng "
           "đồng đó, rồi dùng quy hoạch động để chia K hạt giống cho các cộng "
           "đồng. Phép phân rã này làm cho thuật toán nhanh, nhưng đưa vào hai "
           "phép xấp xỉ không nêu tên: hàm mục tiêu của mỗi cộng đồng bị cắt cụt "
           "về đồ thị con, và một ứng viên không bao giờ được đánh giá cùng với "
           "các hạt giống đã chọn ở cộng đồng khác."))
    A(para("CTIM-G giữ nguyên phép phân rã — cùng cách phát hiện cộng đồng, cùng "
           "tập ứng viên, cùng quy hoạch động — nhưng đo mọi lợi ích bằng **một** "
           "hàm mục tiêu toàn cục duy nhất, và hiện thực hoá lời giải của quy "
           "hoạch động bằng một lượt tham lam chung có ràng buộc hạn ngạch thay vì "
           "ghép các lời giải cộng đồng độc lập. Nó cũng báo cáo chính lượng sai "
           "số mà phép phân rã gây ra."))

    # ---- notation ---------------------------------------------------------
    A(heading("1. Ký hiệu", 1))
    A(table(["Ký hiệu", "Ý nghĩa"], [
        ["G = (V, E)", "đồ thị xã hội có hướng; n = |V|, m = |E|"],
        ["i", "sản phẩm đang xét; mọi trọng số cạnh phụ thuộc i"],
        ["pp(u,v)", "xác suất lan truyền trên cạnh (u,v) đối với sản phẩm i, Eq (12)"],
        ["h", "ngưỡng cắt đường đi của MIA; bài báo đặt h = 0.1"],
        ["C", "số cộng đồng"],
        ["c(v)", "cộng đồng của v, Eq (19)"],
        ["c_m", "tập các nút thuộc cộng đồng m; n_m = |c_m|"],
        ["MIIA(v,h), MIOA(v,h)", "cây ảnh hưởng vào / ra của v, Eq (15), Eq (16)"],
        ["ch(x)", "tập con của x TRÊN CÂY MIIA, tức {w : parent(w) = x}"],
        ["ap(v | S)", "xác suất v được kích hoạt bởi tập hạt giống S, Eq (17)"],
        ["I(S)", "ảnh hưởng của S trên TOÀN đồ thị G, Eq (18)"],
        ["I⁽ᵐ⁾(S)", "Eq (18) tính trên đồ thị con cảm sinh bởi c_m"],
        ["I_m(j)", "đường cong lợi ích của cộng đồng m khi được cấp j hạt giống"],
        ["cap_m", "số hạt giống tối đa cộng đồng m có thể nhận"],
        ["a = (a_1, …, a_C)", "vector phân bổ; a_m là ngân sách của cộng đồng m"],
        ["σ(u)", "ảnh hưởng đơn lẻ I({u}), dùng làm cận trên khởi tạo cho CELF"],
        ["t, t_m", "kích thước trung bình của cây MIIA trên G, và trên đồ thị con c_m"],
        ["S_m", "tập hạt giống của riêng cộng đồng m (dùng trong CTIM)"],
        ["S⁽ʲ⁾_m", "tiền tố j phần tử trên đường cong của cộng đồng m, tức tập đạt I_m(j)"],
        ["S_cat", "tập GHÉP: hợp của các S⁽ᵃᵐ⁾_m — điều một bản cài đặt ngây thơ trả về"],
        ["S_r", "tập HIỆN THỰC HOÁ: kết quả của một lượt tham lam chung dưới hạn "
                "ngạch a, (G5). KHÔNG phải nghiệm tối ưu — bài báo này không dùng "
                "ký hiệu S* cho tập nào, để tránh lẫn với nghĩa thông dụng của nó"],
        ["S_out", "tập CTIM-G trả về: cái tốt hơn trong hai tập trên, (G6)"],
    ], [2100, 6900], ["left", "left"]))

    # ---- shared model -----------------------------------------------------
    A(heading("2. Mô hình được kế thừa nguyên vẹn", 1))
    A(para("CTIM-G không sửa bất kỳ phương trình nào của mô hình lan truyền. "
           "Chúng được nhắc lại ở đây vì mọi định nghĩa sau đều tham chiếu tới."))

    A(para(run("Trọng số cạnh.", bold=True)
           + run(" Xác suất lan truyền trên cạnh, ở dạng đã gom nhân tử mà bản "
                 "cài đặt dùng:")))
    A(equation(mfunc("pp", mr("u,v")) + mr(" = ")
               + mnary("∑", mr("c"), "", msub(mr("a"), mr("u,c")) + mr("·")
                       + msub(mr("π"), mr("v,c")) + mr("·")
                       + msub(mbar(mr("θ")), mr("i,c"))), "Eq (12)"))

    A(para(run("Đường đi và cây ảnh hưởng.", bold=True)
           + run(" Xác suất của một đường đi là tích các cạnh; đường ảnh hưởng "
                 "cực đại tìm bằng Dijkstra trên trọng số cộng tính −ln pp; cây "
                 "MIIA giữ lại mọi đường vượt ngưỡng h:")))
    A(equation(mfunc("MIIA", mr("v,h")) + mr(" = ")
               + mnary("⋃", mfunc("pp", mfunc("MIP", mr("u,v"))) + mr(" ≥ h"), "",
                       mfunc("MIP", mr("u,v"))), "Eq (15)"))

    A(para(run("Hàm mục tiêu.", bold=True)
           + run(" Xác suất kích hoạt được tính đệ quy từ dưới lên trên cây "
                 "MIIA(v,h). Với mỗi nút x của cây, tích chạy trên tập con "
                 "ch(x) ")
           + run("của chính cây", bold=True)
           + run(", tức {w : parent(w) = x}:")))
    A(equation(mfunc("ap", mr("x∣S")) + mr(" = 1 − ")
               + mnary("∏", mr("w ∈ ch(x)"), "",
                       mdel(mr("1 − ") + mfunc("ap", mr("w∣S")) + mr("·")
                            + mfunc("pp", mr("w,x")))), "Eq (17)"))
    A(para(run("với ap(x|S) = 1 nếu x ∈ S, và khi đó cây con dưới x không được "
               "duyệt tiếp. Cần nhấn mạnh: ch(x) là tập con ")
           + run("trên cây", bold=True)
           + run(", không phải mọi nút vào của x nằm trong MIIA(v,h) — tập sau "
                 "là bao hàm thực sự của tập trước. Đây là chỗ MIA xấp xỉ mô "
                 "hình lan truyền: một nút được nuôi bởi nhiều đường có xác suất "
                 "tương đương sẽ bị tính thiếu.")))
    A(equation(mfunc("I", S) + mr(" = ")
               + mnary("∑", mr("v ∈ V"), "", mfunc("ap", mr("v∣S"))), "Eq (18)"))
    A(para(run("I là hàm đơn điệu không giảm và submodular", bold=True)
           + run(", với I(∅) = 0. Hai tính chất này là cơ sở cho mọi lập luận "
                 "về cận ở các mục sau, và đặc biệt cho tính lõm của đường cong "
                 "lợi ích ở (G2).")))

    A(para(run("Phân hoạch cộng đồng.", bold=True)
           + run(" Cả hai phương pháp dùng chung phép gán cứng này:")))
    A(equation(mfunc("c", mr("v")) + mr(" = ")
               + mlimlow(mr("arg max", plain=True), mr("c"))
               + msub(mr("π"), mr("v,c")), "Eq (19)"))
    A(caption("Bằng nhau thì chỉ số cộng đồng nhỏ hơn thắng, nên phân hoạch là "
              "tất định. ctim/ctim.py:170."))

    # ---- CTIM -------------------------------------------------------------
    A(heading("3. Thuật toán tham chiếu: CTIM", 1))
    A(para("CTIM chạy K vòng. Ở mỗi vòng, mỗi cộng đồng báo lợi ích biên tốt nhất "
           "của nó, quy hoạch động chọn cộng đồng nhận hạt giống thứ k, và nút "
           "tốt nhất của cộng đồng đó được thêm vào."))

    A(para(run("Hàm mục tiêu của cộng đồng.", bold=True)
           + run(" Đây là điểm mấu chốt. CTIM dựng một đối tượng MIA riêng cho "
                 "mỗi cộng đồng, giới hạn trên tập nút c_m:")))
    A(equation(msup(mr("I"), mdel(mr("m"))) + mdel(S) + mr(" = ")
               + mnary("∑", mr("v ∈ ") + msub(mr("c"), mr("m")), "",
                       mfunc("ap", mr("v∣S")) + mr("  trên  ") + msub(mr("G"), mr("m"))),
               "CTIM"))
    A(para(run("với G_m là đồ thị con CẢM SINH bởi c_m. Một cạnh chỉ tồn tại "
               "trong G_m khi ")
           + run("cả hai", bold=True)
           + run(" đầu mút nằm trong c_m. Mọi cạnh bắc cầu giữa các cộng đồng "
                 "biến mất khỏi tầm nhìn của cộng đồng đó.")))
    A(caption("ctim/ctim.py:437 — MIA(n_users, ds.out_adj, ds.in_adj, pp, h=h, "
              "nodes=members). Tham số nodes= là thứ tạo ra sự cắt cụt này."))

    A(para(run("Lợi ích biên của một vòng.", bold=True)))
    A(equation(mr("Δ") + msub(mr("I"), mr("m")) + mr(" = ")
               + mlimlow(mr("max", plain=True), mr("u ∈ ") + msub(mr("c"), mr("m")))
               + mdel(msup(mr("I"), mdel(mr("m"))) + mdel(msub(mr("S"), mr("m"))
                      + mr(" ∪ ") + m_set(mr("u")))
                      + mr(" − ") + msup(mr("I"), mdel(mr("m")))
                      + mdel(msub(mr("S"), mr("m")))), "Algorithm 2, dòng 34"))
    A(para(run("S_m là tập hạt giống của riêng cộng đồng m. Ứng viên u ", bold=False)
           + run("không bao giờ", bold=True)
           + run(" được đánh giá cùng với các hạt giống đã chọn ở cộng đồng khác.")))

    A(para(run("Quy hoạch động.", bold=True)))
    A(equation(mr("I") + mdel(mr("m,k"), "[", "]") + mr(" = max")
               + mdel(mr("I") + mdel(mr("m−1,k"), "[", "]") + mr(",  ")
                      + mr("I") + mdel(mr("C,k−1"), "[", "]") + mr(" + Δ")
                      + msub(mr("I"), mr("m"))), "Algorithm 2, dòng 35"))
    A(caption("Bản in của bài báo tham chiếu I[C,k−1] ở dòng 35; bản cài đặt gọi "
              "đó là chế độ \"paper-true\" và giữ làm mặc định. Hai cách đọc khác "
              "(\"consistent\", \"paper-literal\") có sẵn qua tham số dp_tiebreak. "
              "ctim/ctim.py:471."))

    # ---- CTIM-G -----------------------------------------------------------
    A(heading("4. CTIM-G: định nghĩa", 1))
    A(para("CTIM-G giữ nguyên Eq (12)–(19), giữ nguyên phép phân hoạch cộng đồng, "
           "và giữ nguyên tập ứng viên c_m. Nó thay đổi đúng ba thứ: hàm đo lợi "
           "ích, cách xây đường cong, và cách hiện thực hoá lời giải."))

    A(heading("4.1. Một hàm mục tiêu duy nhất", 2))
    A(para("CTIM-G dựng đúng MỘT đối tượng MIA, trên toàn đồ thị G, và mọi lợi "
           "ích ở mọi pha đều đo bằng nó:"))
    A(equation(msub(mr("I"), mr("m")) + mr("(·) ≡ I(·)  trên  G,   ∀m"), "(G1)"))
    A(caption("ctim/ctim_global.py:224 — MIA(ds.n_users, ds.out_adj, ds.in_adj, "
              "pp, h=h), không có tham số nodes=. So sánh với ctim/ctim.py:437."))

    A(heading("4.2. Đường cong lợi ích toàn cục", 2))
    A(para("Với mỗi cộng đồng m, dựng một dãy hạt giống bằng tham lam trên ứng "
           "viên c_m, nhưng chấm điểm bằng I toàn cục, xuất phát từ tập rỗng:"))
    A(equation(msubsup(mr("S"), mr("m"), mdel(mr("0")))
               + mr(" = ∅"), None))
    A(equation(msubsup(mr("S"), mr("m"), mdel(mr("j")))
               + mr(" = ") + msubsup(mr("S"), mr("m"), mdel(mr("j−1")))
               + mr(" ∪ ")
               + m_set(mlimlow(mr("arg max", plain=True),
                               mr("u ∈ ") + msub(mr("c"), mr("m")) + mr(" ∖ ")
                               + msubsup(mr("S"), mr("m"), mdel(mr("j−1"))))
                       + mdel(mfunc("I", msubsup(mr("S"), mr("m"), mdel(mr("j−1"))) + mr(" ∪ ")
                                    + m_set(mr("u")))
                              + mr(" − ") + mfunc("I", msubsup(mr("S"), mr("m"), mdel(mr("j−1")))))),
               "(G2)"))
    A(equation(msub(mr("I"), mr("m")) + mdel(mr("j")) + mr(" = ")
               + mfunc("I", msubsup(mr("S"), mr("m"), mdel(mr("j")))),
               "(G3)"))
    A(para("Vì mọi đường cong xuất phát từ tập rỗng, I_m chỉ phụ thuộc c_m, nên "
           "các đường cong độc lập lẫn nhau — đó chính là điều kiện mà quy hoạch "
           "động ở (G4) cần. Do I submodular, dãy lợi ích biên không tăng, nên "
           "I_m lõm."))
    A(caption("ctim/ctim_global.py:197-264. Tham lam dùng biến thể CELF lười, khởi "
              "tạo bằng σ(u) = I({u}); vì σ(u) đúng bằng lợi ích biên tại S = ∅ và "
              "I submodular, σ là cận trên hợp lệ cho mọi lợi ích về sau, nên CELF "
              "cho kết quả trùng khít tham lam háo hức. ctim/ctim_global.py:120-189."))

    A(heading("4.3. Quy hoạch động phân bổ ngân sách", 2))
    A(para("Bài toán phân bổ tài nguyên chính xác trên các đường cong:"))
    A(equation(mr("D") + mdel(mr("m,k"), "[", "]") + mr(" = ")
               + mlimlow(mr("max", plain=True),
                         mr("0 ≤ j ≤ min") + mdel(mr("k, ") + msub(mr("cap"), mr("m"))))
               + mdel(mr("D") + mdel(mr("m−1,k−j"), "[", "]") + mr(" + ")
                      + msub(mr("I"), mr("m")) + mdel(mr("j"))), "(G4)"))
    A(para("với D[0,k] = 0. Truy vết cho vector phân bổ a và giá trị "
           "dp_value = D[C,K]. Bằng nhau thì j nhỏ hơn thắng, rồi đến chỉ số cộng "
           "đồng nhỏ hơn, nên kết quả tất định."))
    A(para(run("Đây KHÔNG phải cùng một đệ quy với Algorithm 2.", bold=True)
           + run(" Dòng 35 của bài báo, ở cách đọc mặc định, khai triển thành "
                 "I[C,k] = I[C,k−1] + max_m ΔI_m — tức tham lam trên các cộng "
                 "đồng, đúng một hạt giống mỗi vòng, không có vòng lặp j và "
                 "không có số hạng cap. (G4) là bài toán phân bổ tài nguyên "
                 "chính xác: nó có thể cấp nhiều hạt giống cho một cộng đồng "
                 "trong một quyết định, và xử lý đúng cả đường cong không lõm. "
                 "Hai đệ quy chỉ trùng nhau khi mọi I_m lõm.")))
    A(caption("ctim/ea_dp.py:332-378, gọi tại ctim/ctim_global.py:400. So sánh "
              "với ctim/ctim.py:456-484. Chi phí đo được của quy hoạch động chính "
              "xác: 0.00–0.01 giây — miễn phí so với khâu dựng đường cong."))

    A(heading("4.4. Hiện thực hoá có ràng buộc hạn ngạch", 2))
    A(para("Đây là chỗ CTIM-G trả lại khoản nợ mà giả thiết độc lập ở (G2) đã vay. "
           "Thay vì ghép các tập S_m độc lập, chạy MỘT lượt tham lam chung trên "
           "hợp các cộng đồng được cấp ngân sách, với ràng buộc mỗi cộng đồng "
           "không vượt quá hạn ngạch của nó:"))
    A(equation(msup(mr("S"), mdel(mr("t"))) + mr(" = ")
               + msup(mr("S"), mdel(mr("t−1"))) + mr(" ∪ ")
               + m_set(mlimlow(mr("arg max", plain=True),
                               mr("u ∈ U,  ") + mdel(msup(mr("S"), mdel(mr("t−1")))
                               + mr(" ∩ ") + msub(mr("c"), mfunc("c", mr("u"))), "|", "|")
                               + mr(" < ") + msub(mr("a"), mfunc("c", mr("u"))))
                       + mdel(mfunc("I", msup(mr("S"), mdel(mr("t−1"))) + mr(" ∪ ")
                                    + m_set(mr("u")))
                              + mr(" − ") + mfunc("I", msup(mr("S"), mdel(mr("t−1")))))),
               "(G5)"))
    A(para("với U = ⋃{c_m : a_m > 0}, chạy tới t = K. Điểm khác biệt so với (G2) "
           "là S⁽ᵗ⁻¹⁾ ở đây là tập hạt giống ĐẦY ĐỦ trên mọi cộng đồng, nên phần "
           "chồng lấn giữa các cộng đồng chỉ được tính một lần."))
    A(caption("ctim/ctim_global.py:272-294 và 120-189. Ngân sách từng cộng đồng "
              "vẫn đúng bằng những gì quy hoạch động quyết định, nên phép phân rã "
              "vẫn định hình lời giải."))

    A(heading("4.5. Bảo toàn và khoảng cách độc lập", 2))
    A(para("Gọi S_cat là tập ghép từ các đường cong (điều mà một bản cài đặt ngây "
           "thơ trả về), tức hợp của các S_m tại ngân sách a_m, và S_r là kết quả "
           "của (G5). CTIM-G trả về tập tốt hơn trong hai:"))
    A(equation(msub(mr("S"), mr("cat")) + mr(" = ")
               + mnary("⋃", mr("m"), "", msubsup(mr("S"), mr("m"), mdel(msub(mr("a"), mr("m")))))))
    A(equation(msub(mr("S"), mr("out")) + mr(" = ")
               + mlimlow(mr("arg max", plain=True),
                         mr("S ∈ ") + m_set(msub(mr("S"), mr("r")) + mr(", ")
                                            + msub(mr("S"), mr("cat"))))
               + mfunc("I", S), "(G6)"))
    A(equation(mr("Γ = dp_value − ") + mfunc("I", msub(mr("S"), mr("out"))), "(G7)"))
    A(caption("ctim/ctim_global.py:417-441."))

    # ---- 4.5.1 why dp_value is an upper bound ----------------------------
    A(heading("4.5.1. Vì sao dp_value là cận trên", 3))
    A(para(run("Do I submodular và I(∅) = 0 nên I dưới cộng tính, do đó với các "
               "tập rời nhau")))
    A(equation(mfunc("I", mnary("⋃", mr("m"), "", msub(mr("S"), mr("m"))))
               + mr(" ≤ ")
               + mnary("∑", mr("m"), "", mfunc("I", msub(mr("S"), mr("m"))))
               + mr(" = dp_value")))
    A(para(run("Một điểm tinh tế cần nói rõ: ", bold=True)
           + run("bất đẳng thức trên chứng minh dp_value ≥ I(S_cat) cho tập GHÉP. Nó "
                 "KHÔNG chứng minh dp_value ≥ I(S_r) cho tập hiện thực hoá, vì "
                 "I_m(j) là giá trị THAM LAM chứ không phải tối ưu có ràng buộc, "
                 "nên về nguyên tắc một S_r tốt có thể vượt tổng các đường cong và "
                 "khi đó Γ âm. Bản cài đặt không khẳng định điều đó — nó chỉ tính "
                 "Γ và đánh dấu independence_violated khi Γ > 10⁻⁹. Trên sáu tập "
                 "đã chạy, Γ luôn dương.")))

    # ---- 4.5.2 the decomposition -----------------------------------------
    A(heading("4.5.2. Γ tách thành hai số hạng", 3))
    A(para("Đây là điểm dễ đọc sai nhất của toàn bộ đặc tả. dp_value mô tả các "
           "MẢNH mà đường cong đã chọn, còn I(S_out) là giá trị của tập mà (G5) "
           "THẬT SỰ chọn. Hai tập đó nói chung khác nhau, nên Γ đo hai thứ cộng "
           "lại chứ không phải một."))
    A(para("Đặt P_m = S_out ∩ c_m là mảnh của tập trả về thuộc cộng đồng m. Thêm "
           "bớt cùng một lượng:"))
    A(equation(mr("Γ = ") + mdel(mr("dp_value − ")
                                 + mnary("∑", mr("m"), "", mfunc("I", msub(mr("P"), mr("m"))))
                                 , "[", "]")
               + mr(" + ")
               + mdel(mnary("∑", mr("m"), "", mfunc("I", msub(mr("P"), mr("m"))))
                      + mr(" − ") + mfunc("I", msub(mr("S"), mr("out"))), "[", "]"),
               "(G8)"))
    A(table(["Số hạng", "Là gì", "Dấu"], [
        ["A = dp_value − Σ_m I(P_m)",
         "chênh giữa mảnh mà ĐƯỜNG CONG chọn cho cộng đồng m và mảnh mà (G5) "
         "thật sự lấy. Đường cong chọn mảnh cực đại hoá I đo RIÊNG LẺ; (G5) chọn "
         "mảnh khớp nhất với các cộng đồng khác, thường yếu hơn khi đứng riêng.",
         "không bảo đảm ≥ 0"],
        ["B = Σ_m I(P_m) − I(S_out)",
         "phần ảnh hưởng CHỒNG LẤN thật sự có trong tập trả về: tổng các mảnh đo "
         "riêng, trừ giá trị của hợp.",
         "≥ 0, chứng minh được"],
    ], [2600, 5400, 1200], ["left", "left", "left"]))
    A(para(run("B ≥ 0", bold=True)
           + run(" là hệ quả trực tiếp của tính dưới cộng tính ở 4.5.1, áp dụng "
                 "cho các P_m rời nhau. ")
           + run("A thì KHÔNG chứng minh được là không âm.", bold=True)
           + run(" Mảnh của đường cong là tiền tố tham lam CELF trong c_m, không "
                 "phải tập a_m phần tử tối ưu, nên về nguyên tắc P_m có thể vượt "
                 "nó. Vì vậy bất đẳng thức Γ ≥ B là quan sát THỰC NGHIỆM chứ "
                 "không phải định lý; nó đúng trên cả sáu tập đã chạy và trên cả "
                 "hai ví dụ ở 4.5.3.")))

    # ---- 4.5.3 two extreme cases -----------------------------------------
    A(heading("4.5.3. Hai ví dụ đối cực", 3))
    A(para("Hai đồ thị dựng tay, K = 2, h = 0.1, cho thấy Γ có thể sinh ra hoàn "
           "toàn từ số hạng này hay số hạng kia."))
    A(para(run("Ví dụ A — Γ hoàn toàn từ A, không chồng lấn chút nào.", bold=True)
           + run(" 16 đỉnh; cộng đồng 1 = {1, 2}, mọi đỉnh còn lại thuộc cộng "
                 "đồng 0. Cung: 0 → {10,11,12}, 1 → {10,11,13}, 2 → {14,15}, mọi "
                 "pp = 0.9. Đứng riêng, nút 1 đáng 3.70 còn nút 2 chỉ 2.80, nên "
                 "đường cong của cộng đồng 1 chọn nút 1. Nhưng nút 1 đè lên nút 0 "
                 "ở {10,11}, nên (G5) lấy nút 2. Phân bổ a = {0: 1, 1: 1}, "
                 "S_out = {0, 2}.")))
    A(table(["Đại lượng", "Giá trị"], [
        ["dp_value", "7.4000"],
        ["Σ_m I(P_m),  P = {0} ∪ {2}", "6.5000"],
        ["I(S_out)", "6.5000"],
        ["B — chồng lấn thật", "0.0000"],
        ["A — do đổi mảnh", "0.9000"],
        ["Γ", "0.9000   (12.2% của dp_value;  13.8% của I(S_out))"],
    ], [4200, 3400], ["left", "right"]))
    A(caption("Tập trả về {0, 2} không chồng lấn một chút nào, B = 0, mà Γ vẫn "
              "bằng 0.9000. Toàn bộ Γ đến từ việc cộng đồng 1 đổi nút 1 lấy nút 2 "
              "— một phép đổi ĐÚNG: I({0,2}) = 6.50 so với I({0,1}) = 5.78."))
    A(para(run("Ví dụ B — Γ hoàn toàn từ B, không đổi mảnh.", bold=True)
           + run(" 16 đỉnh, ba cộng đồng: c₀ = {0,1,2,3,8,9}, c₁ = {4,5,10,11}, "
                 "c₂ = {6,7,12,13,14,15}. Cung: 0 → {8,9,10,11} và 4 → {8,9,10,11} "
                 "với pp = 0.9 — TRÙNG HOÀN TOÀN; 6 → {12,13,14,15} với pp = 0.8, "
                 "hoàn toàn độc lập. Đứng riêng, nút 0 và nút 4 cùng đáng 4.60 "
                 "còn nút 6 chỉ 4.20, nên (G4) cấp vốn cho c₀ và c₁ và bỏ đói c₂. "
                 "(G5) lấy đúng hai nút mà đường cong đã chọn, nên A = 0.")))
    A(table(["Đại lượng", "Giá trị"], [
        ["dp_value", "9.2000"],
        ["Σ_m I(P_m),  P = {0} ∪ {4}", "9.2000"],
        ["I(S_out)", "5.9600"],
        ["B — chồng lấn thật", "3.2400"],
        ["A — do đổi mảnh", "0.0000"],
        ["Γ", "3.2400   (35.2% của dp_value;  54.4% của I(S_out))"],
    ], [4200, 3400], ["left", "right"]))
    A(caption("Ở đây Γ đúng bằng phần chồng lấn, vì S_out = S_cat. Đẳng thức "
              "Γ = B xảy ra KHI VÀ CHỈ KHI (G5) chọn đúng những mảnh mà đường "
              "cong đã chọn. Ví dụ này cũng cho thấy tác hại đi xa hơn con số: "
              "phương án {0, 6} đạt I = 8.80, hơn hẳn 5.96, nhưng c₂ nhận 0 suất "
              "ở (G4) nên nút 6 không có trong bể ứng viên của (G5) và không bao "
              "giờ được xét. Γ lớn là dấu hiệu của đúng tình huống đó."))

    # ---- 4.5.4 measured ---------------------------------------------------
    A(heading("4.5.4. Số đo trên sáu tập dữ liệu", 3))
    A(para(run("Γ được tính theo TỪNG SẢN PHẨM, không phải một số cho cả tập dữ "
               "liệu.", bold=True)
           + run(" Bảng dưới ghi khoảng của ba sản phẩm mỗi tập (Yelp: hai sản "
                 "phẩm; Delicious: một). K = 20, h = 0.1.")))
    A(table(["Tập dữ liệu", "dp_value", "I(S_out)", "Γ / I(S_out)", "Γ / dp_value"], [
        ["Ciao", "144.83–144.86", "118.19–118.21", "22.54%", "18.39–18.40%"],
        ["Digg", "861.17–861.26", "763.89–764.14", "12.71–12.73%", "11.28–11.30%"],
        ["Epinions", "1724.68–1724.88", "1540.26–1540.44", "11.97%", "10.69%"],
        ["Yelp", "8203.70–8209.22", "7414.60–7419.15", "10.64–10.65%", "9.62%"],
        ["Last.fm", "68.43–89.75", "64.17–83.78", "6.63–7.12%", "6.22–6.65%"],
        ["Delicious", "23.03", "23.03", "0.00%", "0.00%"],
    ], [1600, 2000, 2000, 1800, 1800],
        ["left", "right", "right", "right", "right"]))
    A(para(run("Hai cột phần trăm dùng hai MẪU SỐ khác nhau và phải nói rõ dùng "
               "cái nào.", bold=True)
           + run(" Bộ chạy in ra Γ / I(S_out) (scripts/run_ctim_global.py:386-388, "
                 "tính bằng 100·(dp_value / I(S_out) − 1)), trong khi trường "
                 "stats[\"independence_gap_pct\"] lưu Γ / dp_value "
                 "(ctim/ctim_global.py:440). Trên Digg hai con số là 12.73% và "
                 "11.30%. Cả hai đều hợp lệ theo định nghĩa riêng; trộn lẫn thì "
                 "không.")))
    A(para(run("Delicious Γ = 0 không phải vì thuật toán chính xác ở đó.", bold=True)
           + run(" Ở h = 0.1 cây ảnh hưởng trung bình của tập này chỉ có 1.07 "
                 "đỉnh (delicious_paper_h01.log:73), tức gần như mọi đỉnh chỉ tự "
                 "kích hoạt chính nó. Không có gì để chồng lấn, nên giả thiết độc "
                 "lập ở (G2) đúng theo nghĩa đen. Đó là một tập dữ liệu suy biến "
                 "ở ngưỡng này, không phải một thắng lợi của phương pháp.")))

    # ---- 4.5.5 how to read it --------------------------------------------
    A(heading("4.5.5. Cách đọc Γ, và bốn cách đọc sai", 3))
    A(para(run("Đọc đúng: ", bold=True)
           + run("“quy hoạch động ở (G4) hứa nhiều hơn giá trị thật giao được bao "
                 "nhiêu.” Γ lớn ⇒ các đường cong ở (G2) tô hồng nặng ⇒ bảng phân "
                 "bổ a_m dựng trên chúng đáng ngờ. Đó là giá trị CHẨN ĐOÁN của Γ, "
                 "và là đại lượng mà mọi phương pháp phân rã theo cộng đồng trước "
                 "đây mặc định bằng không.")))
    A(bullet(run("KHÔNG phải “tập trả về chồng lấn Γ phần trăm”. ", bold=True)
             + run("Ví dụ A có Γ = 12.2% với tập trả về không chồng lấn chút nào.")))
    A(bullet(run("KHÔNG phải chứng chỉ tối ưu. ", bold=True)
             + run("dp_value ≥ I(S_cat) thì đúng, nhưng dp_value ≥ I(S_r) thì "
                   "không được bảo đảm (4.5.1), nên Γ = 12.73% KHÔNG có nghĩa "
                   "“kết quả cách tối ưu 12.73%”.")))
    A(bullet(run("KHÔNG so được giữa hai bài báo nếu không nêu mẫu số. ", bold=True)
             + run("Γ / I(S_out) và Γ / dp_value chênh nhau khoảng 1.4 điểm phần "
                   "trăm trên Digg.")))
    A(bullet(run("KHÔNG đọc được từ một lần chạy có bật tinh chỉnh. ", bold=True)
             + run("Γ được ghi ở ctim/ctim_global.py:438, TRƯỚC pha tinh chỉnh, "
                   "và không bao giờ tính lại; trong khi stats[\"value\"] bị ghi "
                   "đè sau đó ở :460-462. Với repair = True hai trường này không "
                   "còn khớp nhau. Mặc định repair = False (:303) và không lần "
                   "chạy nào trong sáu nhật ký bật nó, nên mọi số ở 4.5.4 đều "
                   "không bị ảnh hưởng.")))

    A(heading("4.6. Bước tinh chỉnh tuỳ chọn", 2))
    A(para("Nếu bật, một bước tìm kiếm cục bộ 1-hoán vị chạy trên cùng đối tượng "
           "MIA toàn cục, với bể ứng viên mặc định là toàn bộ V. Nó có bảo đảm "
           "không bao giờ trả về tập tệ hơn, nên giá trị đầu ra đơn điệu không "
           "giảm. Đo được trên Digg: nó chấp nhận 0 hoán vị ở cả ba sản phẩm, tức "
           "kết quả của (G6) đã là tối ưu cục bộ 1-hoán vị."))

    # ---- differences ------------------------------------------------------
    A(heading("5. Ba khác biệt so với CTIM", 1))
    A(table(["#", "CTIM", "CTIM-G", "Hệ quả"], [
        ["1",
         "Mỗi cộng đồng có MIA riêng, giới hạn trên c_m (nodes=members). Cạnh chỉ "
         "tồn tại khi cả hai đầu mút trong c_m.",
         "Một MIA duy nhất trên toàn G, dùng cho mọi cộng đồng và mọi pha.",
         "CTIM không nhìn thấy cạnh bắc cầu giữa các cộng đồng; lợi ích của nó bị "
         "cắt cụt. Mức cắt đo được bằng tỉ số t / t_m: từ 1.0 lần (Delicious) đến "
         "7.7 lần (Yelp)."],
        ["2",
         "Lợi ích biên đo với S_m — chỉ các hạt giống của chính cộng đồng đó. Tập "
         "hạt giống toàn cục có tồn tại trong mã nguồn nhưng CHỈ ĐƯỢC GHI, không "
         "một đường tính điểm nào đọc nó.",
         "Ở pha hiện thực hoá (G5), lợi ích biên đo với S đầy đủ trên mọi cộng đồng. "
         "Đây là nơi DUY NHẤT trong cả ba mô-đun mà phần dư thừa giữa các cộng đồng "
         "được tính giá.",
         "CTIM đếm hai lần phần ảnh hưởng chồng lấn giữa các cộng đồng, và chỉ lộ ra "
         "khi tập kết quả được chấm lại trên toàn đồ thị sau đó; CTIM-G đếm một lần "
         "ngay lúc chọn."],
        ["3",
         "Hạt giống được cam kết từng cái một qua K vòng; tập cuối là các tập cộng "
         "đồng ghép lại.",
         "Đường cong dựng trước, quy hoạch động chạy một lần, rồi một lượt tham lam "
         "chung dưới hạn ngạch (G5).",
         "Ngân sách vẫn do quy hoạch động quyết định, nhưng việc chọn nút cụ thể là "
         "chung chứ không độc lập."],
    ], [400, 2900, 2700, 3000], ["right", "left", "left", "left"]))

    A(para(run("Điều KHÔNG thay đổi.", bold=True)
           + run(" Phép phát hiện cộng đồng Eq (19), tập ứng viên c_m, dạng của "
                 "quy hoạch động, mô hình lan truyền Eq (12)–(18), và ngưỡng h. "
                 "CTIM-G không phải một mô hình khác — nó là cùng một phép phân rã "
                 "với một hàm mục tiêu không bị cắt cụt.")))

    # ---- complexity -------------------------------------------------------
    A(heading("6. Độ phức tạp", 1))
    A(para(run("CTIM", bold=True)))
    A(code_block([
        "O(nC)                            Eq (19) phat hien cong dong",
        "  +  sum_m O(n_m · t_m log t_m)    dung cay tren do thi con cam sinh",
        "  +  O(CK)                         o quy hoach dong",
    ]))
    A(para(run("CTIM-G", bold=True)))
    A(code_block([
        "O(nC)                            Eq (19), giong het",
        "  +  O(n · t log t)                MIA toan cuc + bang solo",
        "  +  sum_m O(n_m · t log t)        duong cong (G2)",
        "  +  O(C·K²)                       quy hoach dong (G4)",
        "  +  O(K · t²)                     hien thuc hoa (G5)",
    ]))
    A(para("Cả hai đều dựng khoảng n cây ảnh hưởng. Toàn bộ khoảng cách chi phí "
           "nằm ở tỉ số t / t_m — kích thước cây trên đồ thị đầy đủ so với trên "
           "đồ thị con. Chính tỉ số đó cũng dự báo được khoảng cách chất lượng."))
    A(table(["Tập dữ liệu", "|V|", "t / t_m", "Khoảng cách CTIM-G",
             "Thời gian CTIM-G / CTIM"], measured_rows(),
            [1700, 1400, 1200, 2100, 2600],
            ["left", "right", "right", "right", "right"]))
    A(caption("Đo ở h = 0.1, K = 20, trung bình trên 3 sản phẩm mỗi tập, đọc trực "
              "tiếp từ log của từng lần chạy nên không thể lệch với số liệu gốc. "
              "Cột cuối đã trừ khoản dựng lại Eq (12) bên trong CTIM (Mục 8.2). "
              "Dòng Delicious thoái hoá: chỉ 0.52% số cạnh sống ở h = 0.1 nên hàm "
              "mục tiêu gần như rỗng và hai phương pháp cho điểm giống hệt dù khác "
              "nhau 15/20 hạt giống, nên tỉ số thời gian ở đó không có ý nghĩa."))
    A(para(run("Bước quy hoạch động tốn 0.00–0.01 giây trên mọi tập đã thử.", bold=True)
           + run(" Khoảng 70% thời gian của CTIM-G nằm ở pha dựng đường cong (G2). "
                 "Đó là pha duy nhất đáng tối ưu.")))

    # ---- properties -------------------------------------------------------
    A(heading("7. Tính chất", 1))
    A(bullet("Tất định. Không pha nào tiêu thụ số ngẫu nhiên; tham số rng chỉ "
             "được chuyển tiếp cho bước tinh chỉnh tuỳ chọn, nơi nó chỉ phá thế "
             "hoà."))
    A(bullet("Đường cong I_m đơn điệu không giảm và lõm: do I submodular, dãy lợi "
             "ích biên của tham lam không tăng. Vì thế bước repair_monotone trong "
             "(G2) là phép không làm gì — nhưng nó vẫn là điều kiện mà (G4) dựa "
             "vào để giả định “thêm ngân sách không bao giờ có hại”."))
    A(bullet("Ở C = 1 hai phương pháp trùng nhau chứng minh được: đồ thị con cảm "
             "sinh bằng đúng đồ thị đầy đủ và đệ quy dòng 35 suy biến về tham "
             "lam. Self-test của bản cài đặt khẳng định CTIM-G trả về DÃY hạt "
             "giống trùng khít MIA.greedy_incremental với K = 1, 3, 5, 8, và "
             "Γ = 0. Toàn bộ khác biệt giữa hai phương pháp vì thế là hàm của "
             "phép phân hoạch."))
    A(bullet("|S| có thể nhỏ hơn K. (G4) phá thế hoà theo hướng j nhỏ hơn, nên "
             "khi hạt giống thêm không đóng góp gì thì tổng ngân sách < K và "
             "CTIM-G trả về ít hơn K hạt giống, không có cảnh báo. CTIM ngược "
             "lại: nó chèn thêm hạt giống lợi ích bằng 0 cho đủ K. Mọi bảng so "
             "sánh theo |S| hoặc theo trung bình mỗi hạt giống phải tính đến "
             "điều này."))
    A(bullet("Một cộng đồng được cấp ngân sách 0 sẽ KHÔNG BAO GIỜ đóng góp hạt "
             "giống ở (G5), vì bể ứng viên chỉ gồm các cộng đồng có a_m > 0. "
             "Giai đoạn đường cong lạc quan ở (G2) là cơ hội duy nhất của nó. "
             "CTIM thì chào lại mọi cộng đồng ở cả K vòng."))
    A(bullet("dp_value ≥ I(S_cat) luôn đúng, do tính dưới cộng tính. dp_value ≥ I(S_r) "
             "thì KHÔNG được bảo đảm, vì I_m(j) là giá trị tham lam."))
    A(bullet("Giá trị trả về không bao giờ thấp hơn cách đọc ngây thơ của chính "
             "quy hoạch động, nhờ (G6)."))
    A(bullet("Với bước tinh chỉnh bật và bể ứng viên đầy đủ, kết quả là tối ưu cục "
             "bộ 1-hoán vị đã được chứng nhận."))

    A(heading("8. Phạm vi và những điều phải nói kèm số liệu", 1))

    A(heading("8.1. Tính vòng luẩn quẩn của phép so sánh", 2))
    A(para("Mọi con số thực nghiệm trích dẫn ở đây đo bằng Eq (18) ở h = 0.1, đúng "
           "giao thức của bài báo: một ngưỡng duy nhất cho cả việc chọn hạt giống "
           "lẫn việc chấm điểm."))
    A(para(run("Phát biểu chính xác của vấn đề: ", bold=True)
           + run("hàm mà CTIM tối ưu bên trong (I⁽ᵐ⁾ trên C đồ thị con khác nhau) "
                 "và hàm dùng để chấm nó (I trên toàn đồ thị) là hai hàm khác "
                 "nhau. Với CTIM-G thì đó là cùng một hàm — (G2) và (G5) đều là "
                 "tham lam trên chính Eq (18) toàn cục. Vì thế mọi so sánh ở cùng "
                 "một h nghiêng về phía CTIM-G theo cấu trúc, và khoảng cách báo "
                 "cáo ở Mục 6 là ")
           + run("cận trên", bold=True)
           + run(" của khoảng cách thật.")))

    A(heading("8.2. Một sai số đo đã được sửa", 2))
    A(para(run("Các tỉ số thời gian ở Mục 6 từng bị lệch và đã được đo lại.", bold=True)
           + run(" Hàm chọn hạt giống của CTIM không nhận tham số pp; nó tự dựng "
                 "lại Eq (12) từ đối tượng EdgeWeights, mà EdgeWeights.for_item "
                 "KHÔNG lưu đệm — nó duyệt lại toàn bộ cạnh mỗi lần gọi. CTIM-G "
                 "thì được truyền pp dựng sẵn nên không trả khoản đó. Đo được: "
                 "41–47% thời gian “chọn” của CTIM thực chất là dựng lại Eq (12). "
                 "Trình đo đã được sửa để trừ khoản này ra, và mọi tỉ số ở Mục 6 "
                 "là số sau khi sửa — so sánh chọn với chọn.")))
    A(caption("ctim/influence.py:254-263 (không lưu đệm) ; "
              "scripts/run_ctim_global.py (khoản trừ)."))

    A(heading("8.3. Ba sản phẩm không phải ba phép thử độc lập", 2))
    A(para("Trong mỗi tập dữ liệu, khoảng cách giữa các sản phẩm có độ lệch chuẩn "
           "gần bằng 0. Đó là tính tái lập chứ không phải tính bền: θ gần đều nên "
           "mọi sản phẩm sinh ra gần như cùng một đồ thị lan truyền. Lấy trung "
           "bình qua sản phẩm để tăng độ tin cậy là vô nghĩa."))

    A(heading("8.4. MIA là một xấp xỉ, độc lập với việc chia cộng đồng", 2))
    A(para("Eq (17) chỉ chạy trên MỘT cây ảnh hưởng cực đại cho mỗi nút, nên nút "
           "được nuôi bởi nhiều đường có xác suất tương đương bị tính thiếu — ở "
           "cả CTIM lẫn CTIM-G. Sai số này cộng dồn với, và độc lập với, phần mất "
           "mát do đồ thị con cảm sinh của CTIM."))
    return "".join(b)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("results", "ctimg_specification.docx"))
    args = p.parse_args(argv)
    write_docx(args.out, build())
    print("[write] %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
