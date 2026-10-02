"""Sections I (Introduction) and II (Related work) of the CTIM-G paper, as .docx.

Numbered references in the style of Applied Intelligence, the venue of the CTIM
paper itself.  Every entry was either transcribed from the CTIM paper's own
bibliography or verified independently against DBLP / the publisher.  Three
entries in that bibliography are demonstrably wrong and are corrected here, with
the correction recorded in a note at the end of the document.

Usage
-----
    python scripts/export_paper_intro_docx.py --out results/paper_sections_I_II.docx
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_docx import (caption, heading, para, run, table,        # noqa: E402
                         write_docx)

# ---------------------------------------------------------------------------
# References.  (authors, year, title, venue) -- rendered [n] in this order.
# ---------------------------------------------------------------------------

REFS = [
    ("Domingos P, Richardson M", "2001", "Mining the network value of customers",
     "In: Proceedings of ACM SIGKDD, pp 57-66"),
    ("Richardson M, Domingos P", "2002",
     "Mining knowledge-sharing sites for viral marketing",
     "In: Proceedings of ACM SIGKDD, pp 61-70"),
    ("Kempe D, Kleinberg J, Tardos E", "2003",
     "Maximizing the spread of influence through a social network",
     "In: Proceedings of ACM SIGKDD, pp 137-146"),
    ("Leskovec J, Krause A, Guestrin C, Faloutsos C, VanBriesen J, Glance N",
     "2007", "Cost-effective outbreak detection in networks",
     "In: Proceedings of ACM SIGKDD, pp 420-429"),
    ("Goyal A, Lu W, Lakshmanan LVS", "2011",
     "CELF++: optimizing the greedy algorithm for influence maximization in "
     "social networks", "In: Proceedings of WWW, pp 47-48"),
    ("Chen W, Wang Y, Yang S", "2009",
     "Efficient influence maximization in social networks",
     "In: Proceedings of ACM SIGKDD, pp 199-208"),
    ("Chen W, Wang C, Wang Y", "2010",
     "Scalable influence maximization for prevalent viral marketing in "
     "large-scale social networks",
     "In: Proceedings of ACM SIGKDD, pp 1029-1038"),
    ("Cheng S, Shen H, Huang J", "2012",
     "StaticGreedy: solving the scalability-accuracy dilemma in influence "
     "maximization", "In: Proceedings of ACM CIKM, pp 509-518"),
    ("Borgs C, Brautbar M, Chayes J, Lucier B", "2014",
     "Maximizing social influence in nearly optimal time",
     "In: Proceedings of ACM-SIAM SODA, pp 946-957"),
    ("Tang Y, Xiao X, Shi Y", "2014",
     "Influence maximization: near-optimal time complexity meets practical "
     "efficiency", "In: Proceedings of ACM SIGMOD, pp 75-86"),
    ("Tang Y, Shi Y, Xiao X", "2015",
     "Influence maximization in near-linear time: a martingale approach",
     "In: Proceedings of ACM SIGMOD, pp 1539-1554"),
    ("Wang Y, Cong G, Song G, Xie K", "2010",
     "Community-based greedy algorithm for mining top-K influential nodes in "
     "mobile social networks", "In: Proceedings of ACM SIGKDD, pp 1039-1048"),
    ("Cao T, Wu X, Wang S, Hu X", "2010",
     "OASNET: an optimal allocation approach to influence maximization in "
     "modular social networks",
     "In: Proceedings of ACM Symposium on Applied Computing, pp 1088-1094"),
    ("Chen YC, Zhu WY, Peng WC, Lee WC, Lee SY", "2014",
     "CIM: community-based influence maximization in social networks",
     "ACM Transactions on Intelligent Systems and Technology 5(2):25"),
    ("Clauset A, Newman MEJ, Moore C", "2004",
     "Finding community structure in very large networks",
     "Physical Review E 70(6):066111"),
    ("Blei DM, Ng AY, Jordan MI", "2003", "Latent Dirichlet allocation",
     "Journal of Machine Learning Research 3:993-1022"),
    ("Barbieri N, Bonchi F, Manco G", "2012",
     "Topic-aware social influence propagation models",
     "In: Proceedings of IEEE ICDM, pp 81-90"),
    ("Liu L, Tang J, Han J, Jiang M, Yang S", "2010",
     "Mining topic-level influence in heterogeneous networks",
     "In: Proceedings of ACM CIKM, pp 199-208"),
    ("Li H, Bhowmick SS, Sun A, Cui J", "2015",
     "Conformity-aware influence maximization in online social networks",
     "The VLDB Journal 24(1):117-141"),
    ("Huang H, Shen H, Meng Z, Chang H, He H", "2019",
     "Community-based influence maximization for viral marketing",
     "Applied Intelligence 49(6):2137-2150"),
    ("Nemhauser GL, Wolsey LA, Fisher ML", "1978",
     "An analysis of approximations for maximizing submodular set functions - I",
     "Mathematical Programming 14(1):265-294"),
]


def ref_block():
    out = []
    for i, (auth, yr, title, venue) in enumerate(REFS, start=1):
        out.append(para(
            run("%2d.  " % i, bold=True)
            + run("%s (%s) " % (auth, yr))
            + run(title + ". ", italic=True)
            + run(venue),
            indent=340, spacing_after=90))
    return "".join(out)


def build():
    b = []
    A = b.append

    A(para(run("CTIM-G: Cực đại hoá ảnh hưởng dựa trên cộng đồng với hàm mục "
               "tiêu không bị cắt cụt", bold=True),
           style="Title", spacing_after=60))
    A(para(run("Bản thảo — Mục I và Mục II", italic=True, color="5A6673")))

    # ------------------------------------------------------------------ I
    A(heading("I.  Mở đầu", 1))

    A(heading("1.1.  Bối cảnh và bài toán", 2))
    A(para("Tiếp thị lan truyền khai thác một quan sát đơn giản: trong mạng xã "
           "hội, quyết định mua hàng của một người chịu ảnh hưởng từ những người "
           "họ tin tưởng. Nếu chọn đúng một nhóm nhỏ người dùng để tặng sản phẩm "
           "hoặc khuyến mãi, hiệu ứng truyền miệng có thể lan tới một lượng khách "
           "hàng lớn hơn nhiều lần chi phí bỏ ra. Domingos và Richardson [1, 2] là "
           "những người đầu tiên đặt ý tưởng này thành một bài toán khai phá dữ "
           "liệu, mô hình hoá giá trị mạng của khách hàng thay vì chỉ giá trị giao "
           "dịch trực tiếp của họ."))
    A(para("Kempe, Kleinberg và Tardos [3] hình thức hoá nó thành bài toán cực "
           "đại hoá ảnh hưởng: cho một đồ thị xã hội và một mô hình lan truyền, "
           "hãy chọn tập hạt giống S có kích thước K sao cho số người dùng kỳ vọng "
           "bị kích hoạt là lớn nhất. Họ chứng minh bài toán là NP-khó dưới cả mô "
           "hình Independent Cascade lẫn Linear Threshold, nhưng hàm lan truyền σ "
           "là đơn điệu không giảm và submodular. Ghép với kết quả kinh điển của "
           "Nemhauser và cộng sự [21] về cực đại hoá hàm submodular đơn điệu dưới "
           "ràng buộc lực lượng, điều đó cho thuật toán tham lam một bảo đảm: tập "
           "S mà nó trả về thoả σ(S) ≥ (1 − 1/e)·σ(S*) ≈ 0,632·σ(S*), trong đó S* "
           "là tập K hạt giống tối ưu. Kết quả đó định hình toàn bộ hướng nghiên "
           "cứu suốt hai thập kỷ sau."))
    A(para("Điểm yếu của thuật toán tham lam là chi phí. Mỗi lần đánh giá một ứng "
           "viên đòi hỏi hàng chục nghìn lần mô phỏng Monte-Carlo, khiến nó không "
           "chạy nổi trên đồ thị quy mô thực tế. Ba hướng khắc phục đã hình thành: "
           "giảm số lần đánh giá bằng cách khai thác tính submodular [4, 5, 8]; "
           "thay mô phỏng bằng một xấp xỉ có cấu trúc như cây ảnh hưởng cực đại "
           "[6, 7]; và chuyển sang lấy mẫu ngược với bảo đảm xác suất [9, 10, 11]."))
    A(para(run("Một hướng thứ tư, độc lập với ba hướng trên, là phân rã theo cộng "
               "đồng.", bold=True)
           + run(" Mạng xã hội thực tế có cấu trúc cộng đồng rõ rệt: ảnh hưởng lan "
                 "mạnh bên trong một cộng đồng và yếu giữa các cộng đồng. Nếu chia "
                 "đồ thị thành C cộng đồng rồi giải C bài toán con nhỏ, chi phí "
                 "giảm đi rất nhiều [12, 13, 14]. Nhưng phép chia đó phải trả một "
                 "cái giá ít khi được nêu tên, và đó chính là điểm xuất phát của "
                 "bài báo này.")))
    A(para(run("Song song, ảnh hưởng được nhận ra là phụ thuộc chủ đề.", bold=True)
           + run(" Một người có uy tín về công nghệ chưa chắc thuyết phục được bạn "
                 "bè về mỹ phẩm. Các mô hình lan truyền nhận biết chủ đề [17, 18] "
                 "gắn trọng số cạnh với nội dung sản phẩm, thường thông qua một mô "
                 "hình chủ đề ẩn [16]. Huang và cộng sự [20] kết hợp hai hướng này "
                 "thành CTIM, thuật toán mà bài báo hiện tại lấy làm cơ sở.")))

    A(heading("1.2.  Vấn đề còn tồn tại", 2))
    A(para(run("CTIM chia đồ thị theo cộng đồng, rồi với mỗi cộng đồng tính độ lan "
               "truyền trên đồ thị con cảm sinh của chính cộng đồng đó", bold=True)
           + run(" — một cạnh chỉ tồn tại khi cả hai đầu mút cùng nằm trong cộng "
                 "đồng. Cách làm này khiến thuật toán nhanh, nhưng kéo theo ba hệ "
                 "quả mà bài báo gốc không định lượng.")))
    A(table(["", "Hệ quả"], [
        ["a", "Mọi cạnh bắc cầu giữa các cộng đồng biến mất khỏi tầm nhìn. Một nút "
              "có ảnh hưởng chủ yếu ra bên ngoài cộng đồng của nó bị đánh giá thấp "
              "một cách có hệ thống."],
        ["b", "Giá trị lợi ích của hai cộng đồng khác nhau được đem so sánh trực "
              "tiếp trong bước phân bổ ngân sách, dù chúng được đo trên hai đồ thị "
              "khác nhau với hai tập cạnh khác nhau."],
        ["c", "Hạt giống của mỗi cộng đồng được chọn độc lập với các cộng đồng "
              "khác, nên phần ảnh hưởng chồng lấn giữa chúng bị đếm hai lần tại "
              "thời điểm chọn."],
    ], [500, 8500], ["right", "left"]))

    A(heading("1.3.  Giải pháp đề xuất", 2))
    A(para("Bài báo này đề xuất CTIM-G, giữ nguyên toàn bộ phép phân rã của CTIM "
           "— cùng cách phát hiện cộng đồng, cùng tập ứng viên, cùng khung quy "
           "hoạch động — nhưng gỡ bỏ ba hệ quả trên. Ba đóng góp chính:"))
    A(table(["", "Đóng góp"], [
        ["1", "Một hàm mục tiêu toàn cục duy nhất. Mọi lợi ích, ở mọi cộng đồng và "
              "mọi pha, được đo trên toàn đồ thị thay vì trên đồ thị con cảm sinh. "
              "Cạnh bắc cầu không còn bị loại bỏ, và giá trị của các cộng đồng "
              "khác nhau trở thành giá trị của cùng một hàm nên so sánh được. Điều "
              "này xử lý (a) và (b)."],
        ["2", "Hiện thực hoá chung có ràng buộc hạn ngạch. Quy hoạch động vẫn "
              "quyết định ngân sách của từng cộng đồng, nhưng hạt giống cụ thể "
              "được chọn bằng một lượt tham lam duy nhất trên toàn bộ ngân sách "
              "đó, nên phần chồng lấn giữa các cộng đồng chỉ được tính một lần. "
              "Điều này xử lý (c)."],
        ["3", "Báo cáo sai số của chính phép phân rã. Chênh lệch giữa giá trị mà "
              "quy hoạch động dự báo và giá trị thật của tập hạt giống được tính "
              "và công bố cùng kết quả, thay vì được mặc định bằng không."],
    ], [500, 8500], ["right", "left"]))
    A(caption("Đóng góp 3 biến một giả thiết ngầm của toàn bộ nhóm phương pháp "
              "phân rã thành một đại lượng đo được trên từng lần chạy."))

    A(heading("1.4.  Cấu trúc bài báo", 2))
    A(para("Phần còn lại của bài báo được tổ chức như sau. Mục II khảo sát các "
           "công trình liên quan theo bốn nhánh: cực đại hoá ảnh hưởng cổ điển, "
           "các hướng tăng tốc, phân rã theo cộng đồng, và mô hình nhận biết chủ "
           "đề. Mục III trình bày phương pháp đề xuất: nhắc lại các định nghĩa "
           "dùng chung, mô tả CTIM làm thuật toán tham chiếu, phân tích ưu và "
           "nhược điểm của nó, rồi định nghĩa hình thức CTIM-G cùng độ phức tạp và "
           "các tính chất chứng minh được. Mục IV trình bày thực nghiệm trên sáu "
           "tập dữ liệu thực, so sánh chất lượng lẫn thời gian chạy, và xác định "
           "điều kiện mà dưới đó CTIM-G thực sự có lợi. Mục V kết luận và nêu "
           "hướng phát triển."))

    # ----------------------------------------------------------------- II
    A(heading("II.  Các công trình liên quan", 1))

    A(heading("2.1.  Cực đại hoá ảnh hưởng và thuật toán tham lam", 2))
    A(para("Kempe và cộng sự [3] đặt nền móng cho hướng nghiên cứu bằng cách "
           "chứng minh tính NP-khó của bài toán và tính đơn điệu, submodular của "
           "hàm lan truyền σ. Ghép với định lý của Nemhauser và cộng sự [21], hai "
           "tính chất đó cho thuật toán tham lam một bảo đảm chất lượng: độ lan "
           "truyền của tập nó chọn không thấp hơn (1 − 1/e) ≈ 63,2% độ lan truyền "
           "của tập tối ưu có cùng kích thước K."))
    A(para(run("Cần nói rõ hai điều về bảo đảm này. ", bold=True)
           + run("Thứ nhất, nó là tỉ lệ so với giá trị tối ưu σ(S*) chứ không phải "
                 "so với tổng số nút của mạng; vì bài toán NP-khó nên σ(S*) không "
                 "biết được, và bảo đảm chỉ nói rằng khoảng cách tới nó bị chặn. "
                 "Thứ hai, hằng số (1 − 1/e) chỉ đúng khi σ được tính chính xác. "
                 "Trên thực tế σ phải ước lượng bằng Monte-Carlo, nên phát biểu "
                 "chặt chẽ là tỉ lệ (1 − 1/e − ε) đạt được với xác suất cao, trong "
                 "đó ε phản ánh sai số lấy mẫu và nhỏ dần khi tăng số lần mô "
                 "phỏng.")))
    A(para("Chính điều thứ hai tạo ra nút thắt tính toán: muốn ε nhỏ thì mỗi bước "
           "cần mô phỏng lặp lại rất nhiều lần, và tham lam có K bước, mỗi bước "
           "duyệt mọi ứng viên."))
    A(para("Leskovec và cộng sự [4] đề xuất CELF, khai thác trực tiếp tính "
           "submodular: lợi ích biên của một nút không bao giờ tăng khi tập hạt "
           "giống lớn dần, nên giá trị cũ là một cận trên hợp lệ và chỉ cần tính "
           "lại khi nút đó thực sự lên đầu hàng đợi. Goyal và cộng sự [5] tinh "
           "chỉnh thêm với CELF++. Cheng và cộng sự [8] tiếp cận từ hướng khác với "
           "StaticGreedy, dùng chung một tập đồ thị mẫu cố định cho mọi lần đánh "
           "giá thay vì lấy mẫu lại mỗi lần."))
    A(para("Cả ba đều giữ nguyên bảo đảm xấp xỉ và chỉ cải thiện hằng số. Đó vừa "
           "là ưu điểm — không đánh đổi chất lượng — vừa là giới hạn của nhóm này."))

    A(heading("2.2.  Xấp xỉ có cấu trúc và lấy mẫu ngược", 2))
    A(para("Nhóm thứ hai thay hẳn phép mô phỏng. Chen và cộng sự [6, 7] đề xuất "
           "mô hình cây ảnh hưởng cực đại: với mỗi nút, chỉ giữ lại đường lan "
           "truyền có xác suất lớn nhất từ mỗi nguồn, và cắt bỏ mọi đường có xác "
           "suất dưới một ngưỡng h. Kết quả là một cây, trên đó xác suất kích hoạt "
           "tính được bằng đệ quy trong thời gian tuyến tính theo kích thước cây. "
           "Đây chính là mô hình mà CTIM và bài báo này sử dụng, và ngưỡng h là "
           "tham số đánh đổi giữa độ chính xác và chi phí."))
    A(para("Nhóm lấy mẫu ngược đi theo hướng lý thuyết hơn. Borgs và cộng sự [9] "
           "giới thiệu ý tưởng sinh các tập khả đạt ngược và quy bài toán về phủ "
           "tập lớn nhất, đạt độ phức tạp gần tối ưu. Tang và cộng sự phát triển "
           "thành TIM [10] rồi IMM [11], đưa số mẫu cần thiết xuống mức thực dụng "
           "trong khi vẫn giữ bảo đảm xác suất. Điểm chung của nhóm này là chúng "
           "tối ưu độ lan truyền thuần tuý theo cấu trúc và không xét nội dung sản "
           "phẩm, nên không trực tiếp áp dụng được cho bài toán tiếp thị lan "
           "truyền theo từng mặt hàng."))

    A(heading("2.3.  Phân rã theo cộng đồng", 2))
    A(para("Wang và cộng sự [12] đề xuất CGA, thuật toán tham lam dựa trên cộng "
           "đồng: chia mạng thành các cộng đồng, chọn hạt giống trong từng cộng "
           "đồng, và dùng quy hoạch động để phân bổ ngân sách giữa chúng. Đây là "
           "tiền thân trực tiếp của khung mà CTIM và CTIM-G kế thừa."))
    A(para("Cao và cộng sự [13] hình thức hoá bước phân bổ thành một bài toán "
           "phân bổ tài nguyên tối ưu trong OASNET. Chen và cộng sự [14] đề xuất "
           "CIM, khai thác cấu trúc cộng đồng để thu hẹp tập ứng viên trước khi "
           "tinh chỉnh. Về phía phát hiện cộng đồng, thuật toán tối ưu modularity "
           "của Clauset, Newman và Moore [15] vẫn là mốc so sánh tiêu chuẩn."))
    A(para(run("Khoảng trống chung của nhóm này. ", bold=True)
           + run("Các công trình trên đều đánh giá lợi ích của một ứng viên trong "
                 "phạm vi cộng đồng của nó. Điều đó làm thuật toán nhanh, nhưng "
                 "không công trình nào định lượng phần ảnh hưởng bị mất do cắt bỏ "
                 "cạnh bắc cầu, cũng không xử lý việc đem so sánh những giá trị "
                 "lợi ích được đo trên các đồ thị con khác nhau. Đó chính là "
                 "khoảng trống mà bài báo này nhắm tới.")))

    A(heading("2.4.  Ảnh hưởng phụ thuộc chủ đề", 2))
    A(para("Ảnh hưởng không đồng nhất giữa các chủ đề. Liu và cộng sự [18] khai "
           "thác ảnh hưởng ở mức chủ đề trên mạng không đồng nhất. Barbieri và "
           "cộng sự [17] đề xuất các mô hình lan truyền nhận biết chủ đề, trong đó "
           "xác suất trên một cạnh phụ thuộc chủ đề của sản phẩm đang lan truyền. "
           "Nền tảng cho nhóm này là phân bổ Dirichlet ẩn của Blei và cộng sự "
           "[16]. Li và cộng sự [19] bổ sung một chiều khác — mức độ a dua của "
           "người dùng — cho thấy trọng số cạnh còn phụ thuộc nhiều yếu tố ngoài "
           "cấu trúc đồ thị."))

    A(heading("2.5.  CTIM và vị trí của công trình này", 2))
    A(para("Huang và cộng sự [20] đề xuất CTIM, kết hợp hai nhánh cuối: một mô "
           "hình sinh suy ra đồng thời cộng đồng của người dùng và chủ đề của sản "
           "phẩm, từ đó tính cường độ ảnh hưởng trên từng cạnh cho từng sản phẩm "
           "cụ thể; sau đó phân bổ K hạt giống cho các cộng đồng bằng quy hoạch "
           "động, với độ lan truyền đo bằng mô hình cây ảnh hưởng cực đại của "
           "[6, 7]."))
    A(para("Công trình hiện tại không đề xuất một mô hình lan truyền mới, cũng "
           "không thay đổi cách phát hiện cộng đồng hay khung quy hoạch động. Nó "
           "chỉ ra rằng bước đo lợi ích trên đồ thị con cảm sinh — điểm chung của "
           "[12, 13, 14, 20] — là một phép xấp xỉ có cái giá đo được, và đề xuất "
           "cách gỡ bỏ nó mà vẫn giữ nguyên cấu trúc phân rã, tức vẫn giữ được "
           "phần lớn lợi thế về tốc độ mà phân rã mang lại."))

    # -------------------------------------------------------------- refs
    A(heading("Tài liệu tham khảo", 1))
    A(ref_block())
    A(para(run("Ghi chú về tính chính xác của thư mục. ", bold=True)
           + run("Mọi mục trên đã được đối chiếu với DBLP và trang của nhà xuất "
                 "bản. Ba mục trong thư mục của [20] có sai sót và được sửa ở đây: "
                 "mục [11] được [20] ghi là “pp 1593–1554”, trong đó số trang đầu "
                 "lớn hơn số trang cuối, đúng là 1539–1554; mục [14] bị thiếu tác "
                 "giả Peng WC và ghi số trang “5(2):1–25”, đúng là số bài 25; mục "
                 "[15] được ghi là “70(6):1–6”, đúng là số bài 066111. Mục [9] "
                 "không xuất hiện trong thư mục của [20] và được bổ sung ở đây vì "
                 "đó là công trình nền của nhánh lấy mẫu ngược mà [10] và [11] kế "
                 "thừa.", size=9)))
    return "".join(b)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default=os.path.join("results", "paper_sections_I_II.docx"))
    args = p.parse_args(argv)
    write_docx(args.out, build())
    print("[write] %s   (%d tai lieu tham khao)" % (args.out, len(REFS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
