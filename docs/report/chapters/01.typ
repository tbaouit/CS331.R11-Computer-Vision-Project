#import "../lib.typ": report-image

// ĐẶT HÌNH CHƯƠNG 1:
// 1. Lưu ảnh minh họa vào docs/report/assets/.
// 2. Thay none bằng đường dẫn tương đối, ví dụ "../assets/ch01_occlusion.png".
// Nếu chưa có ảnh, báo cáo vẫn biên dịch và hiển thị khung giữ chỗ.
#let occlusion-image = none
#let task-image = none

#let illustration-slot(path, instruction) = {
  if path == none {
    block(width: 100%, inset: 14pt, stroke: 0.6pt + luma(170), fill: luma(247))[
      #set par(first-line-indent: 0pt, justify: false)
      #align(center)[
        *VỊ TRÍ CHÈN HÌNH MINH HỌA*

        #text(size: 10pt)[#instruction]
      ]
    ]
  } else {
    image(path, width: 100%)
  }
}

= Tổng quan

== Bối cảnh và động cơ nghiên cứu

Trong ảnh chụp các cảnh tự nhiên, đối tượng thường không được quan sát đầy đủ do sự chồng lấp giữa các thành phần trong cảnh. Một chiếc ghế có thể bị người ngồi che khuất, bề mặt bàn có thể bị các vật đặt phía trên che phủ, hoặc một phần cơ thể động vật có thể nằm sau đối tượng khác. Khi đó, ảnh chỉ cung cấp trực tiếp thông tin về vùng nhìn thấy; hình dạng và diện mạo của phần bị che phải được suy luận từ các dấu hiệu còn lại. Hiện tượng này đặt ra yêu cầu nghiên cứu các biểu diễn vượt ra ngoài phạm vi những điểm ảnh quan sát được.

Trong phân đoạn thông thường, miền được xác định chủ yếu là phần đối tượng xuất hiện trong ảnh. Ngược lại, biểu diễn _amodal_ hướng đến phạm vi đầy đủ của đối tượng, bao gồm cả phần nhìn thấy và phần bị che khuất. Nghiên cứu của Zhu và cộng sự đã xây dựng chú thích amodal cho các vùng trong ảnh, kết hợp thông tin về phạm vi đầy đủ và quan hệ thứ tự chiều sâu @zhu2017amodal. Tuy nhiên, xác định miền không gian của đối tượng và khôi phục nội dung RGB bên trong miền đó là hai yêu cầu khác nhau. Một mặt nạ đúng chưa bảo đảm phần được tạo sinh có chất liệu, màu sắc hoặc cấu trúc chi tiết phù hợp.

Khôi phục diện mạo đối tượng bị che khuất (_amodal appearance completion_) tập trung vào yêu cầu thứ hai. Đầu ra mong muốn là một biểu diễn hình ảnh của đối tượng sau khi phần bị che được bổ sung, trong khi thông tin quan sát được vẫn được bảo toàn. Vấn đề cần giải quyết không chỉ là tạo ra các điểm ảnh có vẻ tự nhiên, mà còn là bảo đảm chúng thuộc về đúng đối tượng đang xét. Chẳng hạn, khi đối tượng mục tiêu là chiếc bàn, vùng bị con mèo che phải được hoàn thiện thành bề mặt bàn thay vì tiếp tục sinh hình dạng của con mèo.

#report-image(
  illustration-slot(occlusion-image, [
    Chèn một ví dụ từ COCO-A: ảnh cảnh gốc, đánh dấu đối tượng mục tiêu
    và vùng bị vật khác che khuất. Ưu tiên trường hợp đơn giản như bàn hoặc ghế.
    Không dùng ảnh kết quả sinh làm RGB ground truth.
  ]),
  caption: [Ví dụ về đối tượng bị che khuất trong ảnh tự nhiên. Việc xác định đúng đối tượng mục tiêu là cơ sở để phân biệt nội dung cần khôi phục với nội dung của vật che.],
)

Sự phát triển của mô hình khuếch tán tạo điều kiện khai thác tri thức thị giác đã học để tổng hợp các vùng ảnh chưa quan sát được. Mô hình khuếch tán trong không gian tiềm ẩn thực hiện quá trình tạo sinh trên biểu diễn nén và hỗ trợ điều kiện hướng dẫn như văn bản @rombach2022ldm. Trong phạm vi đồ án, hướng tiếp cận này được khai thác để kết hợp ràng buộc hình dạng từ mặt nạ, prior về đối tượng và thông tin ngữ cảnh khi khôi phục vùng bị che.

== Các hướng tiếp cận liên quan

=== Biểu diễn hình dạng amodal và hoàn thiện diện mạo

Phân đoạn amodal và hoàn thiện diện mạo có mối liên hệ chặt chẽ nhưng khác nhau về đầu ra. Phân đoạn amodal xác định đối tượng chiếm những vị trí nào trên mặt phẳng ảnh; hoàn thiện diện mạo xác định nội dung hình ảnh tại các vị trí đó. Với chú thích COCO-A, miền đầy đủ của đối tượng có thể được sử dụng làm điều kiện hình học. Tuy nhiên, chú thích mặt nạ không cung cấp màu sắc hoặc kết cấu thật ở những vị trí bị che.

Pix2Gestalt tiếp cận bài toán bằng cách tổng hợp hình ảnh của toàn bộ đối tượng từ ảnh bị che và thông tin chỉ định phần nhìn thấy. Ảnh hoàn thiện có thể tiếp tục được sử dụng cho phân đoạn amodal và các tác vụ khác @ozguroglu2024pix2gestalt. Trong đồ án này, Pix2Gestalt được sử dụng để hỗ trợ hình thành prior về cấu trúc và diện mạo; phạm vi đầu ra cuối cùng vẫn tuân theo mặt nạ đã được cung cấp.

=== Hướng dẫn ngữ nghĩa và ngữ cảnh cho quá trình tạo sinh

Thông tin hình học chỉ ràng buộc vị trí cần khôi phục, trong khi nội dung bên trong vùng đó còn phụ thuộc vào loại đối tượng, tư thế, chất liệu và quan hệ với cảnh xung quanh. Các mô hình thị giác--ngôn ngữ cung cấp một cách biểu diễn những thông tin này thông qua mô tả văn bản. Công trình Open-World Amodal Appearance Completion kết hợp phân đoạn, phân tích che khuất và inpainting trong một hệ thống sử dụng truy vấn ngôn ngữ để xác định đối tượng @ao2025openworld. Hướng tiếp cận này gợi ý việc sử dụng suy luận ngữ nghĩa như một nguồn điều kiện bổ sung cho tổng hợp ảnh.

Đồ án tham khảo hướng tổ chức hệ thống của SCAmodal và mã nguồn được công bố trong kho dự án tương ứng @scamodalrepo. Trên cơ sở đó, nhóm triển khai một phiên bản điều chỉnh cho thiết lập được cung cấp mặt nạ, kết hợp prior từ Pix2Gestalt, hướng dẫn ngữ nghĩa bằng Qwen và khôi phục ảnh bằng Stable Diffusion 2. Quan hệ giữa các thành phần kế thừa, các điều chỉnh trong triển khai và cấu hình thực nghiệm được trình bày cụ thể ở Chương 2. Phạm vi đóng góp của báo cáo là xây dựng và đánh giá hệ thống trong thiết lập này, không phải đề xuất một kiến trúc mô hình khuếch tán mới.

== Phát biểu bài toán

=== Đầu vào và các ký hiệu

Xét ảnh màu $I$ có chiều cao $h$ và chiều rộng $w$. Sau khi chuẩn hóa để thuận tiện cho ký hiệu, $I in [0, 1]^(h times w times 3)$. Mỗi lần xử lý tương ứng với một đối tượng mục tiêu. Hai mặt nạ nhị phân cùng kích thước ảnh được cung cấp gồm $M_v$, biểu diễn phần nhìn thấy của đối tượng, và $M_h$, biểu diễn phần bị che khuất. Mặt nạ amodal $M_a$ biểu diễn toàn bộ miền của đối tượng trong khung ảnh. Với vị trí điểm ảnh $p$, các mặt nạ thỏa mãn:

$
  M_a (p) = max(M_v (p), M_h (p)), quad M_v (p) M_h (p) = 0.
$

Tương đương, nếu đầu vào cung cấp mặt nạ nhìn thấy và mặt nạ amodal, vùng bị che được xác định bởi:

$
  M_h (p) = M_a (p) (1 - M_v (p)).
$

Ảnh $I$ là ảnh cảnh quan sát được, không phải ảnh của đối tượng trong trạng thái hoàn toàn không bị che. Đặc biệt, tại vị trí $M_h (p) = 1$, giá trị $I(p)$ thường mô tả vật che và không được xem là màu thật của đối tượng mục tiêu. Sự phân biệt này quyết định cách chuẩn bị đầu vào tạo sinh và cách diễn giải các độ đo đánh giá.

=== Đầu ra và yêu cầu khôi phục

Hệ thống cần tạo ảnh diện mạo hoàn thiện $hat(I)$ cho đối tượng mục tiêu. Về mặt khái quát, quá trình suy luận được mô tả bởi:

$
  hat(I) = F_theta (I, M_v, M_h; xi),
$

trong đó $F_theta$ là hệ thống gồm các thành phần đã được huấn luyện trước và các bước xử lý ảnh; $theta$ biểu thị các tham số mô hình được giữ cố định trong thực nghiệm; $xi$ biểu thị yếu tố ngẫu nhiên của quá trình tạo sinh. Đây là mô tả ánh xạ suy luận, không hàm ý toàn bộ hệ thống là một mạng được huấn luyện chung.

Ảnh kết quả được biểu diễn dưới dạng RGBA, với kênh alpha được xác định từ mặt nạ amodal:

$
  R(p) = (hat(I)(p), M_a (p)).
$

Nội dung RGB ngoài miền đối tượng không phải mục tiêu đánh giá của bài toán khôi phục. Trong miền đối tượng, kết quả cần đáp ứng ba yêu cầu: bảo toàn diện mạo quan sát được, tạo phần thiếu phù hợp với đối tượng và duy trì sự liên tục tại ranh giới giữa vùng nhìn thấy với vùng khôi phục. Yêu cầu bảo toàn có thể được diễn đạt như sau:

$
  hat(I)(p) approx I(p) quad "khi" quad M_v (p) = 1.
$

Ký hiệu xấp xỉ cho phép những sai khác nhỏ phát sinh từ xử lý ảnh và ghép biên. Tại vùng bị che, mục tiêu là tạo nội dung hợp lý về cấu trúc, màu sắc, kết cấu và ngữ nghĩa. Không thể mặc nhiên đồng nhất nội dung hợp lý với nội dung thật đã tồn tại phía sau vật che.

#report-image(
  illustration-slot(task-image, [
    Chèn sơ đồ gồm: ảnh RGB + visible mask + hidden mask → hệ thống khôi phục
    → đối tượng RGBA. Có thể thêm amodal mask = visible ∪ hidden để thể hiện
    hình dạng đã được cung cấp. Không cần đưa chi tiết từng model vào sơ đồ này.
  ]),
  caption: [Thiết lập khôi phục diện mạo với mặt nạ cho trước. Miền đối tượng được xác định bởi chú thích, còn nội dung RGB tại vùng bị che là kết quả suy luận của hệ thống.],
)

=== Tính không xác định của nội dung bị che khuất

Một ảnh quan sát có thể tương thích với nhiều cách hoàn thiện khác nhau. Phần thân bị che của một chiếc ghế có thể có nhiều loại chân hoặc kết cấu; phần trang phục nằm sau vật khác có thể chứa họa tiết không xuất hiện trong vùng nhìn thấy. Do đó, ngay cả khi mặt nạ amodal chính xác, nội dung RGB của vùng ẩn nhìn chung không được xác định duy nhất.

Trong thiết lập sử dụng ảnh che khuất tự nhiên của COCO-A, không có cặp ảnh tham chiếu thể hiện đầy đủ RGB thật của vùng ẩn cho từng đối tượng được đánh giá. Vì vậy, báo cáo phân biệt giữa bảo toàn vùng quan sát, tính nhất quán thị giác và mức phù hợp ngữ nghĩa. Các độ đo trên vùng nhìn thấy không được sử dụng để kết luận trực tiếp rằng hệ thống tái tạo chính xác vùng bị che. Phân tích trực quan và nhận diện các dạng lỗi được xem là những phần cần thiết của quá trình đánh giá.

== Mục tiêu và câu hỏi nghiên cứu

Mục tiêu của đồ án là xây dựng một quy trình khôi phục diện mạo đối tượng với mặt nạ cho trước, đồng thời đánh giá lợi ích của việc bổ sung prior và hướng dẫn ngữ nghĩa so với inpainting trực tiếp. Việc nghiên cứu tập trung vào khả năng vận hành, chất lượng đầu ra và chi phí xử lý của hệ thống trong điều kiện thực nghiệm xác định.

Ba câu hỏi thực nghiệm được đặt ra:

+ Với cùng ảnh đầu vào và mặt nạ, hệ thống kết hợp prior, hướng dẫn ngữ nghĩa và cơ chế phản hồi tạo ra khác biệt gì so với Stable Diffusion 2 inpainting trực tiếp?
+ Các khác biệt về điểm số có tương ứng với sự cải thiện quan sát được về chất liệu, cấu trúc và tính liên tục của đối tượng hay không? Những loại đối tượng hoặc tình huống nào vẫn dẫn đến thất bại?
+ Chi phí thời gian của các bước bổ sung thay đổi như thế nào so với baseline, và lợi ích chất lượng quan sát được có tương xứng với chi phí đó trong phạm vi ứng dụng của đồ án hay không?

Phép so sánh chính đối chiếu hai hệ thống hoàn chỉnh. Do nhiều thành phần thay đổi đồng thời, kết quả của phép so sánh này không tự xác định đóng góp riêng của từng thành phần. Những kết luận về vai trò độc lập của prior, hướng dẫn ngữ nghĩa hoặc critic chỉ có thể được đưa ra khi có thí nghiệm loại bỏ thành phần tương ứng.

== Phạm vi và giả định

=== Phạm vi xử lý

Đồ án xử lý ảnh RGB tĩnh và một đối tượng mục tiêu trong mỗi lần suy luận. Mặt nạ nhìn thấy và mặt nạ vùng bị che được cung cấp từ chú thích dữ liệu. Vì vậy, phát hiện đối tượng, phân đoạn vùng nhìn thấy và dự đoán mặt nạ amodal không thuộc các đầu ra cần học hoặc đánh giá của nhánh thực nghiệm chính.

Nội dung được khôi phục giới hạn trong khung ảnh và miền amodal đã được cung cấp. Với đối tượng bị cắt tại biên ảnh, hệ thống chỉ hoàn thiện phần nằm trong miền được biểu diễn trên ảnh; báo cáo không đặt mục tiêu khôi phục toàn bộ phần ngoài khung hoặc suy luận hình học ba chiều. Đầu ra là đối tượng RGBA được tách khỏi cảnh, không phải ảnh cảnh đầy đủ sau khi loại bỏ vật che.

=== Giả định về dữ liệu và mô hình

Các mặt nạ được giả định là cùng hệ tọa độ với ảnh, có phần nhìn thấy khác rỗng và mô tả đúng đối tượng mục tiêu. Mặt nạ từ chú thích được dùng làm điều kiện hình học, nhưng vẫn có thể chứa sai số hoặc khác biệt trong cách diễn giải ranh giới. Khi kết quả không hợp lý, cần kiểm tra cả dữ liệu đầu vào lẫn hành vi tạo sinh để tránh quy toàn bộ sai lệch cho mô hình.

Hệ thống sử dụng các mô hình đã huấn luyện trước, không cập nhật trọng số trên COCO-A trong các lượt chạy được báo cáo. Thuật ngữ _training-free_ trong đồ án được hiểu theo nghĩa không huấn luyện bổ sung cho thiết lập thực nghiệm này; nó không có nghĩa các mô hình nền tảng chưa từng được huấn luyện. Tương tự, việc không fine-tune không đủ để bảo đảm dữ liệu tiền huấn luyện của mọi mô hình hoàn toàn không giao với nguồn ảnh đánh giá.

Tên lớp trong annotation phục vụ thống kê và đánh giá mức phù hợp ngữ nghĩa. Trong cấu hình so sánh chính, tên lớp này không được đưa trực tiếp vào prompt tạo sinh. Pipeline chính xây dựng hướng dẫn từ ảnh và mặt nạ, trong khi baseline sử dụng một prompt cố định. Cách thiết lập này giúp phân biệt thông tin được cung cấp cho hệ thống với thông tin chỉ được sử dụng khi đánh giá.

=== Phạm vi thực nghiệm

COCO-A được sử dụng làm nguồn ảnh và chú thích cho đánh giá. Các mẫu chạy thử phục vụ kiểm tra quá trình đọc dữ liệu và hoạt động của chương trình; tập phát triển phục vụ kiểm tra cấu hình; tập kiểm tra được sử dụng cho đánh giá cuối sau khi cấu hình được chốt. Tên gọi tập huấn luyện trong cấu trúc dữ liệu không đồng nghĩa với việc đồ án đã dùng tập này để cập nhật trọng số mô hình.

Đơn vị đánh giá là một đối tượng đi kèm ảnh và các mặt nạ tương ứng. Hai phương pháp được so sánh trên cùng danh sách đối tượng và cùng điều kiện đầu vào. Do mặt nạ đầu ra đã được cung cấp, độ trùng khớp hình học với chú thích không được xem là bằng chứng về năng lực dự đoán hình dạng. Quy tắc chọn mẫu, cấu hình suy luận, độ đo và cách tổng hợp kết quả được trình bày trong Chương 3.

== Nội dung thực hiện và bố cục báo cáo

Trong phạm vi nêu trên, nhóm thực hiện ba nội dung chính. Thứ nhất, nhóm triển khai và điều chỉnh quy trình tham khảo từ SCAmodal cho bài toán khôi phục diện mạo với mặt nạ cho trước. Thứ hai, nhóm xây dựng baseline inpainting trực tiếp bằng Stable Diffusion 2 để tạo cơ sở đối chiếu. Thứ ba, nhóm tổ chức thực nghiệm trên COCO-A, lưu kết quả theo từng đối tượng và phân tích kết hợp các độ đo, ảnh trực quan, trường hợp thất bại và chi phí xử lý. Các nội dung này được xem là công việc triển khai và đánh giá của đồ án, không phải tuyên bố về tính mới của các mô hình nền tảng được sử dụng.

Báo cáo gồm bốn chương. Chương 1 giới thiệu bối cảnh, các hướng tiếp cận liên quan, phát biểu bài toán và xác định phạm vi nghiên cứu. Chương 2 trình bày các thành phần của hệ thống, quan hệ kế thừa từ phương pháp tham khảo và cách xây dựng baseline. Chương 3 mô tả thiết kế thực nghiệm, báo cáo kết quả và phân tích những ưu điểm, hạn chế quan sát được. Chương 4 tổng hợp các kết luận trong phạm vi bằng chứng thực nghiệm và đề xuất hướng phát triển tiếp theo.
