#import "lib.typ": *
// 1. Cấu hình giao diện và typography
#show: report.with(
  theme: "academic",
  draft: false,
)

// 2. Trang bìa
#report-cover(
  course-name: "Thị giác máy tính nâng cao",
  course-id: "CS331.R11",
  instructor: "TS. Mai Tiên Dũng",
  group-name: "Nhóm 4",
  members: (
    ("Nguyễn Thái Bảo", "24520173"),
    ("Bùi Huy Chương", "24520230"),
    ("Trần Thị Ngọc Quý", "24521496"),
  ),
  date: "Thành phố Hồ Chí Minh, Tháng 10, năm 2026",
  // Resolve the asset relative to main.typ, independent of the compile root.
  logo: read("assets/LOGO_UIT.PNG", encoding: none),
)

// 3. Phần mở đầu (Frontmatter)
#pagebreak()
#outline()

#list-of-tables()
#list-of-figures()
#list-of-abbreviations((
  CNN: (
    "Convolutional Neural Network",
    "Mạng nơ-ron tích chập",
  ),
  NLP: (
    "Natural Language Processing",
    "Xử lý ngôn ngữ tự nhiên",
  ),
))

#pagebreak()
#heading(level: 1, numbering: none)[MỞ ĐẦU]

Nội dung phần mở đầu đặt tại đây.
#pagebreak()
#heading(level: 1, numbering: none)[THÀNH VIÊN VÀ PHÂN CÔNG]
#set text(font: "New Computer Modern")

// 4. Nội dung các chương (Mainmatter)
#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)
#include "chapters/01.typ"

#pagebreak()
#include "chapters/02.typ"

#pagebreak()
#include "chapters/03.typ"

#pagebreak()
#include "chapters/04.typ"

// 5. Kết bài & Tài liệu tham khảo (Backmatter)
#pagebreak()
#bibliography(
  "ref.bib",
  title: "TÀI LIỆU THAM KHẢO",
  style: "ieee",
  full: true,
)
