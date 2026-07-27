# Sườn bài mới

Cập nhật 2026-07-27. Phạm vi: chỉ máy biến áp (C-12). Ma trận baseline ở C-11 trong DECISIONS.md.

---

## Định vị

Bỏ tên "Cascaded Operator Decomposition" và bỏ mọi tuyên bố decomposition là mới (C-8). Định vị trong khung **known operator learning** (Maier et al. 2019) và **gray-box modeling** (Psichogios & Ungar 1992).

Tên đề xuất: *A structure-preserving thermal surrogate for fleet-scale transformer ageing prognostics*

---

## Cấu trúc

### 1. Introduction
Mở bằng bài toán vận hành: thiếu hụt máy biến áp, lead time 128-144 tuần, tải biến động do năng lượng tái tạo.

Giới thiệu khoảng cách Jensen (bảng ở C-10) làm gap: thực hành hiện nay dùng nhiệt độ trung bình ngày, cách đó đánh giá thấp tốc độ lão hóa 1,70 lần và sinh C2H2 2,59 lần ở dao động ±15°C.

Ba yêu cầu rút ra: quỹ đạo nhiệt phân giải theo giờ, chi phí ở quy mô đội, không vi phạm ràng buộc vật lý.

Không mở bằng operator learning. Không mở bằng stiff ODE.

### 2. Problem statement
Hệ sáu state, one-way coupling. Nêu điều kiện cascade như một **tính chất của hệ**, không phải phát hiện. Trích operator splitting và skew-product system để nói rõ cấu trúc này đã biết từ lâu.

### 3. Method
Surrogate hybrid: delta học được trên baseline IEC 60076-7 cho θ_TO, hạ nguồn tính bằng quadrature giải tích ngoài đồ thị mạng.

Ba đảm bảo trình bày **như hệ quả đã biết, có trích dẫn**:

| Đảm bảo | Cơ chế | Trích dẫn |
|---|---|---|
| IC chính xác | mask φ(t), φ(0)=0 | Lagaris et al. 1998 |
| Không âm | tích phân của hàm dương | nUDE, Philipps et al. 2024 |
| Cận sai số `L_down·ε_up` | ISS cascade | Sontag |

Định vị: một instantiation của known operator learning cho lớp bài toán này.

### 4. Bài toán sáu state thu về một state
Đóng góp thực nghiệm mới. Bảng gradient probe: `∂L_gas/∂θ = 0` chính xác trên cả năm state khí, khác 0 khi bỏ một dòng `.detach()`, thermal bit-identical giữa hai trường hợp. Cộng quan sát residual khí nhỏ hơn thermal 4 đến 7 bậc.

Kết luận đo được, không phải lập luận kiến trúc: với hệ one-way coupled, bài toán physics-informed sáu state **là** bài toán một state.

### 5. Experimental setup
Ba tầng test đóng băng trước khi train (T1 trong phân phối, T2 ngoại suy tham số, T3 ngoài họ phân phối dùng profile tải thật từ ETT). Metric đơn vị vật lý (C-9). Ba seed. Tiêu chí hội tụ định trước áp đồng nhất. Ghi nhận bệnh lý huấn luyện.

### 6. Results
1. Độ chính xác nhiệt, đơn vị °C, ba tầng test riêng biệt
2. Ma trận baseline: mỗi kiến trúc hai cấu hình (C-11)
3. Rollout một năm, sai số theo số cửa sổ đã roll
4. **Khoảng cách Jensen đo thực nghiệm**: daily-mean Arrhenius so với quỹ đạo phân giải
5. Chi phí ở quy mô đội, so với solver ở dung sai khớp độ chính xác

### 7. CHI
Giữ nguyên như bản cũ: Cobb-Douglas, bốn tiên đề, Proposition 2, Katser field validation. Cần sửa ba lỗi audit tìm ra: thực sự lấy logarithm (hiện `log_chi` không có log), dùng đủ năm khí thay vì bốn, báo cáo cả ngưỡng 90% (54%) lẫn 80% (71%) kèm giải thích vì sao bộ bốn máy biến áp bị loại.

### 8. Discussion
Tổng quát hóa bằng công thức độ nhạy `E/(Rg·T²)`, không thí nghiệm (C-12). Ghi nhận pin cho khoảng cách 1,79 ở ±15K quanh 35°C, gần bằng máy biến áp.

Phân biệt rõ với Ramirez et al. 2025 (EAAI 139:109556), đối thủ cùng miền gần nhất.

### 9. Limitations
Không có SCADA thật. Tham số động học chưa có nguồn (O-3). AMORE không benchmark được (C-7). Rollout dài chưa validate bằng dữ liệu thực. Bias nhiệt hệ thống chưa có lời giải thích (O-9).

---

## Thứ tự thực thi

| # | Việc | Mục | Phụ thuộc | Ước lượng |
|---|---|---|---|---|
| 1 | Kiểm tra gas output của monolithic | O-1 | không | 1 giờ |
| 2 | `daily_mean.py` và đo khoảng cách Jensen | O-8 | không | 2 giờ |
| 3 | Retrain COD trên physics đã sửa | O-5 | không | 45 phút |
| 4 | Chẩn đoán bias −3 °C | O-9 | 3 | 1 ngày |
| 5 | Viết `cod/eval/rollout.py` | O-7 | 3 | 1 ngày |
| 6 | Đo rollout một năm cho COD | O-7 | 5 | 1 ngày |
| 7 | Triển khai FNO và MIONet, hai cấu hình mỗi cái | C-11 | không | 3 ngày |
| 8 | Đóng băng phân phối và ba tầng test, ghi hash | | 7 | nửa ngày |
| 9 | Chạy toàn bộ ma trận, 3 seed | C-11 | 8 | 1 tuần |
| 10 | Nguồn tham số động học | O-3 | không | song song |

Việc 1, 2, 3 chạy được ngay và độc lập nhau.
