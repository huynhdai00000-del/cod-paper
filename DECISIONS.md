# Nhật ký quyết định COD

Đặt ở gốc repo. Đọc file này trước khi đề xuất bất kỳ hướng đi nào.

Mục đích: chặn việc mở lại những câu hỏi đã chốt. Mỗi mục CLOSED chỉ được mở lại khi có bằng chứng mới, và khi mở lại phải nói rõ "điều này mở lại C-n vì X".

Cập nhật lần cuối: 2026-07-27

---

## Quy tắc chống lặp

Áp dụng cho cả Claude (chat) và Claude Code.

1. Trước khi đề xuất một quyết định, kiểm tra mục CLOSED. Nếu đã có, không đề xuất lại.
2. Không đề xuất "bàn với HH Le / KA Nguyen" cho quyết định kỹ thuật. Xem C-2.
3. Không nhắc lại claim của bản thảo như thể là sự thật. Bản thảo bị desk-reject vì chính những claim đó. Xem C-8.
4. Mỗi lượt trả lời phải nối được vào một mục OPEN cụ thể. Nếu không nối được, nói thẳng là đang lạc đề.
5. Khi phát hiện điều mâu thuẫn với một mục CLOSED, ghi vào mục "Bằng chứng mới" ở cuối, không tự ý mở lại.

---

## CLOSED — không mở lại

### C-1. Hướng reframe: engineering application
HH Le và KA Nguyen đã chốt. Bỏ framing "general math contribution", đưa theorem xuống vị trí hỗ trợ, mở bài bằng bài toán vận hành.

### C-2. Vai trò: Huỳnh Đại là người implement duy nhất
HH Le và KA Nguyen chỉ giám sát và định hướng. Mọi quyết định kỹ thuật và thiết kế thí nghiệm do Đại quyết. Không đề xuất "hỏi ý hai thầy" cho những việc này. Hệ quả: quy mô thí nghiệm phải vừa sức một người, không có HPC, tự túc compute.

### C-3. Giữ nguyên cấu trúc cascade và đủ sáu state
Không bỏ state nào. Không thay cascade bằng latent hay cách khác.

### C-4. Cửa sổ 12 giờ là đơn vị dự báo, rollout là cách đánh giá
Lý do vật lý: 12h = 4,8 lần τ_oil = 150 phút, đủ phân giải chu kỳ nhiệt theo tải. Vì tốc độ sinh khí là hàm điểm của trạng thái nhiệt, phân giải tốt chu kỳ nhiệt đồng nghĩa phân giải tốt quá trình sinh khí. Vì V_arr lồi, tích phân trên quỹ đạo phân giải khác giá trị tại nhiệt độ trung bình (±15°C quanh 100°C cho chênh 2,5 lần), nên cửa sổ ngắn là bắt buộc chứ không phải lựa chọn rẻ tiền.

Không train ở chân trời năm. Chân trời dài chỉ xuất hiện ở khâu đánh giá qua rollout nhiều cửa sổ. Không tăng chi phí train.

Claude đã mở lại mục này hai lần. Không mở lại nữa.

### C-5. Hạ tầng: package Python + Git + Colab
`C:\dev\cod-paper`, push GitHub, Colab clone và chạy. Đã kiểm chứng round-trip: config hash `dfa87a9e2973b55f` trùng khớp local và Colab.

### C-6. Port Phase 1 và Phase 2 đã hoàn tất
Ba gate pass. Tập huấn luyện sinh lại khớp từng byte. Năm fix, mỗi fix một commit. Fix 1, 4, 5 đổi training distribution nên checkpoint cũ vô hiệu.

### C-7. AMORE hoãn
Không có code công khai, độ khó cao, một người implement thì đó là dự án riêng. Nêu ở Limitation, ghi rõ đã liên hệ tác giả.

### C-8. Nguyên lý decomposition KHÔNG mới
Đã tồn tại: Universal Differential Equations (Rackauckas 2020), hybrid/gray-box modeling (từ 1990s), operator splitting khai thác cấu trúc tam giác, CASCADE (2026, mạng chỉ học phần hiệu chỉnh subgrid), và ref [55] của chính đồng tác giả HH Le cho stiff DAE.

Bài phải trích những nguồn này và định vị COD là một instantiation cho lớp bài toán cụ thể. Novelty nằm ở ứng dụng và ở structural guarantee, không nằm ở decomposition. Giữ claim cũ thì venue nào cũng gặp lại phản biện của JCP.

### C-9. Metric: đơn vị vật lý là chính
MAE tuyệt đối (°C, ppm, DP) làm chỉ số chính, NMAE làm phụ kèm tỷ lệ chạm sàn. Ba tầng test không bao giờ gộp. Phần pin trong repo đã làm đúng, áp dụng cách đó cho máy biến áp.

---

## OPEN — đang chờ

Mỗi mục ghi rõ cần gì để đóng.

### O-1. Gas output của monolithic có đi qua Arrhenius quadrature không
Brief đã soạn, chưa chạy. Đóng khi có `audit_port/AMPLIFICATION_MECHANISM.md`.

Vì sao quan trọng: θ_TO của Mono tệ hơn COD 33,6 lần (13,41 vs 0,399 °C) nhưng c_C2H2 chỉ tệ hơn 1,19 lần (0,705 vs 0,593 ppm). Nếu khuếch đại Arrhenius đang hoạt động thì 13,41 °C phải cho sai số khí cỡ 20 ppm. Nghi vấn: chuỗi khuếch đại mà Section 7.1 mô tả không tồn tại trong forward pass của baseline. Nếu đúng, Section 7.1 và Table 4 phải viết lại.

### O-2. Phân bố sai số tuyệt đối
Bảng hiện dùng mean MAE với median denominator. Cần median, mean, p90, max, kèm case index của max, cho cả COD và Mono. Ưu tiên thấp, gộp vào lúc thiết kế metric cho benchmark mới.

### O-3. Nguồn của k_gen, k_dis, E_act, A_SEI, E_SEI
IEC 60599 là hướng dẫn diễn giải chẩn đoán, không quy định hằng số động học. Ba phương án: tìm nguồn thật trong literature, khai báo là giá trị giả định của benchmark tổng hợp kèm sensitivity analysis, hoặc hiệu chỉnh từ bộ Katser. Cần research tool.

### O-4. Rà soát literature về khai thác one-way coupling trong operator learning
Chưa làm đầy đủ. Cần biết trước khi viết Related Work, vì đây là chỗ JCP đã bắt lỗi. Cần research tool.

### O-5. Retrain COD trên physics đã sửa
Cấu hình `example_cod_seed1.yaml`, 12 giờ, 1 seed, khoảng 45 phút. Mục đích: xác nhận fix 1 không phá gì. Không phải số cho bài. Chạy được ngay.

### O-6. Ma trận baseline cuối cùng
Phụ thuộc O-1 và O-4. Khung hiện tại: mỗi kiến trúc chạy hai cấu hình (monolithic và trong cascade), cộng ba baseline không học được (LSODA ở dung sai khớp độ chính xác COD, IEC giải tích thuần, Arrhenius tại nhiệt độ trung bình ngày). Ứng viên: PI-DeepONet, FNO, MIONet, Goswami, và một đại diện họ UDE.

### O-7. Thiết kế thí nghiệm rollout
Reference bằng LSODA chạy liên tục toàn chân trời, không chia cửa sổ. Đo sai số theo số cửa sổ đã roll. Tách bias hệ thống khỏi sai số ngẫu nhiên. Chỉ số cuối: sai số thời điểm end-of-life theo tháng. Thử cả cửa sổ 12h và 24h.

---

## Bằng chứng đã xác lập

Những phát hiện đã kiểm chứng, dùng làm căn cứ, không cần chứng minh lại.

**J-8. Mọi checkpoint monolithic lưu `ne = 12.0`, COD lưu `0.8`.** Đọc trực tiếp từ trọng số đã train. Constructor argument `n_exp=12` shadow biến global `n_exp = 0.8`. Baseline tính trunk feature với `K^24` thay vì `K^1.6`, lệch 137 lần ở K = 1,3. Đây là lý do độc lập thứ ba khiến M-2 không đứng vững, cùng với `wm = 0.000` và loss cuối cao hơn COD năm bậc. Audit không phát hiện ra.

**Sai số tuyệt đối, seed 999, N=100.** COD: θ_TO 0,399 °C, C2H2 0,593 ppm. Mono Fair: θ_TO 13,41 °C, C2H2 0,705 ppm. Sai số khí tệ nhất của Mono bằng 2,01% ngưỡng IEC. Kết luận: thất bại của monolithic thuần là thất bại nhiệt.

**Sàn NMAE chạm trên 38% case C2H2.** Biến thiên ground truth trung vị của C2H2 trong 12 giờ là 6,73e-4 ppm. Mọi con số phần trăm khí trong bản thảo là artifact của chuẩn hóa.

**Ba con số monolithic là ba kiến trúc khác nhau**, không phải ba lần chạy: 13.199,7% (mono_fair_v2, đây là 13.200% trong bài), 18.078,4% (multi-head), 18.933,3% (mono_fair_v1, checkpoint không có, IC bị vi phạm theo cấu trúc do soft mask `sigmoid(10t/T)` bằng 0,5 tại t=0).

**Audit đếm sai M-8**: 14 case ngoài dải K, không phải 20. Bản chất phát hiện vẫn đúng.

**11 claim trong bản thảo không truy được artifact**, 5 claim sai lệch. Gồm PINN baseline (§7.2, không tồn tại), OU 0,6%, 3,0 ms, 84 ms, 28× fleet.

---

## Bằng chứng mới cần xem xét

Ghi vào đây khi phát hiện điều mâu thuẫn với một mục CLOSED. Không tự ý mở lại mục đó.

(trống)
