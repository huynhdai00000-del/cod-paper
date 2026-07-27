# Nhật ký quyết định COD

Đặt ở gốc repo. Đọc file này trước khi đề xuất bất kỳ hướng đi nào.

Mục đích: chặn việc mở lại những câu hỏi đã chốt. Mỗi mục CLOSED chỉ được mở lại khi có bằng chứng mới, và khi mở lại phải nói rõ "điều này mở lại C-n vì X".

Cập nhật lần cuối: 2026-07-27 (thêm C-10, C-11, C-12; đóng O-4 và O-6; thêm O-8, O-9)

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
Lý do vật lý: 12h = 4,8 lần τ_oil = 150 phút, đủ phân giải chu kỳ nhiệt theo tải. Vì tốc độ sinh khí là hàm điểm của trạng thái nhiệt, phân giải tốt chu kỳ nhiệt đồng nghĩa phân giải tốt quá trình sinh khí. Vì V_arr lồi, tích phân trên quỹ đạo phân giải khác giá trị tại nhiệt độ trung bình, xem C-10.

Không train ở chân trời năm. Chân trời dài chỉ xuất hiện ở khâu đánh giá qua rollout nhiều cửa sổ. Không tăng chi phí train.

Claude đã mở lại mục này hai lần. Không mở lại nữa.

### C-5. Hạ tầng: package Python + Git + Colab
`C:\dev\cod-paper`, push GitHub, Colab clone và chạy. Đã kiểm chứng round-trip: config hash `dfa87a9e2973b55f` trùng khớp local và Colab.

### C-6. Port Phase 1 và Phase 2 đã hoàn tất
Ba gate pass. Tập huấn luyện sinh lại khớp từng byte. Năm fix, mỗi fix một commit. Fix 1, 4, 5 đổi training distribution nên checkpoint cũ vô hiệu.

### C-7. AMORE hoãn
Không có code công khai, độ khó cao, một người implement thì đó là dự án riêng. Nêu ở Limitation, ghi rõ đã liên hệ tác giả.

### C-8. Nguyên lý decomposition KHÔNG mới
Đã tồn tại: Universal Differential Equations (Rackauckas 2020), hybrid/gray-box modeling (Psichogios & Ungar 1992), operator splitting khai thác cấu trúc tam giác, CASCADE (2026), và ref [55] của chính đồng tác giả HH Le cho stiff DAE.

Bài phải trích những nguồn này và định vị phương pháp là một instantiation cho lớp bài toán cụ thể. Novelty nằm ở ứng dụng và ở structural guarantee, không nằm ở decomposition. Giữ claim cũ thì venue nào cũng gặp lại phản biện của JCP.

### C-9. Metric: đơn vị vật lý là chính
MAE tuyệt đối (°C, ppm, DP) làm chỉ số chính, NMAE làm phụ kèm tỷ lệ chạm sàn. Ba tầng test không bao giờ gộp.

### C-10. Trụ của bài là khoảng cách Jensen
Arrhenius lồi nên tích phân trên quỹ đạo khác giá trị tại nhiệt độ trung bình. Với dao động sin biên độ A quanh 100 °C:

| State | Ea (kJ/mol) | ±5°C | ±10°C | ±15°C | ±20°C |
|---|---|---|---|---|---|
| CO2 | 74,8 | 1,02 | 1,10 | 1,22 | 1,41 |
| CO | 87,3 | 1,03 | 1,14 | 1,31 | 1,58 |
| H2 | 112,2 | 1,06 | 1,23 | 1,55 | 2,05 |
| DP (lão hóa) | 124,7 | 1,07 | 1,29 | **1,70** | 2,37 |
| C2H4 | 137,2 | 1,09 | 1,36 | 1,88 | 2,75 |
| C2H2 | 174,6 | 1,14 | 1,62 | **2,59** | 4,42 |

Thực hành hiện nay của ngành là lấy nhiệt độ trung bình ngày rồi nhân hệ số Arrhenius. Cách đó đánh giá thấp tốc độ lão hóa 1,70 lần và sinh C2H2 2,59 lần ở dao động ±15°C. Khoảng cách tăng theo năng lượng hoạt hóa, lớn nhất ở C2H2, tức khí chỉ thị phóng điện hồ quang.

Đây là gap kỹ thuật mà bài lấp. Thay hoàn toàn framing cũ "monolithic thất bại hai bậc độ lớn", vốn đã bị bác bởi bằng chứng đơn vị vật lý.

### C-11. Ma trận baseline

**Tầng 0, không học:**
- LSODA ở dung sai khớp độ chính xác surrogate (chuẩn tốc độ công bằng, luôn ghi rõ rtol)
- IEC 60076-7 giải tích thuần (sàn trung thực)
- **Arrhenius tại nhiệt độ trung bình ngày** (thực hành hiện nay, đo trực tiếp khoảng cách Jensen, baseline quan trọng nhất và chưa từng có trong bản thảo)

**Tầng 1, mỗi kiến trúc hai cấu hình (monolithic và trong cascade), 3 seed:**
- PI-DeepONet Modified MLP (đã có code trong repo)
- FNO (github.com/neuraloperator)
- MIONet (github.com/lu-group/mionet)

Sáu lần train × 3 seed = 18 run × 45 phút ≈ 14 giờ GPU. Kaggle cho 30 giờ mỗi tuần.

Cấu hình "trong cascade" nghĩa là kiến trúc đó chỉ dự đoán θ_TO, hạ nguồn tính bằng quadrature. Nếu FNO-monolithic hỏng còn FNO-in-cascade chạy tốt thì cascade là nguyên nhân, không phải kiến trúc.

**Loại:** AMORE (C-7), Goswami (trùng vai trò với FNO và DeepONet), đại diện họ UDE (phương pháp của bài **là** một instance của UDE, benchmark với UDE là benchmark với chính mình; trích dẫn làm framework), Zanardi (không có code, phân rã theo timescale không áp dụng).

**Stretch nếu còn thời gian:** RBA-PINN (Ramirez et al. 2025, EAAI 139:109556) tái hiện trên dữ liệu tổng hợp, phục vụ lập luận amortization.

### C-12. Chỉ máy biến áp, bỏ hoàn toàn case study pin
Lý do: không thiết kế CHI cho pin nên phần pin không tham gia được nửa sau của bài (Cobb-Douglas, bốn tiên đề, Proposition 2, Appendix B, Katser validation), thành mục hụt hơi; bỏ bớt baseline gây bất đối xứng mời reviewer đặt câu hỏi; bài đã 30 trang một cột.

Giữ lập luận tổng quát trong Discussion bằng công thức độ nhạy `E/(Rg·T²)`, không chạy thí nghiệm. Ghi nhận rằng pin cho khoảng cách 1,79 ở ±15K quanh 35°C, gần bằng máy biến áp 1,70, vì Ea nhỏ hơn nhưng nhiệt độ vận hành thấp hơn.

Section 8 cũ biến mất. Table 7 rút xuống một đoạn Discussion.

---

## OPEN — đang chờ

### O-1. Gas output của monolithic có đi qua Arrhenius quadrature không
Brief đã soạn, chưa chạy. Đóng khi có `audit_port/AMPLIFICATION_MECHANISM.md`.

θ_TO của Mono tệ hơn COD 33,6 lần (13,41 vs 0,399 °C) nhưng c_C2H2 chỉ tệ hơn 1,19 lần (0,705 vs 0,593 ppm). Nếu khuếch đại Arrhenius đang hoạt động thì 13,41 °C phải cho sai số khí cỡ 20 ppm. Nghi vấn: chuỗi khuếch đại mà Section 7.1 mô tả không tồn tại trong forward pass của baseline. Nếu đúng, Section 7.1 và Table 4 phải viết lại.

### O-2. Phân bố sai số tuyệt đối
Bảng hiện dùng mean MAE với median denominator. Cần median, mean, p90, max, kèm case index của max. Ưu tiên thấp, gộp vào lúc thiết kế metric cho benchmark mới.

### O-3. Nguồn của k_gen, k_dis, E_act
IEC 60599 là hướng dẫn diễn giải chẩn đoán, không quy định hằng số động học. Ba phương án: tìm nguồn thật trong literature, khai báo là giá trị giả định của benchmark tổng hợp kèm sensitivity analysis, hoặc hiệu chỉnh từ bộ Katser. Bỏ A_SEI và E_SEI theo C-12.

### O-5. Retrain COD trên physics đã sửa
Cấu hình `example_cod_seed1.yaml`, 12 giờ, 1 seed, khoảng 45 phút. Mục đích: xác nhận fix 1 không phá gì. Không phải số cho bài.

### O-7. Thiết kế thí nghiệm rollout
Reference bằng LSODA chạy liên tục toàn chân trời, không chia cửa sổ. Đo sai số theo số cửa sổ đã roll. Tách bias hệ thống khỏi sai số ngẫu nhiên. Chỉ số cuối: sai số thời điểm end-of-life theo tháng. Thử cả cửa sổ 12h và 24h.

### O-8. Đo khoảng cách Jensen thực nghiệm
Triển khai `cod/models/daily_mean.py` và đo tỷ số giữa tích phân trên quỹ đạo phân giải và đánh giá tại nhiệt độ trung bình, trên test set thật, đối chiếu với bảng giải tích ở C-10. Báo cáo kèm biên độ dao động hot-spot thực tế của từng case, vì khoảng cách phụ thuộc biên độ. Đóng khi có `audit_port/JENSEN_GAP.md`.

### O-9. Bias −3 °C chưa có lời giải thích
Audit xác nhận bias tồn tại (`|bias|_mean = 3,09 °C`) nhưng bác lời giải thích trong bài (cấu trúc bậc thang ETC tại K = 1: hai công thức trùng nhau chính xác tại K = 1). Ở cửa sổ đơn thì 3 °C nhỏ, nhưng qua rollout với độ nhạy 10,8 %/K thì bias hệ thống này làm sai tốc độ lão hóa khoảng 30 phần trăm. Cần chẩn đoán thật trước khi công bố bất kỳ con số end-of-life nào.

---

## Đã đóng

**O-4. Rà soát literature về one-way coupling.** Đóng 2026-07-27. Xác nhận C-8, không có tuyên bố novelty nào đứng vững ở tầng phương pháp. Thu hoạch dùng được ghi ở mục Bằng chứng.

**O-6. Ma trận baseline.** Đóng bởi C-11.

---

## Bằng chứng đã xác lập

**J-8. Mọi checkpoint monolithic lưu `ne = 12.0`, COD lưu `0.8`.** Đọc trực tiếp từ trọng số đã train. Constructor argument `n_exp=12` shadow biến global `n_exp = 0.8`. Baseline tính trunk feature với `K^24` thay vì `K^1.6`, lệch 137 lần ở K = 1,3. Lý do độc lập thứ ba khiến M-2 không đứng vững, cùng với `wm = 0.000` và loss cuối cao hơn COD năm bậc. Audit không phát hiện ra.

**Sai số tuyệt đối, seed 999, N=100.** COD: θ_TO 0,399 °C, C2H2 0,593 ppm. Mono Fair: θ_TO 13,41 °C, C2H2 0,705 ppm. Sai số khí tệ nhất của Mono bằng 2,01% ngưỡng IEC. Kết luận: thất bại của monolithic thuần là thất bại nhiệt.

**Sàn NMAE chạm trên 38% case C2H2.** Biến thiên ground truth trung vị của C2H2 trong 12 giờ là 6,73e-4 ppm. Mọi con số phần trăm khí trong bản thảo là artifact của chuẩn hóa.

**Ba con số monolithic là ba kiến trúc khác nhau**, không phải ba lần chạy: 13.199,7% (mono_fair_v2, đây là 13.200% trong bài), 18.078,4% (multi-head), 18.933,3% (mono_fair_v1, checkpoint không có, IC bị vi phạm theo cấu trúc do soft mask `sigmoid(10t/T)` bằng 0,5 tại t=0).

**Audit đếm sai M-8**: 14 case ngoài dải K, không phải 20. Bản chất phát hiện vẫn đúng.

**11 claim trong bản thảo không truy được artifact**, 5 claim sai lệch. Gồm PINN baseline (§7.2, không tồn tại), OU 0,6%, 3,0 ms, 84 ms, 28× fleet.

**Mẫu thiết kế của bài đã có tên trong literature.** Known operator learning (Maier et al. 2019, Nature Machine Intelligence 1:373, DOI 10.1038/s42256-019-0077-5) và gray-box modeling (Psichogios & Ungar 1992, AIChE J 38:1499, DOI 10.1002/aic.690381003). Dùng ngôn ngữ này thay vì đặt tên mới.

**Các đảm bảo đã có tiền lệ riêng lẻ.** Mask φ(t) với φ(0)=0 cho IC chính xác là ansatz Lagaris et al. 1998 (arXiv physics/9705023). Non-negativity by construction đã có ở nUDE (Philipps et al. 2024, IFAC-PapersOnLine 58(23):25, DOI 10.1016/j.ifacol.2024.10.005). Cận `L_down·ε_up` là ISS cascade chuẩn (Sontag). Trình bày như hệ quả có trích dẫn, không như kết quả mới.

**Đối thủ cùng miền gần nhất:** Ramirez, Pino, Pardo, Aizpurua et al., Engineering Applications of Artificial Intelligence 139:109556 (2025), DOI 10.1016/j.engappai.2024.109556. PINN residual-based-attention ước lượng nhiệt độ máy biến áp rồi suy lão hóa spatio-temporal, validate bằng fiber optic sensor tại nhà máy điện mặt trời thật. Bắt buộc trích và phân biệt rõ. Dữ liệu thật của họ không công khai.

---

## Bằng chứng mới cần xem xét

Ghi vào đây khi phát hiện điều mâu thuẫn với một mục CLOSED. Không tự ý mở lại mục đó.

(trống)
