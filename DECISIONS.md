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

**Siết thêm 2026-08-02: NMAE khí bỏ hoàn toàn, không còn là chỉ số phụ.** Đây là
thắt chặt C-9 chứ không mở lại: C-9 vẫn đúng, chỉ là "phụ" còn quá rộng cho khí.

Đo bằng `audit_port/scripts/28_gas_nmae_floor.py`, chỉ trên ground truth nên
không phụ thuộc model nào. Biến thiên trung vị trong cửa sổ 12 h, so với ngưỡng
kỹ thuật của chính state đó (`cod/eval/metrics.py`), trên tập T1 hiện thực:

| state | biến thiên trung vị | ngưỡng | tỷ lệ |
|---|---|---|---|
| theta_TO | 16,72 °C | 2 °C | **836%** |
| c_H2 | 0,00093 ppm | 2 ppm | 0,046% |
| c_C2H2 | 0,000014 ppm | 1 ppm | **0,001%** |
| c_C2H4 | 0,00023 ppm | 2 ppm | 0,012% |
| c_CO | 0,0018 ppm | 5 ppm | 0,035% |
| c_CO2 | 0,0035 ppm | 10 ppm | 0,035% |

Cả năm khí có mẫu số thấp hơn ngưỡng đo được 3 đến 5 bậc. Chia hai số đều nằm
dưới khả năng phân giải của thiết bị cho nhau là một tỷ số **hợp lệ về số học và
rỗng về vật lý**.

Hai cơ chế, phải phân biệt vì chúng cho kết luận khác nhau nếu chỉ nhìn một cái:

1. **Chạm sàn.** `eval/benchmark.py` có sàn 1e-4. Trên T1 hiện thực, sàn ăn 75%
   số case `c_C2H2` (v57: 38%), và biến thiên trung vị thật là 1,4e-05, tức
   **dưới sàn 7 lần** — con số "mẫu số trung vị 0,0001" báo ra chính là cái sàn,
   không phải tín hiệu. Phần trăm khi đó là `MAE / 1e-4`, sai số tuyệt đối đội lốt
   phần trăm.
2. **Mẫu số nhỏ nhưng trên sàn.** `c_CO` và `c_CO2` **không chạm sàn lần nào**
   (0%), nên tiêu chí sàn tha chúng — nhưng mẫu số vẫn chỉ bằng 0,035% ngưỡng.
   Tỷ số hợp lệ, vô nghĩa.

Nếu chỉ dùng tiêu chí sàn thì sẽ kết luận sai rằng CO/CO2 còn dùng được.

**Điều này đúng cả trên benchmark v57** (0,07% đến 0,92% ngưỡng), tức NMAE khí
**chưa bao giờ** có nghĩa. Con số 34.558% acetylene mà audit tìm ra là cùng một
lỗi nhìn từ phía kia. Không phải bộ lấy mẫu hiện thực làm hỏng thước đo — nó chỉ
phơi ra rằng khí gần như không đổi trong 12 giờ, mà đó là vật lý của
`k_gen·V_arr/k_dis` trên cửa sổ nửa ngày, không phải lỗi mô hình. IC nhất quán
còn làm rõ hơn vì mỗi máy khởi động gần cân bằng khí của chính nó thay vì lệch
30 °C (M-9).

**Quy tắc thao tác:** không trích bất kỳ phần trăm khí nào từ benchmark này —
không ở bảng chính, không ở phụ lục, không kèm chú thích. Báo cáo ppm tuyệt đối
so với ngưỡng IEC 60599. `theta_TO` không bị ảnh hưởng (836% ngưỡng, không bao
giờ chạm sàn) nên NMAE nhiệt vẫn dùng được. Xem
`audit_port/GAS_NMAE_FLOOR.md`.

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

**Kiểm chứng thực nghiệm (2026-07-27).** Trên phân phối hiện thực mới (biên độ
trung vị 11,20 °C), khoảng cách đo được là DP 1,39 và C2H2 1,83, khớp với nội
suy từ bảng giải tích tại 11,2 °C (1,39 và 1,85). Đường cong giải tích dự đoán
đúng đo đạc.

Hệ quả cho cách trình bày: sản phẩm là **đường cong theo biên độ**, không phải
một con số. Vẽ đường cong kèm phân phối biên độ thực tế chồng lên. Nhóm máy
bài nhắm tới (tải biến động do năng lượng tái tạo) nằm ở dải 15-20 °C, nơi
khoảng cách là 1,70 đến 2,37 trên DP.

Không trích con số từ phân phối cũ (DP 2,31; C2H2 4,76 trên trung vị) vì đó là
artifact của bộ lấy mẫu IC.

### C-11. Ma trận baseline

Quy mô đặt theo kỳ vọng Q1, không theo ngân sách compute. Compute giải quyết
bằng chạy song song nhiều tài khoản Colab, cùng ghi kết quả về một thư mục
Drive. Xem `scripts/colab_run.md`.

**Điều kiện tiên quyết:** phải cache `true_fixed_point()` vào dataset trước.
Fix 1 hiện làm mỗi epoch đắt 5,1 lần vì giải contraction trong vòng train,
trong khi điểm bất động chỉ phụ thuộc (K, T_a) là input cố định của từng
sample. Không cache thì ma trận này mất 160 giờ GPU thay vì 30.

**Tầng 0, không học:**
- LSODA ở dung sai khớp độ chính xác surrogate (chuẩn tốc độ công bằng, luôn
  ghi rõ rtol; so với rtol=1e-8 là so với thứ chính xác hơn bảy bậc)
- IEC 60076-7 giải tích thuần (sàn trung thực)
- Arrhenius tại nhiệt độ trung bình ngày (thực hành hiện nay của ngành, đo
  trực tiếp khoảng cách Jensen ở C-10; baseline quan trọng nhất và chưa từng
  có trong bản thảo)

**Tầng 1, neural operator, mỗi kiến trúc hai cấu hình (monolithic và trong
cascade), 5 seed:**
- PI-DeepONet Modified MLP (đã có code trong repo)
- FNO (github.com/neuraloperator) — Li et al. 2021, ICLR
- MIONet (github.com/lu-group/mionet) — Jin, Meng & Lu 2022
- S-DeepONet (github.com/Jasiuk-Research-Group/S-DeepONet) — He, Kushwaha, Park,
  Koric, Abueidda, Jasiuk, EAAI 127 (2024) 107258. **Không được bỏ.** Đây là kiến
  trúc duy nhất trong ma trận được thiết kế riêng cho **tải phụ thuộc thời gian**,
  tức đúng bài toán này, nên thiếu nó là mời đúng câu hỏi reviewer sẽ hỏi. Thứ tự
  triển khai: sau FNO và MIONet.

Cả bốn paper đã có trong `reference/papers/`. Không kiến trúc nào được dựng từ
trí nhớ — đọc paper trước khi port, và ghi vào PORT_LOG tham số nào lấy từ paper,
tham số nào phải tự chọn (xem J-90).

4 kiến trúc × 2 cấu hình × 5 seed = 40 run. Cấu hình "trong cascade" nghĩa là
kiến trúc đó chỉ dự đoán θ_TO, hạ nguồn tính bằng quadrature. Nếu
FNO-monolithic hỏng còn FNO-in-cascade chạy tốt thì cascade là nguyên nhân,
không phải kiến trúc. Đây là bằng chứng reviewer sẽ đòi.

**Tầng 2, per-profile, cho lập luận amortization:**
- PINN per-profile, 10 profile (thay thế PINN baseline không tồn tại ở §7.2)
- RBA-PINN (Ramirez et al. 2025, EAAI 139:109556) tái hiện trên dữ liệu tổng
  hợp. Đối thủ cùng miền gần nhất, cùng dùng nhiệt độ học được rồi suy lão
  hóa. Reviewer chắc chắn hỏi tại sao không so. Dữ liệu thật của họ không
  công khai, nêu rõ giới hạn.

**Tầng 3, chuỗi thời gian thuần dữ liệu:**
- LSTM hoặc GRU. Thứ một kỹ sư sẽ thử đầu tiên; không có nó thì bài thiếu
  điểm quy chiếu thực dụng.

**Giao thức công bằng:** ngân sách theo wall-clock, không theo epoch (audit
B-1: cùng 25.000 epoch nhưng thời gian chạy chênh 4,6 lần). Mỗi baseline được
một đợt tìm siêu tham số bằng ngân sách của phương pháp chính, và đợt tìm đó
phải báo cáo. Trích Lu et al. 2022 (CMAME 393:114778, 16 benchmark, FAIR) làm
khung. Model không hội tụ báo cáo là không hội tụ kèm learning curve, không
quy thành con số hiệu năng.

**Ngân sách wall-clock: ĐỀ XUẤT 10.800 s (3 giờ), chờ chốt (2026-08-03).**
C-11 nói cách lấy số là "đặt ở mức COD hội tụ thoải mái, rồi áp đúng con số đó
cho mọi kiến trúc". Nay đã có dữ liệu để làm điều đó.

**Dữ liệu.** O-5 (`artifacts/o5/run.json`, Tesla T4): hội tụ `converged_plateau`
ở **11.900 epoch / 4.911 s**, tức 0,4127 s/epoch, dưới trần 7.200 s và bỏ không
32%. Tiêu chí cần 20 × 100 = **2.000 epoch cao nguyên** để tuyên bố hội tụ, nên
phần "học thật" là khoảng 9.900 epoch.

**Đề xuất 10.800 s**: 2,2 lần thời gian COD thực sự cần, đủ biên cho biến động
giữa 5 seed; tròn theo giờ nên khớp giới hạn phiên Colab; không rộng đến mức lãng
phí trên 40 lần chạy.

**Hệ quả cho các ô đắt hơn mỗi bước.** Đo tương đối trên CPU, cùng batch và cùng
số điểm collocation. GPU sẽ khác vì FFT và GRU song song hoá khác nhau — đây là
số để **xếp hạng**, không phải để ngoại suy:

| ô | tham số | s/bước tương đối | epoch ước tính ở 10.800 s |
|---|---|---|---|
| COD | 154.178 | 1,00 | ~26.000 (chạm trần epoch) |
| `fno_in_cascade` | 550.535 | **1,91** | **~14.100** |
| `fno_monolithic` | 551.180 | 1,79 | ~15.000 |
| `mionet_in_cascade` | 363.807 | 0,16 | chạm trần epoch |
| `mionet_monolithic` | 564.812 | 0,33 | chạm trần epoch |
| `sdeeponet_in_cascade` | 798.996 | 1,17 | ~22.000 |
| `sdeeponet_monolithic` | 849.501 | 1,25 | ~21.000 |

**FNO là ô rủi ro nhất, phải kiểm trước.** Ở 10.800 s nó chỉ được ~14.000 epoch
so với 11.900 mà COD cần. Nếu FNO cần nhiều epoch hơn COD — hoàn toàn có thể vì
kiến trúc khác — ngân sách sẽ chặn nó, và một baseline bị ngân sách chặn không
nói gì về kiến trúc.

**Do đó: chạy thử MỘT seed FNO trước khi cam kết cả ma trận.** 3 giờ để biết
`stop_reason` là `converged_plateau` hay `wall_clock_budget`. Nếu là cái thứ hai,
nâng ngân sách rồi chạy lại từ đầu — bỏ 3 giờ để khỏi vứt 120 giờ.

MIONet rẻ tới mức chạm **trần epoch 25.000** trước khi hết giờ. Không phải lỗi:
báo `stop_reason = epoch_budget` và ghi rõ nó **không** bị ngân sách thời gian
chặn. Không nâng riêng trần epoch cho MIONet — làm thế là cho một kiến trúc nhiều
bước hơn kiến trúc khác, đúng thứ mà giao thức wall-clock loại bỏ.

**Ghi chú lịch sử (2026-07-30):** trước khi có số trên, mục này ghi rằng ngân
sách là quyết định riêng chưa lấy.
`example_cod_seed1.yaml` để `max_wall_seconds: 7200`, và con số đó **không phải**
ngân sách ma trận. O-5 không phải phép so sánh: nó hỏi fix 1-9 có làm hỏng gì
không, mà model không hội tụ thì không trả lời được câu đó — sẽ lẫn "fix vật lý
làm hỏng train" với "ngân sách chặn". Khác mục đích thì khác ngân sách.

Cách lấy số cho ma trận, khi tới lúc: đặt ở mức COD hội tụ thoải mái, rồi áp
đúng con số đó cho mọi kiến trúc. Không chép 7200 từ config O-5 sang, vì nó được
chọn vì lý do khác. Đã ghi chú ngay tại chỗ định nghĩa trong config.

**Giao thức trung thực chu kỳ (thêm 2026-07-29).** Mọi model trong ma trận, kể
cả tầng 0 và tầng 3, phải báo cáo **ba** thứ chứ không chỉ MAE nhiệt:

1. bảng biên độ dao động phân tầng theo biên độ thật (tỷ số dự đoán/thật, theo
   dải 1-5, 5-10, 10-15, 15-25, 25-200 °C);
2. bảng khoảng cách Jensen dọc quỹ đạo, dự đoán so với thật, cho cả sáu state;
3. MAE nhiệt, giữ nguyên, nhưng đọc **cùng** hai bảng trên.

Lý do: MAE và biên độ đỉnh-đáy là hai phép đo khác nhau, và chỉ một trong hai là
thứ lập luận về tính lồi phụ thuộc vào. Một quỹ đạo bị làm trơn có thể gần về
MAE mà vẫn đánh mất phần khoảng cách Jensen — tức đúng thứ phương pháp này tồn
tại để giữ. `audit_port/scripts/18_swing_fidelity.py` là script chuẩn, chạy
được cho mọi model có chữ ký `model(x0, u, t)`.

Đây cũng là chỗ kiểm giả thuyết cơ chế: COD dự đoán **hiệu chỉnh** so với nghiệm
giải tích bậc một, nên hình dạng chu kỳ do baseline IEC cấp chứ không do mạng
sinh ra, và spectral bias không có gì để làm phẳng. Kiến trúc không có baseline
phải tự sinh chu kỳ, tức đúng phần tần số mà spectral bias triệt tiêu. Nếu đúng,
delta-learning trên baseline giải tích không chỉ giảm gánh nặng xấp xỉ (điều ai
cũng nói) mà còn **giữ khoảng cách Jensen trước spectral bias của mạng** — lập
luận chưa ai đưa ra. Xem N-8 về tình trạng bằng chứng hiện tại.

**Loại:** AMORE (C-7), Goswami (trùng vai trò với FNO và DeepONet), đại diện
họ UDE (phương pháp của bài là một instance của UDE, benchmark với UDE là
benchmark với chính mình; trích dẫn làm framework), Zanardi (không có code,
phân rã theo timescale không áp dụng).

### C-12. Chỉ máy biến áp, bỏ hoàn toàn case study pin
Lý do: không thiết kế CHI cho pin nên phần pin không tham gia được nửa sau của bài (Cobb-Douglas, bốn tiên đề, Proposition 2, Appendix B, Katser validation), thành mục hụt hơi; bỏ bớt baseline gây bất đối xứng mời reviewer đặt câu hỏi; bài đã 30 trang một cột.

Giữ lập luận tổng quát trong Discussion bằng công thức độ nhạy `E/(Rg·T²)`, không chạy thí nghiệm. Ghi nhận rằng pin cho khoảng cách 1,79 ở ±15K quanh 35°C, gần bằng máy biến áp 1,70, vì Ea nhỏ hơn nhưng nhiệt độ vận hành thấp hơn.

Section 8 cũ biến mất. Table 7 rút xuống một đoạn Discussion.

### C-13. Khoảng cách Jensen trình bày theo đường cong kèm phân phối thật
Không công bố một con số. Vẽ đường cong khoảng cách theo biên độ (giải tích,
C-10) chồng lên phân phối biên độ đo từ ETT. Đây là Hình 1 của bài, dựng hoàn
toàn từ dữ liệu công khai và vật lý giải tích, không cần model.

Biên độ trung vị đo được: ETTh2 (tải quy ước) 8,7% rated, 85% số ngày dưới dải
12-28% của sampler; ETTh1 không back-feed 17,8%; ETTh1 có back-feed 29,7%.
ETTh1 back-feed giữa trưa đỉnh tháng 3-6, tức đấu nối điện mặt trời, và đó là
loại máy bài nhắm tới.

Phát biểu phạm vi: khoảng cách đáng kể với máy đấu nối tái tạo và máy tải biến
động; máy chạy nền phẳng gần như không có khoảng cách nào để khai thác (N-4).

### C-14. Phạm vi: benchmark mô tả một CHẾ ĐỘ VẬN HÀNH, không phải một đội máy
Đóng O-10, 2026-08-03. **ĐÃ CHỐT 2026-08-03** — không còn là đề xuất.

**Cách trình bày trong bài, đã chốt:**

1. **Con số neo là DP 1,42 / C2H2 1,90** (ETTh1 ngày không phát ngược, K_amp
   17,8%, biên độ hot-spot thực tế 12,38 °C). Không phải 1,70 / 2,59.
2. **Phát biểu là về CHẾ ĐỘ VẬN HÀNH, không về đội máy.** Viết "với máy có chu kỳ
   tải biên độ ~18% định mức, khoảng cách là 1,42 trên DP", không viết "máy biến
   áp phân phối điển hình có khoảng cách 1,42". n = 2 máy không đỡ được câu thứ
   hai. Hai câu nghe giống nhau và chỉ một câu có bằng chứng.
3. **Trình bày trên đường cong** (C-13), với cả ba chế độ được gắn nhãn:
   - **ETTh2, 8,7%** → biên độ 6,64 °C, DP 1,113 / C2H2 1,232. Nhãn: **máy chạy
     nền, nơi khoảng cách gần như biến mất**. Phải có mặt: nó là giới hạn thành
     thật của phương pháp (N-4), và giấu nó đi thì reviewer sẽ tìm ra.
   - **ETTh1 không phát ngược, 17,8%** → 12,38 °C, **DP 1,420 / C2H2 1,896**.
     Nhãn: **điểm neo**. Feeder đo được duy nhất nằm trong dải sampler, tức điểm
     duy nhất vừa có dữ liệu thật vừa có model đã train và test tại đó.
   - **ETTh1 phát ngược, 29,7%** → 19,84 °C, DP 2,275 / C2H2 4,203. Nhãn:
     **ngoại suy dọc đường cong đã kiểm chứng tại bốn điểm**, và **dựa trên
     n = 1** — tập con ngày của một máy. Cả hai chữ này phải xuất hiện cùng con
     số, không tách ra chú thích cuối trang.

**Vì sao ngoại suy ở điểm ba là hợp lệ:** đường cong C-10 đã được kiểm tại bốn
điểm trải 6,6 đến 19,8 °C, lệch 2-5% và luôn cùng chiều (nội suy tuyến tính của
hàm lồi thì vượt lên). Đó là thứ cho phép **dẫn bài bằng đường cong** thay vì
bằng một con số, và cho phép gọi chế độ phát ngược là ngoại suy dọc một đường
cong đã kiểm chứng — chứ không phải là một phép đo.

Lập luận đầy đủ và chỗ yếu nhất ở §4 bên dưới; giữ nguyên để phản đối được.

**Số đo.** `26_prefix7_arm_and_ett_gap.py --gap-only`, N=400, seed 999, biên độ
hot-spot đo bằng RK45, gap đo dọc quỹ đạo thật:

| quần thể | K_amp | biên độ (p25/med/p75) | gap DP | gap C2H2 | C-10 tại biên độ đó |
|---|---|---|---|---|---|
| ETTh2, mọi ngày | 8,7% | 4,6 / **6,64** / 10,0 | 1,113 | 1,232 | 1,142 / 1,298 |
| ETTh1, không phát ngược | 17,8% | 7,5 / **12,38** / 18,1 | 1,420 | 1,896 | 1,485 / 2,082 |
| **sampler đóng băng** | 12-28% | 7,5 / **13,03** / 18,9 | 1,469 | 2,079 | 1,538 / 2,207 |
| ETTh1, phát ngược | 29,7% | 11,2 / **19,84** / 31,5 | **2,275** | **4,203** | 2,348 / 4,361 |

**§1. Sampler đóng băng đại diện cho cái gì.** Một **feeder phân phối có chu kỳ
tải thông thường**. Biên độ hot-spot trung vị 13,03 °C của nó gần như trùng với
ETTh1 ngày không phát ngược (12,38 °C) — hai con số cách nhau 5%, và đó là feeder
thật duy nhất trong khoảng của nó. Không phải là suy luận: sampler được hiệu
chuẩn để nhắm 10-15 °C và nó rơi đúng vào đó.

**§2. Nó KHÔNG đại diện cho cái gì.** Hai đầu, và cả hai đều phải nói ra:

- **Máy chạy nền phẳng.** ETTh2 ở 8,7% cho gap DP 1,113 — gần như không còn gì
  để khai thác, khớp tuyên bố phạm vi N-4. Với lớp máy này phương pháp không có
  lợi thế nào đáng kể so với tính ở nhiệt độ trung bình, và bài phải nói thẳng
  chứ không im lặng.
- **Máy đấu nối điện mặt trời phát ngược.** Ở 29,7% gap là DP **2,275** và C2H2
  **4,203**, tức **cao hơn** headline 1,70 / 2,59 chứ không thấp hơn. Đây là chỗ
  lật ngược lo ngại ban đầu của O-10: mối lo là sampler **lạc quan**, và so với
  ETTh2 thì đúng, nhưng so với **chính lớp máy bài mở đầu** (C-13: PV phát ngược
  là "đúng loại máy bài nhắm tới") sampler lại **bảo thủ**. 29,7% nằm ngoài dải
  12-28% của sampler, nên model chưa từng được train hay test ở đó.

**§3. Con số bài nên trích.** Đề xuất: **trích đường cong, không trích một số**,
đúng như C-13 đã chốt — và khi buộc phải có một con số, trích **DP 1,42 / C2H2
1,90** của ETTh1 không phát ngược, không phải 1,70 / 2,59.

Lý do trích điểm đó chứ không phải điểm khác:
- Nó là feeder **đo được**, không phải cấu trúc tổng hợp. 1,469 / 2,079 của
  sampler gần bằng nhưng là một phân phối do ta tự dựng.
- Nó nằm **trong** dải sampler, tức là điểm duy nhất vừa có feeder thật vừa có
  model đã được train và test tại đó. Trích một gap ở chế độ mà benchmark không
  phủ là đúng loại vượt quá đã khiến bài bị desk-reject.
- 1,70 / 2,59 là C-10 tại ±15 °C, và **không feeder nào trong hai feeder đo được
  đạt tới 15 °C** ở chế độ thông thường. Trích nó như giá trị điển hình là sai.

**Đường cong C-10 đã được kiểm ở bốn điểm** trải 6,6 đến 19,8 °C, lệch 2-5% và
luôn cùng chiều (nội suy tuyến tính của hàm lồi thì vượt lên). Đó là điều cho
phép bài **dẫn bằng đường cong**: chế độ phát ngược khi đó là **ngoại suy dọc một
đường cong đã kiểm chứng**, phải ghi rõ là ngoại suy chứ không phải phép đo.

**§4. Chỗ yếu nhất, để phản đối.** n = 2 máy, một bộ dữ liệu, một vùng, hai năm.
Không ước lượng được phân phối đội máy từ đó, và mọi phát biểu dạng "x% đội máy
nằm trong dải này" đều không có cơ sở. Vì vậy phát biểu phải là **chế độ vận
hành** ("với máy có chu kỳ tải biên độ ~18% định mức, khoảng cách là 1,42 trên
DP"), không phải **tuyên bố đội máy** ("máy biến áp phân phối điển hình có khoảng
cách 1,42"). Hai câu nghe giống nhau và chỉ một câu có bằng chứng.

Phản đối hợp lý mà tôi không bác được: ngày phát ngược của ETTh1 là tập con của
**một** máy, nên 29,7% và gap 2,275 kèm theo còn mỏng hơn cả 17,8%. Ai muốn lấy
2,275 làm headline sẽ dựa lên n=1. Ngược lại ai muốn giữ 1,70 / 2,59 phải giải
thích vì sao trích một biên độ mà không feeder đo được nào chạm tới ở chế độ
thường. Tôi chọn 1,42 vì nó là điểm ít giả định nhất, không phải vì nó lớn nhất
hay nhỏ nhất.

Xem `audit_port/PREFIX7_ARM_AND_ETT_GAP.md` và
`audit_port/ETT_LOAD_CALIBRATION.md`.

---

## OPEN — đang chờ

**O-1. Cơ chế khuếch đại.** Đóng 2026-07-27. Gas output của monolithic là
output trực tiếp của mạng, xác nhận ba cách: trace tĩnh, gradient test (0/28
tensor của COD nhận gradient từ loss khí, 30/30 của Mono Fair), và phép thử
lai. Đẩy θ_TO của monolithic qua cascade của COD cho sai số khí **thấp hơn**
chính đầu ra của nó, 1,12 đến 1,60 lần, trên mọi khí. Nếu cascade khuếch đại
thì bản lai phải tệ hơn.

13,41 °C qua cascade thật chỉ cho tối đa 0,63 ppm. Các con số 1.000 đến
35.000% ở Section 7.1 là **thay đổi tương đối của tốc độ V_arr**, đúng số học
nhưng sai đại lượng khi trích như sai số nồng độ. Section 7.1 và Table 4 phải
viết lại hoàn toàn.

### O-2. Phân bố sai số tuyệt đối
Bảng hiện dùng mean MAE với median denominator. Cần median, mean, p90, max, kèm case index của max. Ưu tiên thấp, gộp vào lúc thiết kế metric cho benchmark mới.

### O-3. Nguồn của k_gen, k_dis, E_act — ĐÓNG 2026-07-29 dưới dạng GIỚI HẠN
IEC 60599 là hướng dẫn diễn giải chẩn đoán, không quy định hằng số động học. Ba phương án: tìm nguồn thật trong literature, khai báo là giá trị giả định của benchmark tổng hợp kèm sensitivity analysis, hoặc hiệu chỉnh từ bộ Katser. Bỏ A_SEI và E_SEI theo C-12.

**Đóng bằng phương án 2: khai báo là giả định, không hiệu chỉnh.** Lý do là kết
quả trung tâm không phụ thuộc vào chúng. Khoảng cách Jensen là **tỷ số** giữa
tích phân Arrhenius trên quỹ đạo và giá trị của nó tại nhiệt độ trung bình, nên
`k_gen` và `k_dis` xuất hiện ở cả tử số lẫn mẫu số và triệt tiêu. Chỉ còn `Ea`.

Mà `Ea` thì có nguồn: giá trị DP 124,7 kJ/mol đúng bằng `B = 15000 K` của IEEE
C57.91, và các giá trị khí nằm trong dải đã công bố cho phân hủy dầu và
cellulose. Tham số không chắc chỉ ảnh hưởng **mức nồng độ tuyệt đối**, và phân
tích độ nhạy ở Hình 11 đã phủ phần đó.

Viết vào bài như một giới hạn đã khoanh vùng: hằng số động học là giả định của
benchmark tổng hợp, kết quả Jensen bất biến với chúng, mức tuyệt đối thì không
và được báo cáo kèm độ nhạy. Không mở lại để đi hiệu chỉnh.

### O-5. Retrain COD trên physics đã sửa
Cấu hình `example_cod_seed1.yaml`, 12 giờ, 1 seed, khoảng 45 phút. Mục đích: xác nhận fix 1 không phá gì. Không phải số cho bài.

**Đã gỡ chặn 2026-07-29 (fix 9, commit 1386af1).** Trước đó O-5 bị chặn chứ
không phải chỉ chờ: `run.py` sinh dữ liệu bằng sampler v57 cũ trong khi mọi con
số Jensen ở `audit_port/` tính trên `build_realistic_set`. Train một phân phối
rồi báo cáo từ một phân phối khác — đúng loại lệch mà audit tìm ra ở bản thảo
gốc. Giờ `distribution.sampler.kind = realistic`, đủ 22 tham số nằm trong khối
được băm, `from_config` từ chối config thiếu **hoặc** thừa khoá.

Train trên hash **`fc4cb76c3b32ec17`**. Đóng băng chỉ có nghĩa vì chưa có gì
train trên nó; đóng băng sau khi train là biên bản, không phải giao thức.

Sau khi retrain, chạy lại theo thứ tự: `18_swing_fidelity.py` (số hiện tại là
của v57 trong phân phối v57), rồi `19_verify_bias_fix.py` với model thật để lấy
sai số nhiệt rollout thật — thứ O-9 còn để mở và là thứ quyết định có công bố
được con số end-of-life hay không.

### O-7. Thiết kế thí nghiệm rollout
Reference bằng LSODA chạy liên tục toàn chân trời, không chia cửa sổ. Đo sai số theo số cửa sổ đã roll. Tách bias hệ thống khỏi sai số ngẫu nhiên. Chỉ số cuối: sai số thời điểm end-of-life theo tháng. Thử cả cửa sổ 12h và 24h.

**Tiến độ 2026-08-03: ba trong năm yêu cầu đã xong, hai còn lại.**
`audit_port/scripts/24_rollout_thermal_error.py`, báo cáo ở
`audit_port/ROLLOUT_THERMAL_ERROR.md`.

Đã xong:
- **Reference chạy liên tục toàn chân trời.** `reference_rollout` mang trạng thái
  qua từng cửa sổ không reset, cùng `window_forcing` và cùng cầu phương DP
  `_advance_dp` với model, nên hiệu số là sai số model chứ không phải sai khác
  thiết lập. **Lệch khỏi mô tả gốc: dùng RK45 (`rtol=1e-9`) chứ không phải
  LSODA.** Lý do: RK45 trên `fast_rhs_np` là tham chiếu mà mọi chỉ số khác trong
  repo đang chấm theo, nên đổi sang LSODA ở riêng chỗ này sẽ đưa vào một khác
  biệt integrator ngay tại phép đo lẽ ra chỉ đo model. LSODA thuộc về mục chuẩn
  tốc độ ở tầng 0 của C-11, không thuộc về đây.
- **Sai số theo số cửa sổ đã roll.** §3 của báo cáo, chia bốn phần chân trời.
- **Tách bias hệ thống khỏi ngẫu nhiên.** Ba rollout trên cùng forcing:
  reference, tự chạy (mang trạng thái của chính nó), và teacher-forced (reset về
  trạng thái reference mỗi cửa sổ). `free − teacher` chính là phần tích lũy.

- **Sai số end-of-life theo tháng — ĐO ĐƯỢC 2026-08-03 tại K=1,10.** 1400 cửa sổ:

  | | EOL (tháng) |
  |---|---|
  | tham chiếu | **16,81** |
  | model | **16,98** |
  | chênh | **+0,17 tháng**, tức 1,0% tuổi thọ tham chiếu |

  Model dự đoán EOL **muộn hơn** sự thật, đúng chiều của bias lạnh: đọc DP cao
  hơn nên tưởng máy còn khỏe. +0,17 tháng ≈ 5 ngày trên tuổi thọ 17 tháng.

  **Cách phát biểu trong bài — phải viết đúng thế này.** Tuyên bố là **quá trình
  truyền từ sai số nhiệt sang tuổi thọ đã được đặc trưng hóa và diễn ra đúng như
  vật lý dự đoán**: bias −0,5 °C, một dấu, không tích lũy qua 1400 lần reset cửa
  sổ, dịch EOL 1,0% đúng chiều mà bias lạnh bắt buộc. Tuyên bố **không phải** là
  "sai số EOL bằng 1% ở điều kiện vận hành".

  Con số 16,8 tháng là **kịch bản đo**, không phải tuổi thọ dịch vụ. K=1,10 là
  quá tải duy trì, chọn vì đó là kịch bản duy nhất chạm EOL với chi phí khả thi;
  các tải thực tế (0,85/0,95/1,00) sau một năm DP mới xuống 974/877/740, tức còn
  hàng chục năm và vẫn `censored`. Một kỹ sư đọc "tuổi thọ 17 tháng" sẽ nhận ra
  ngay điều đó, nên bài phải nói trước, không để người đọc tự phát hiện.

  Hai lỗi phải sửa trước khi con số này có nghĩa, cả hai đều tự gây ra:
  1. `eol_months` dùng `np.argmax(inv >= target)`, mà `argmax` trên mảng toàn
     False trả về 0 — không phân biệt được với "chạm ngay cửa sổ đầu". Cộng thêm
     việc luật dừng (`DP <= DP_EOL + 1`) và định nghĩa EOL (`DP <= DP_EOL`) khác
     nhau, nên tham chiếu dừng ở DP 200,91 và điều kiện không bao giờ đúng. Kết
     quả báo ra là **EOL 0,0 tháng**. Đã sửa: tìm điểm cắt tường minh và đọc EOL
     ở đúng ngưỡng mà rollout dừng.
  2. `run_scenario` cắt rollout của model theo độ dài của tham chiếu. Model bias
     lạnh thì luôn chạm EOL **muộn hơn**, nên EOL của model bị censor **do cấu
     trúc chứ không do chân trời** — đúng vào trường hợp đáng quan tâm nhất. Đã
     sửa: model chạy tới EOL của chính nó, chuỗi sai số so trên phần chân trời
     chung.

Chưa xong:
- **EOL ở tải thấp.** K=0,85/0,95/1,00 vẫn `censored`: sau một năm DP mới xuống
  974/877/740 nên còn hàng chục năm. Chi phí đo được: ~3,2 s mỗi cửa-sổ-kịch-bản
  ở năm đầu, và burn-in bão hòa ở 730 vì forcing mùa lặp theo năm nên năm thứ hai
  trở đi chỉ còn ~0,4 s.
- **Cửa sổ 24h.** Chưa thử, mới chỉ 12h.

### O-8. Đo khoảng cách Jensen thực nghiệm
Triển khai `cod/models/daily_mean.py` và đo tỷ số giữa tích phân trên quỹ đạo phân giải và đánh giá tại nhiệt độ trung bình, trên test set thật, đối chiếu với bảng giải tích ở C-10. Báo cáo kèm biên độ dao động hot-spot thực tế của từng case, vì khoảng cách phụ thuộc biên độ. Đóng khi có `audit_port/JENSEN_GAP.md`.

**O-8. Khoảng cách Jensen thực nghiệm.** Đóng 2026-07-27. `daily_mean.py` tái
hiện bảng C-10 sai lệch tối đa 0,0048 từ chính hằng số trong code (DP 1,701 so
với 1,70; C2H2 2,594 so với 2,59). Đo trên quỹ đạo thật, khoảng cách bám theo
dự đoán giải tích ở mọi dải biên độ, đơn điệu theo biên độ, xếp thứ tự theo
năng lượng hoạt hóa.

### O-9. Bias −3 °C — ĐÃ CHẨN ĐOÁN 2026-07-28, chưa đóng
`audit_port/BIAS_DIAGNOSIS.md`. **Bias là artifact của chính thước đo, không
phải của model cũng không phải của vật lý.**

`RolloutResult.theta_bias = theta_TO_end − theta_ss_ref`. Vế đầu là nhiệt độ dầu
cuối cửa sổ; vế sau là `steady_state(K_w, Ta_w)`, tức nhiệt độ ổn định nếu giữ
mãi tải và môi trường **trung bình** của cửa sổ. Trong mỗi cửa sổ rollout áp một
chu kỳ sin đầy đủ (`Ta_w ± 2 °C`, `K_w ± 0,05`), và dầu bám theo qua trễ bậc một
với `tau_oil = 150` phút trên cửa sổ `T = 720` phút. Đúng lúc kích thích quay về
giá trị trung bình thì đáp ứng trễ **chưa** quay về. Hiệu số khác 0 ngay cả với
model sai số bằng 0.

Ba cách chứng minh khớp nhau:

1. Giải tích: gain `1/sqrt(1+(ωτ)²) = 0,607`, trễ `atan(ωτ) = 52,6°`, hệ số cuối
   cửa sổ `−gain·sin(trễ) = −0,482`. Dự đoán một offset âm, một dấu, tăng theo tải.
2. `ExactModel` (RK45 trên `fast_rhs_np`, `rtol = 1e-10`, mang đúng chữ ký gọi
   của `CODOperator`) chạy qua `chi_lifetime_rollout`: bias **−2,863 °C** tại
   K_base = 0,85 tăng đơn điệu tới **−3,876** tại 1,10, âm ở 100% của 540 cửa sổ.
   Con số 3,09 của audit nằm trong dải này.
3. Đổi tham chiếu sang điểm cuối chu kỳ của chính kích thích cửa sổ đó (đã chứa
   sẵn độ trễ): còn **−0,002 °C**. Tức 99,94% hiệu ứng là độ trễ pha.

**Tính đơn điệu theo K bác lời giải thích của bài lần thứ hai.** Bias tăng trơn
từ −2,86 lên −3,88 khi K_base đi từ 0,85 lên 1,10, không có đặc trưng gì tại
K = 1. Bậc thang ETC tại K = 1 phải cho gián đoạn **tại** K = 1.

**Manh mối formula A/B: đúng cỡ, sai hình dạng, và không nằm trên đường đi.**
`A − B` có đi qua −3 °C trong hộp vận hành. Nhưng đo dọc theo chuỗi `(K_w, Ta_w)`
thật của rollout suốt một năm, nó dao động khoảng 12 °C theo mùa, trong khi bias
đo được có sd 0,07 °C và phẳng. Ngoài ra `formula_A` không xuất hiện trên đường
đi của rollout: `chi_lifetime_rollout` chỉ nhận một tham số `steady_state` và
dùng nó cho cả `theta_ss0`, IC khí, lẫn `theta_ss_ref`.

**Hệ quả lão hóa nhỏ hơn lo ngại.** Số học 10,8 %/K đúng (`B_aging/T² = 10,77`
%/K tại 100 °C, kiểm từ chính hằng số trong code), nhưng **−3 °C chưa bao giờ đi
vào phép tính DP**: với `dp_source="model"` (mặc định), DP tiến theo
`theta_for_dp = xp[:, 0]`, tức quỹ đạo dự đoán trên 20 điểm cầu phương.
`theta_ss_ref` chỉ vào phép tính DP khi `dp_source="reference"`, nơi model vắng
mặt theo thiết kế. Trường bias được báo cáo và không ai tiêu thụ.

**Chưa đóng.** Điều này gỡ một lý do cụ thể để nghi ngờ, và thay bằng một lỗ
hổng thành thật: **sai số nhiệt thật của model qua rollout chưa từng được đo**,
vì trường lẽ ra đo nó lại đang đo thứ khác. Chưa đo được ngay: fix 6 vô hiệu
checkpoint lần nữa nên chưa có model đã train để roll. Đóng khi có model retrain
và `theta_bias` được chấm lại với tham chiếu là một phép tích phân
`fast_rhs_np` trên cùng cửa sổ từ cùng IC. Chưa sửa `rollout.py` ở đây: O-9 yêu
cầu chẩn đoán, và sửa thước đo trong cùng commit giải thích tại sao nó sai sẽ
gộp before và after vào một diff.

**ĐÓNG 2026-08-03.** Điều kiện đã nêu đã đủ: có model retrain (O-5,
`artifacts/o5/model.pt`, hash phân phối `fc4cb76c3b32ec17`, hội tụ
`converged_plateau`), và `theta_bias` được chấm lại với tham chiếu là tích phân
`fast_rhs_np` liên tục trên cùng forcing từ cùng IC. 730 cửa sổ mỗi kịch bản,
`audit_port/ROLLOUT_THERMAL_ERROR.md`.

| K_base | bias °C | sd | một cửa sổ | tích lũy |
|---|---|---|---|---|
| 0,85 | **−0,5622** | 0,077 | −0,5578 | −0,0044 |
| 0,95 | **−0,5526** | 0,070 | −0,5479 | −0,0047 |
| 1,00 | **−0,5224** | 0,068 | −0,5179 | −0,0045 |
| 1,10 | **−0,4129** | 0,067 | −0,4093 | −0,0036 |

Ba kết luận:

1. **Sai số nhiệt thật là khoảng −0,5 °C, và là sai số model thật.** Cùng thước
   đo cho `ExactModel` −0,002 °C và thu lại đúng +0,5000 khi bơm vào +0,5 °C, nên
   đây không phải artifact của thước đo nữa.
2. **Tính đơn điệu theo tải biến mất.** Thước đo cũ cho −2,86 → −3,88 tăng đều
   theo K, và mục này dùng chính tính đơn điệu đó để bác lời giải thích bậc thang
   ETC của bản thảo. Với thước đo đã sửa, bias **phẳng và hơi giảm** theo tải:
   −0,56, −0,55, −0,52, −0,41. Toàn bộ xu hướng theo tải là artifact của độ trễ
   pha, nay xác nhận trên model đã hội tụ chứ không chỉ trên `ExactModel`.
3. **Không tích lũy.** Phần tích lũy là −0,004 °C ở mọi tải, tức 0,8% của tổng.
   Qua 730 cửa sổ tự nạp lại đầu ra của chính mình, model cộng thêm 4 mK. Gần như
   toàn bộ sai số là sai số một cửa sổ. §3 cho thấy không tăng theo chân trời.

**Hệ quả lão hóa, và đây là phần phải nói thẳng.** Bias lạnh nên model đọc DP
**cao hơn** sự thật ở mọi tải: +0,15%, +0,65%, +1,23%, +2,17% sau một năm. Tức
máy trông khỏe hơn thực tế — chiều sai **không an toàn** cho một tuyên bố quản lý
tài sản. Sai số tốc độ lão hóa suy ra là **−4,5% đến −6,1%**, một dấu, ổn định
theo tải.

Còn để mở, chuyển sang O-7: **con số end-of-life vẫn chưa đo được**, cột EOL là
`censored` ở chân trời một năm. Cái đã có là một chặn: bias một dấu, không tích
lũy, −5% tốc độ lão hóa. Điều đó **không** đủ để công bố một con số EOL, chỉ đủ
để nói bias đã được đặc trưng hóa. Không đặt ngưỡng chấp nhận trong script: chưa
mục nào trong DECISIONS quyết định ngưỡng bias rollout chấp nhận được, và tự chế
ra một ngưỡng ở đây là lặp lại đúng động tác của hệ số 0,850.

**O-10. Hiệu chuẩn phân phối tải theo ETT.** Đo xong 2026-07-28,
`audit_port/ETT_LOAD_CALIBRATION.md`. **Chưa đóng**: kết quả là một quyết định
phạm vi chứ không phải một con số hiệu chuẩn, và quyết định đó chưa có.

Đo trên ETTh1 và ETTh2 (hai máy khác nhau, 17.420 giờ mỗi máy, 2016-07 đến
2018-06). Dùng công suất biểu kiến `|S| = sqrt(P² + Q²)` vì K của IEC 60076-7
là tỷ số dòng điện; so với **nửa** biên độ đỉnh-đáy vì `K = K_base + K_amp·sin`.
ETT không công bố công suất định mức nên phần "phần trăm của định mức" dựa trên
proxy `p99/0,85`, có bảng độ nhạy.

Biên độ dao động tải ngày, tính theo phần của định mức, so với K_amp = 12-28%:

| | trung vị | dưới dải | trong dải | trên dải |
|---|---|---|---|---|
| ETTh2, mọi ngày | 8,7% | 85,2% | 14,8% | 0,0% |
| ETTh1, ngày không phát ngược | 17,8% | 15,6% | 77,5% | 6,9% |
| ETTh1, ngày phát ngược | 29,7% | 0,0% | 39,1% | 60,9% |

Nói thẳng: **trên máy tải quy ước, feeder thật dao động khoảng một nửa mức
RealisticParams giả định**, nên headline Jensen tính trên bộ lấy mẫu hiện tại
là lạc quan. Trên máy còn lại thì không.

ETTh1 phát ngược giữa trưa: 0% ban đêm, 51% lúc 12h, đỉnh tháng 3-6. Đó là điện
mặt trời phát ngược, tức đúng chế độ vận hành mà bài mở đầu. Nhưng PV chỉ giải
thích 7 trên 16 điểm chênh giữa hai máy; ngày không phát ngược của ETTh1 vẫn dao
động gấp 2,0 lần ETTh2. Chênh lệch giữa các feeder không rút gọn được với n = 2.

Hai điều kèm theo, ghi lại vì quan trọng hơn chính O-10:

1. Biên độ **nhiệt độ dầu đo được** trên ETT là 2,39 và 5,60 °C (trung vị),
   trong khi bộ lấy mẫu nhắm 10-15 °C ở hot-spot. Ba lý do khiến đây là gợi ý
   chứ chưa kết luận nằm ở §5 của báo cáo — quan trọng nhất: hai máy này chạy
   **lạnh** (OT trung vị 11,4 và 26,6 °C so với `hot_spot_mean = 86`), mà độ
   tăng nhiệt tỷ lệ khoảng `K^(2n)`.
2. Do đó `tau_oil`, `DTheta_oil_R`, `n_exp` cũng ở đúng tình trạng của `k_gen`
   và `k_dis` trong O-3: lấy mặc định IEC, chưa khớp với dữ liệu nào. ETT có OT
   đo cùng tải đo nên kiểm được trực tiếp. Xem O-11.

Chưa sửa gì trong `RealisticParams`, theo đúng yêu cầu báo cáo trước.

**ĐÓNG 2026-08-03 bởi C-14.** Quyết định phạm vi đã lấy: benchmark mô tả một
**chế độ vận hành** (feeder có chu kỳ tải thông thường, ≈ ETTh1 ngày không phát
ngược), không phải một đội máy — n=2 không đỡ được tuyên bố phân phối. `K_amp`
giữ nguyên 12-28%, không rescale: N-11 đã cho thấy hệ số 0,850 dựa trên một uplift
không tái hiện được, và §1 của C-14 cho thấy dải hiện tại rơi đúng vào feeder thật
duy nhất nằm trong nó. Lo ngại "sampler lạc quan" của mục này chỉ đúng khi so với
máy chạy nền; so với lớp máy bài nhắm tới thì sampler bảo thủ.

### O-11. Hiệu chỉnh tham số nhiệt theo ETT — ĐÓNG 2026-07-29 dưới dạng GIỚI HẠN
tau_oil, DTheta_oil_R, n_exp là mặc định IEC, giả định chứ không hiệu chỉnh,
cùng vấn đề với O-3. ETT có nhiệt độ dầu đo được cùng tải đo được trên hai máy
thật. Kiểm tra trực tiếp được. Chưa làm; lớn hơn phạm vi O-10.

Liên quan: biên độ nhiệt độ dầu đo được là 2,39 °C (ETTh1) và 5,60 °C (ETTh2),
so với sampler nhắm 10-15 °C ở hot-spot. Ba lý do chưa kết luận được: hot-spot
dao động mạnh hơn top-oil (khoảng 2 lần, không đủ 4 lần cần thiết); hai máy này
chạy rất nguội (OT trung vị 11,4 và 26,6 °C so với hot_spot_mean = 86) nên độ
tăng nhiệt nhỏ theo K^1,6; và OT là số đo cảm biến, không phải theta_TO theo
định nghĩa của model. Phép đo tải chuyển được sang máy nóng hơn, phép đo nhiệt
độ thì không.

**Đóng như giới hạn, không hiệu chỉnh.** Cùng lý do hình thức với O-3: ba tham
số này định ra **mức** và **hằng số thời gian** của đáp ứng nhiệt, và bài không
tuyên bố gì về mức tuyệt đối của một máy thật. Hiệu chỉnh chúng theo ETT lại
vướng đúng ba lý do ở trên — hai máy chạy quá nguội, hot-spot không suy ra được
từ OT, và OT là số đo cảm biến chứ không phải theta_TO của model — nên phép hiệu
chỉnh sẽ đưa vào nhiều giả định hơn là nó gỡ bỏ.

Viết vào bài: tham số nhiệt lấy mặc định IEC 60076-7, benchmark là tổng hợp chứ
không phải bản sao của một máy cụ thể, và biên độ nhiệt độ dầu đo được trên ETT
thấp hơn mức tham số này ngụ ý — nêu thẳng con số 2,39 và 5,60 °C. Không mở lại
để đi hiệu chỉnh; nếu về sau có bộ dữ liệu máy chạy nóng kèm hot-spot đo được
thì đó là bài khác.

**O-12 — CODE ĐÃ SẴN SÀNG 2026-08-03, chờ train trên Colab.**
`configs/matrix/cod_no_baseline.yaml`, hash phân phối `fc4cb76c3b32ec17`, đăng ký
trong `run.py`, round-trip checkpoint đạt, smoke test qua `run.py` trên CPU đạt.
Cùng ngân sách với ma trận (10.800 s đề xuất).

**Một variable, đã kiểm chứ không giả định:** `CODNoBaseline` có **cùng số tham
số (154.178) và cùng danh sách khoá state_dict** với `CODOperator`; chỉ
`_ode_baseline` trả về `x0` thay vì nghiệm giải tích. Loop cũng phải giống —
`build_trainer` trước đây mặc định `train_v34` chỉ khi `kind == "cod"`, nên
Ablation A sẽ âm thầm nhận trainer khác; đã sửa thành `kind.startswith("cod")` và
config ghi `loop: train_v34` tường minh.

**Lỗi phải sửa trước đó, đáng ghi lại:** `CODNoBaseline._ode_baseline` mang chữ
ký cũ `(self, x0, u, t)` từ trước fix 1, trong khi `forward` và
`_thermal_predict_grid` đều gọi kèm `theta_ss_grid=`. **Mọi forward pass đều ném
`TypeError`** — class này chưa từng chạy được lần nào. Chính docstring của nó ghi
"not exercised by any verification gate", và đó là lý do lỗi tồn tại suốt từ khi
fix 1 thêm tham số. O-12 lẽ ra không thể bắt đầu.

**O-12 (nội dung gốc). Train CODNoBaseline (Ablation A) trên phân phối fix-7.** Phép thử một
biến cho luận điểm N-8: COD thay baseline H bằng hằng số x0, giữ nguyên mạng và
pipeline. Nếu nó làm trơn biên độ còn COD thì không, delta-learning trên nền
giải tích có lý do chưa ai nêu là bảo toàn khoảng cách Jensen khỏi spectral
bias. Checkpoint monolithic không dùng thay được: chúng đổi hai thứ cùng lúc
(bỏ baseline và bỏ cascade), mang lỗi J-8, và chưa train.

---

## Đã đóng

**O-4. Rà soát literature về one-way coupling.** Đóng 2026-07-27. Xác nhận C-8, không có tuyên bố novelty nào đứng vững ở tầng phương pháp. Thu hoạch dùng được ghi ở mục Bằng chứng.

**O-6. Ma trận baseline.** Đóng bởi C-11.

**O-1. Gas output của monolithic KHÔNG đi qua Arrhenius quadrature.** Đóng
2026-07-27 bởi `audit_port/AMPLIFICATION_MECHANISM.md`.

Gas của monolithic là output mạng trực tiếp: cả sáu state ra từ một `Linear(p, 6)`
duy nhất, và `V_arr`, `k_gen`, `k_dis` không xuất hiện trong `monolithic.py`. Kiểm
chứng bằng gradient: loss chỉ trên gas làm 0/28 tensor tham số của COD nhận
gradient, nhưng 30/30 của Mono Fair và 28/28 của Mono MH. Chuỗi khuếch đại mà
Section 7.1 mô tả không tồn tại trong forward pass của baseline. Section 7.1 và
Table 4 phải viết lại.

Hai kết quả kèm theo. Một, đẩy θ_TO của Mono qua cascade của COD cho sai số khí
**thấp hơn** output thật của Mono trên cả năm khí (1,12x đến 1,60x), tức cascade
không khuếch đại thảm khốc. Hai, 13,41 °C qua cascade chỉ cho tối đa 0,63 ppm,
không phải ~20 ppm như nghi vấn ban đầu; con số 1.000-35.000% là tỷ lệ thay đổi
*tốc độ* V_arr, đúng về số học, sai khi trích như sai số nồng độ.

Tiền đề "c_C2H2 chỉ tệ hơn 1,19 lần" của mục này là artifact, xem N-1.

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

**N-1. GIẢI QUYẾT 2026-07-28 bằng Phase 2 fix 6.** Reference ODE và model dùng
hàm Arrhenius khác nhau. fast_rhs_np không chặn V_arr, fast_rhs_torch và
_gas_integral clamp ở 1e4. C2H2 vượt ngưỡng tại theta_HS = 187,2 degC, test set
chạm 236,9 degC (hệ quả trực tiếp của audit M-9). Ảnh hưởng 6 trên 100 case.

Quyết định: **bỏ trần tốc độ ở phía model, lấy đúng bao nhiệt độ của
reference** `T_HS_K = clip(theta_HS + 273.15, 313.15, 573.15)`. Lý do vật lý,
không phải tiện lợi:

1. Một hằng số 1e4 duy nhất áp cho năm khí có nghĩa là năm nhiệt độ chặn khác
   nhau: 187,2 degC cho C2H2, 356,7 degC cho CO2, trải 170 degC. Không cơ chế
   bão hòa nào chặn năm phản ứng ở cùng một tốc độ không thứ nguyên. Chặn vật
   lý của động học Arrhenius là phát biểu về **nhiệt độ**, và reference đã có
   sẵn đúng phát biểu đó.
2. Clamp không phải để chống tràn số. `exp(B·e·(1/T_ref − 1/T))` tăng theo T
   với chặn trên `exp(B·e/T_ref)`, số mũ lớn nhất là 54,83 (C2H2), dưới ngưỡng
   tràn float32 là 88,7. Không tràn được ở bất kỳ nhiệt độ nào.
3. Chế độ hỏng nó nhắm tới là biên độ residual trên output mạng chưa hội tụ.
   `STATE_CLAMP_HI[0] = 200` degC đã chặn theta_TO; góc xấu nhất dựng được từ
   đó là theta_HS = 300,6 degC tại K = 1,5, tức đúng bao 573,15 K trong vòng
   một độ. Bao nhiệt độ chặn ở gần như cùng chỗ mà lại khớp ground truth theo
   cấu trúc.
4. Với bộ lấy mẫu hiện thực, hot-spot chỉ tới 179,8 degC nên 0/100 case kích
   hoạt clamp, so với 8/100 ở bộ cũ. Chế độ hỏng không còn phát sinh — nhưng
   fix không phụ thuộc điều đó, vì benchmark mà reference và model giải hai
   phương trình khác nhau thì vô hiệu bất kể mẫu hiện tại có nhận ra hay không.

Phương án thay thế (chặn cả reference) đã bác: làm ground truth phi-Arrhenius
trên một ngưỡng phụ thuộc loài khí không có cơ chế nào đứng sau, phải sinh lại
toàn bộ nhãn, và benchmark sẽ đo một động học không chuẩn nào mô tả.

Kết quả đo: 4000 trạng thái ngẫu nhiên phủ toàn hộp, float64 — trước fix 996
hàng lệch (tối đa 100% của đạo hàm), sau fix 0 hàng lệch (5,5e-14). Gate 1:
overall 1,49% → 1,26%, `c_C2H2` MAE 0,5926 → 0,1138 ppm (5,2 lần), case dưới
10% từ 99 lên 100. **Vô hiệu checkpoint lần nữa**: fast_rhs_torch là physics
residual nên mục tiêu huấn luyện đã đổi. Xem `PHASE2_EFFECTS.md` mục Fix 6.

Hệ quả cho C-10 và các lập luận: tỷ số COD/Mono trên C2H2 là 141 lần trên 94
case sạch, không phải 1,19 lần như bảng all-100 cho thấy. Con số 1,19 phải rút.

**N-2. Cache true_fixed_point.** Chi phí fix 1 từ 5,1× xuống 1,48×. Hai chỗ
dùng đắt nhất nằm trên sensor grid và không mang gradient, nên tính một lần
lúc sinh dataset. Thêm một tiết kiệm chính xác: `_thermal_predict_grid` trước
giải trên tensor (B*ns, ns), nay giải (B, ns) rồi expand. Bit-identical, ba
gate pass, `JENSEN_GAP.md` không đổi. Phần dư 1,48× là θ_ss tại query time,
gỡ được bằng implicit differentiation nhưng đã quyết không làm.

**N-3. Test set hiện tại không hiện thực về mặt vận hành.** Biên độ dao động
hot-spot trung vị 21,4 °C, 40/100 case trên 25 °C, do `sample_consistent_ic`
rút θ_TO(0) độc lập với profile tải (audit M-9). Hệ quả: trung bình Jensen
trên toàn bộ 100 case (DP 3,211; C2H2 9,505) là artifact, không phải tuyên bố
vận hành. Ở dải hiện thực ±10-15 °C, đo được DP 1,418 và C2H2 1,994. **Con số
công bố phải là bảng giải tích C-10 (1,70 và 2,59)**, không phải trung bình
thực nghiệm trên phân phối cũ. Phân phối mới phải sửa bộ lấy mẫu IC.

**N-4. Khoảng cách Jensen chỉ tồn tại khi tải biến động.** Với IC nhất quán,
case constant-K cho biên độ 0,00 °C, tức khoảng cách bằng 1,00 theo định nghĩa.
Máy chạy nền phẳng không có gì để khai thác bằng bất kỳ phương pháp nào. Đây
là tuyên bố phạm vi, khớp với framing mở bài về tải biến động do năng lượng
tái tạo. Hệ quả: tỷ lệ CK/TV 50/50 trong test set phải thiết kế lại cho phần
Jensen.

**N-5. Ngưỡng vượt IEC dư lại gần như toàn bộ là H2.** c_eq(H2) = 76 ppm ở
hot-spot 110 °C so với ngưỡng 100 ppm, trong khi hiện trường máy khỏe nằm ở
5-50 ppm. Không phải lỗi sampler mà là lỗi tham số. O-3 chuyển từ vấn đề trích
dẫn thành vấn đề hiệu chuẩn.

**N-6. Chu kỳ tải trong sampler sai tần số.** make_realistic_profile hoàn thành
trọn một chu kỳ sin trong cửa sổ 12 giờ, tức chu kỳ tải 12 giờ, trong khi tải
thật có chu kỳ 24 giờ. Một cửa sổ 12 giờ thật chỉ thấy nửa chu kỳ, biên độ quan
sát được bằng 0,66-0,79 lần biên độ ngày. Cộng với việc ETT cho thấy sampler đã
giả định biên độ gấp đôi thực tế, hai sai lệch cùng chiều.

Sửa bằng cách để chu kỳ nền là 24 giờ và cửa sổ 12 giờ cắt một đoạn với pha
ngẫu nhiên. Không đụng K_amp.

**ĐÃ SỬA 2026-07-29 (fix 7, commit 727d77c).** `make_realistic_day` +
`window_from_day`, `cycle_period = 1440`, IC lấy từ trạng thái tuần hoàn của
ngày đọc tại offset của cửa sổ. Biên độ hot-spot thực tế đo bằng RK45: trung vị
**13,18 °C** so với 11,20 của sampler cũ, `K_amp` giữ nguyên. Chi tiết ở
`audit_port/PERIOD_FIX.md`, nhật ký ở PORT_LOG J-73/J-74.

Ghi chú cho C-4: lý do chọn 12 giờ là 4,8 lần tau_oil, tức về **thời gian đáp
ứng nhiệt**, không phải về **chu kỳ cưỡng bức**. Hai thứ khác nhau, đến giờ mới
tách bạch. C-4 vẫn đứng.

**N-7. N-6 và O-9 là cùng một vật lý.** Suy giảm biên độ của hệ bậc một là
1/sqrt(1+(ω·τ)²): 0,607 ở chu kỳ 12 giờ, 0,837 ở 24 giờ, tỷ số 1,378. Sampler
phải giả định biên độ tải lớn hơn thực tế 1,38 lần để bù cho việc cưỡng bức
sai tần số. Chia K_amp = 12-28% cho 1,378 ra 8,7-20,3%, đúng dải ETT đo được
(ETTh2 8,7%; ETTh1 không back-feed 17,8%). Sửa N-6 là hòa giải sampler với dữ
liệu thật mà không cần chỉnh tay.

**Đo được sau fix 7: uplift 1,177 chứ không phải 1,378, và đó là điều phải xảy
ra.** 1,378 là gain của **sin thuần**, còn sampler là hỗn hợp. Lý do chính kiểm
được thẳng từ định nghĩa family: fix 7 **không** đổi nội dung tần số của các
family dạng sự kiện — spike vẫn 58-144 phút, evening peak vẫn rộng 130-216 phút,
chỉ vị trí là rải theo ngày — nên uplift 1,378 chưa bao giờ áp cho chúng; chỉ
`daily` và `base_load` hưởng trọn. Trung vị của hỗn hợp buộc phải thấp hơn.
Hai lý do phụ: cửa sổ 12 h chỉ thấy nửa chu kỳ ở pha ngẫu nhiên, và những cửa sổ
không chứa sự kiện nào giờ là một phần thật của quần thể.

**N-8 — DỰ ĐOÁN CƠ CHẾ BỊ BÁC 2026-08-04. Không được viết vào bài.**
Ablation A đã train xong trên phân phối fix-7 (`artifacts/o12/`,
`converged_plateau` ở 7.000 epoch / 2.835 s, hash `fc4cb76c3b32ec17`), một biến
đổi duy nhất, cùng mạng cùng trainer cùng ngân sách với COD. Chấm bằng
`18_swing_fidelity.py` trên cùng tập T1, hai model trong cùng một bảng:

| | tỷ số biên độ (trung vị) | thiếu biên độ | MAE nhiệt |
|---|---|---|---|
| COD (có baseline H) | 1,0148 | 33% | 0,384 °C |
| **Ablation A (bỏ baseline H)** | **1,0830** | **33%** | **2,282 °C** |

Phân tầng theo biên độ, Ablation A: 1,094 / 1,101 / 1,088 / 1,033 / 1,021 —
**vượt 1 ở mọi dải**, không dải nào làm trơn.

Khoảng cách Jensen dọc quỹ đạo, Ablation A so với sự thật: **+2,25% đến +6,96%
trên cả sáu state**. Không mất, mà thừa.

**Dự đoán của N-8 là "kiến trúc không có baseline giải tích phải làm trơn". Đo
được điều ngược lại.** Bỏ baseline làm MAE nhiệt xấu đi **6 lần**, nhưng **không**
làm mất khoảng cách Jensen và **không** làm trơn chu kỳ.

**Hệ quả: lập luận "delta-learning trên nền giải tích bảo toàn khoảng cách Jensen
trước spectral bias" KHÔNG được viết vào bài.** Đây là phép thử sạch nhất có thể
có — model đã hội tụ, một biến, cùng pipeline — và nó trả lời không. N-9 để mở
khả năng này ("nếu vẫn không hội tụ dưới ngân sách công bằng thì báo cáo đó là
kết quả hội tụ và bỏ tuyên bố spectral bias"); nay model hội tụ và tuyên bố vẫn
phải bỏ, vì lý do khác: không có hiện tượng làm trơn để giải thích.

**Đọc thêm, quan trọng.** Tỷ số 1,083 của Ablation A **không** phải bằng chứng
nó giữ chu kỳ tốt hơn COD. Chính N-9 đã ghi: sai số độc lập không thiên lệch làm
**tăng** biên độ đỉnh-đáy đo được, vì max trừ min của một tín hiệu nhiễu thì rộng
ra. Ablation A có MAE gấp 6 lần, nên +8% biên độ nhiều khả năng là nhiễu nở ra
chứ không phải bám chu kỳ. Đó là lý do phải đọc bảng biên độ **cùng** MAE, đúng
như giao thức C-11 yêu cầu.

**Điều còn đứng vững, và đủ mạnh để vào bài:** bỏ baseline giải tích làm sai số
nhiệt xấu đi 6 lần với **một biến duy nhất thay đổi**. Đó là kết quả sạch về
**độ chính xác**, chỉ không phải kết quả về **cơ chế**. Viết đúng như vậy.

**Cảnh báo:** n = 1 seed. Bảy seed theo runbook mới sẽ cho dải; tỷ số 6 lần khó
đảo ngược nhưng chưa đo.

**N-8 (nội dung gốc). Surrogate KHÔNG làm trơn biên độ.** Tỷ số biên độ dự đoán trên thật là
1,0038 (trung vị, n=100), hơi vượt chứ không thiếu. Phân tầng bác bỏ spectral
bias: tỷ số 1,0176 ở dải 10-15 °C, 1,0100 ở dải 25-200 °C, tức tiến về 1 khi
biên độ lớn nhất, ngược chiều với spectral bias. Khoảng cách Jensen dọc quỹ đạo
dự đoán **tăng** 0,4 đến 3,0% so với quỹ đạo thật, không mất.

Nguyên nhân là kiến trúc: COD dự đoán hiệu chỉnh trên nền nghiệm giải tích, nên
hình dạng chu kỳ do baseline mang chứ không do mạng mang.

Dự đoán kiểm chứng được: kiến trúc không có baseline giải tích phải làm trơn.
Phép thử sạch nhất là Ablation A (COD bỏ baseline H). Nếu đúng, delta-learning
trên nền giải tích có một lý do chưa ai nêu: bảo toàn khoảng cách Jensen khỏi
spectral bias, không phải chỉ giảm gánh nặng xấp xỉ.

Caveat: đo trên checkpoint v57 mà fix 6 đã vô hiệu, chỉ trong phân phối, trên
sampler cũ. Phải chạy lại sau retrain. Phương pháp thì đã lập.

**N-10. Tập test không nằm trong phân phối huấn luyện — ĐÃ SỬA 2026-08-02.**
`build_test_set` không đọc config: IC lấy từ `sample_consistent_ic` (M-9), hai
dạng sóng hardcode ở chu kỳ tải 720 phút, tức đúng lỗi N-6 mà fix 7 đã gỡ khỏi
tập huấn luyện. O-5 train trên sampler hiện thực rồi chấm trên benchmark v57.

Đo trên 9 trục (`audit_port/TEST_SET_PROVENANCE.md`), **7 trục nằm ngoài** miền
huấn luyện: theta_TO(0) tới 150 °C so với [38,7; 115,8]; lệch IC 24,82 so với
3,76 °C; **37%** case có hot-spot trung bình ngoài dải [62, 122] mà sampler kẹp
theo cấu trúc; **46%** IC khí vượt ngưỡng IEC so với 9,3%; **30%** IC vượt state
clamp so với 1,0%.

Hệ quả: nhãn `T1_in_distribution` sai, và **1,072 °C không so được với 0,399 °C
của v57** — một bên ngoài họ phân phối, một bên trong phân phối của chính nó.
Phép so đó trộn lệch phân phối với tác động của fix 1-9, đúng câu hỏi O-5 muốn
tách. O-5 vẫn chưa có câu trả lời.

Đã sửa: `build_realistic_test_set` (sampler đã đóng băng, seed tách khỏi seed
huấn luyện), tier bắt buộc khai báo `source`, `run.py` từ chối tier
`realistic_sampler` khi train bằng sampler khác và từ chối seed tier trùng
`distribution.seed`. Khối `distribution` không đụng tới:
`fc4cb76c3b32ec17` giữ nguyên. Chi tiết ở PORT_LOG J-83.

**N-11. Ba con số của fix 7 phụ thuộc seed, phải rút.** `PERIOD_FIX.md` §2 ghi
biên độ hot-spot trung vị 13,18 °C, đo ở N=200 seed 999. Đo lại 6 seed, N=500 mỗi
seed: trung vị theo seed từ 10,767 đến 12,593, **gộp N=3000 ra 11,634 °C**, sd
giữa các seed 0,668. 13,18 nằm ngoài toàn bộ dải.

Chạy cả hai chu kỳ trên cùng 6 seed, `cycle_period` là tham số duy nhất khác:
uplift đo được là **0,965** (12,053 -> 11,634), và cả 6 seed đều cho tỷ số dưới 1.
N-7 ghi 1,177, tính bằng 13,18/11,20 từ hai ước lượng một seed ở N=200 và N=100.

Ba con số phải rút hoặc tính lại: 13,18 °C; uplift 1,177; và dải `K_amp`
10,2-23,8% vốn là `K_amp` nhân 0,850 = 11,20/13,18, tức thừa hưởng cả hai sai số.
Con số thứ ba quan trọng nhất vì PERIOD_FIX §2 dùng nó để lập luận rằng phần
"sampler giả định biên độ quá lớn" là lỗi **chu kỳ** chứ không phải lỗi **biên
độ** — lập luận biến một điểm yếu thừa nhận thành một chi tiết mô hình đã giải
quyết. Nếu uplift thật gần 1,0 thì lập luận đó yếu đi và quyết định phạm vi ở
O-10 mở lại.

**ĐÃ CHỐT 2026-08-02 bằng nhánh trung thực.** Lần đo đầu dùng code hiện tại đặt
`cycle_period=720`, không phải code trước fix 7, nên lẫn chu kỳ với thời lượng sự
kiện. `26_prefix7_arm_and_ett_gap.py` dựng nhánh đúng: `cod/data/realistic.py` tại
`727d77c^`, đọc thẳng từ git. Cả hai nhánh chạy physics hiện tại (commit đó đã có
fix 1-6 và 8), `K_amp` giống hệt, sampler là khác biệt duy nhất; `cycle_period` là
trường duy nhất fix 7 thêm vào.

6 seed × N=500 mỗi nhánh:

| | trung vị gộp | sd giữa seed | dải theo seed |
|---|---|---|---|
| trước fix 7 (`727d77c^`) | **11,531** | 0,320 | 11,159-12,113 |
| fix 7 | **11,634** | 0,668 | 10,767-12,593 |
| tỷ số | **1,0089** | | 0,910-1,110 |

**Sửa chu kỳ đổi biên độ hot-spot thực tế 0,9%.** N-7 ghi 1,177; con số đó nằm
ngoài dải theo seed.

Chẩn đoán chính xác hơn "cả hai đều nhiễu": **11,20 của sampler cũ là ước lượng
đúng**, nằm trong dải 11,159-12,113. Sai là ở **13,18** — trung vị thật của nhánh
fix 7 là 11,634, và seed 999 ở N=200 rút trúng phía cao. Uplift gần như hoàn toàn
do tử số tạo ra. Đáng chú ý: sd giữa seed của nhánh mới (0,668) gấp đôi nhánh cũ
(0,320), tức cắt cửa sổ theo pha ngẫu nhiên làm thống kê nhạy seed hơn hẳn — đó là
lý do ước lượng một seed hỏng ở đúng phía đó.

Ba lý do N-7 nêu để giải thích vì sao 1,378 của sin thuần không thành hiện thực
(cửa sổ 12 h chỉ thấy nửa chu kỳ ở pha ngẫu nhiên; family sự kiện giữ thời lượng
tuyệt đối nên không hưởng gì; cửa sổ không chứa sự kiện nay là một phần thật của
quần thể) vẫn đúng — nhưng đo trung thực thì chúng **triệt tiêu** uplift chứ không
giảm nó xuống 1,177.

**Hệ quả: PERIOD_FIX §2 không đứng.** Hệ số rescale là **0,9912**, cho `K_amp`
**11,9-27,8%**, không phải 10,2-23,8%. Đối chiếu ETT: ETTh2 8,7% nằm **dưới** dải;
ETTh1 không phát ngược 17,8% nằm trong; ETTh1 phát ngược 29,7% nằm trên. §2 lập
luận rằng phần "sampler giả định biên độ quá lớn" là lỗi **chu kỳ** chứ không phải
lỗi **biên độ**, "biến một điểm yếu thừa nhận thành một chi tiết mô hình đã giải
quyết mà không phải chỉnh tay biên độ theo dữ liệu". Sửa chu kỳ đáng 0,9%. Trên
feeder quy ước, sampler giả định biên độ gấp khoảng hai lần ETTh2, và đó là một
giả định về **biên độ**. **O-10 mở lại** đúng trên điều kiện nó đã được để mở:
benchmark nói về quần thể feeder nào.

**Hai điểm vận hành hiệu chuẩn theo ETT** (C-13 yêu cầu đường cong kèm phân phối
thật, không phải một con số). `K_amp` đặt bằng đúng trung vị ETT, N=400, seed 999:

| cấu hình | K_amp | biên độ hot-spot thực tế | gap DP | gap C2H2 |
|---|---|---|---|---|
| sampler đóng băng | 12-28% | 13,03 °C | 1,469 | 2,079 |
| ETTh2, mọi ngày | 8,7% | 6,64 °C | **1,113** | **1,232** |
| ETTh1, không phát ngược | 17,8% | 12,38 °C | **1,420** | **1,896** |

Thấp hơn hẳn headline 1,70 / 2,59, vốn là C-10 tại ±15 °C — biên độ mà **không
feeder nào trong hai feeder đo được đạt tới**. Điểm ETTh1 không phát ngược gần với
thứ sampler đóng băng sinh ra, nên benchmark bảo vệ được **với tư cách mô hình của
feeder đó**; ETTh2 là nơi gap gần như biến mất, khớp với tuyên bố phạm vi N-4.

Lưu ý `K_amp` suy biến cho mọi máy cùng một biên độ tải, nên độ tản của biên độ
thực tế chỉ đến từ hỗn hợp family, điểm vận hành và pha cửa sổ. Đó là một **điểm
vận hành**, không phải phân phối đội máy, và không được trích như phân phối.

Xem PORT_LOG J-87 và `audit_port/PREFIX7_ARM_AND_ETT_GAP.md`.

**N-9. Kiểm dự đoán của N-8: đúng chiều, nhưng CHƯA lập được cơ chế.** Chạy
`18_swing_fidelity.py` trên hai checkpoint không có baseline giải tích:

| model | baseline H | tỷ số biên độ (trung vị) | thiếu biên độ | MAE nhiệt |
|---|---|---|---|---|
| COD v57 | có | **1,0121** | 14% số case | 0,51 °C |
| Mono FAIR | không | **0,6802** | **100%** | 12,69 °C |
| Mono multi-head | không | **0,6493** | **100%** | 8,81 °C |

Mất 30-80% khoảng cách Jensen dọc quỹ đạo (c_C2H2 mất 80% ở Mono FAIR). Tính
**một dấu** là phần khó giải thích bằng cách khác: sai số độc lập không thiên
lệch phải làm *tăng* biên độ đỉnh-đáy chứ không giảm, nên thiếu biên độ ở 100%
số case là dấu hiệu của làm trơn, không phải của sai số lớn.

**Nhưng hai checkpoint này không hội tụ.** MAE nhiệt 12,7 và 8,8 °C so với 0,51
của COD; riêng dải 25-200 °C thì MAE 37,8 °C, tức model không bám quỹ đạo và tỷ
số biên độ ở đó không nói gì về spectral bias. Ở hai dải model còn bám được
(10-15 và 15-25 °C, MAE 2,2-4,5 °C) thì mất 17-28% biên độ, và tỷ số **không**
xấu đi theo biên độ mà tốt lên (0,774 → 0,828 ở Mono FAIR). Sụp xuống 0,61 xảy
ra cùng lúc với MAE bùng nổ.

Audit M-2 đã kết luận lỗi monolithic *tăng* 47 lần khi capacity tăng 16 lần, với
causal weight underflow về 0 — tức "không train được baseline này", không phải
"kiến trúc này không biểu diễn được hệ". Quy phần mất biên độ cho spectral bias
là lặp đúng suy luận đó. Quy tắc repo giữ nguyên: model không hội tụ thì báo cáo
là không hội tụ. Thêm nữa mọi checkpoint monolithic đều dính lỗi J-8 (số mũ
nhiệt bị che thành 12 thay vì 0,8 lúc train).

**Ablation A không chạy được: checkpoint không tồn tại.** Cả
`ablation_a_no_baseline.pt` lẫn `transformer_pideepOnet_abl_A_no_baseline.pt`
đều không nằm trong `reference/artifacts/`; `cod/models/cod.py` đã ghi điều này
ở `CODNoBaseline`. Monolithic đổi **hai** biến cùng lúc (bỏ baseline H *và* bỏ
cascade khí), nên không phải phép thử một biến.

Kết luận thao tác: lập luận "delta-learning giữ khoảng cách Jensen trước spectral
bias" **chưa được phép viết vào bài**. Nó cần một model không-baseline **đã hội
tụ**. Đường rẻ nhất là retrain `CODNoBaseline` trên phân phối fix 7 với đúng
ngân sách của COD rồi chạy lại script này — một lần train. Nếu vẫn không hội tụ
dưới ngân sách công bằng thì báo cáo đó là kết quả hội tụ và bỏ tuyên bố
spectral bias, không dựa nó lên checkpoint hỏng.

**N-12. Ba lỗ hổng trong bộ ô đã khai báo, phát hiện 2026-08-06 khi dựng
aggregator.** Không mâu thuẫn với C-11 hay ANALYSIS_PLAN — chúng là chỗ mà hai
tài liệu đã chốt **không khớp với những gì thực sự chạy được**. Ghi lại để là lỗ
hổng đã biết chứ không phải lỗ hổng im lặng. Bảng coverage của
`scripts/aggregate_results.py` in cả ba ra mỗi lần chạy.

1. **Ô 2 của PI-DeepONet không tồn tại.** Ma trận giai thừa 2×2 cần
   `MonolithicFair` **không** baseline giải tích. J-92 gọi tên nó ("cell 2 is
   MonolithicFair") nhưng không có config nào trong `configs/matrix/`. Hệ quả:
   góc phần tư PI-DeepONet có 3 trên 4 ô, nên **không tách được** main effect
   của baseline khỏi main effect của cascade — chỉ còn simple effect.
2. **COD không nằm trong runbook.** ANALYSIS_PLAN §4 yêu cầu COD và Ablation A
   nhận đúng 7 seed như mọi baseline ("một phương pháp chính ở một seed đấu với
   baseline ở bảy seed là bất đối xứng làm hỏng phép so"). Hai vòng lặp trong
   `scripts/matrix_runbook.md` liệt kê 15 config của `configs/matrix/`; `cod`
   dùng `configs/example_cod_seed1.yaml` và **không có dòng nào chạy nó**. Hiện
   COD vẫn n = 1 (O-5).
3. **Chỉ số chính của Amendment 1 chưa có script nào sinh ra.** Amendment 1
   (2026-08-04) chuyển chỉ số chính sang **ppm khí cuối rollout 2 năm** tại
   K = 0,95 và 1,10. `24_rollout_thermal_error.py` ghi bias nhiệt, DP và EOL —
   **không ghi nồng độ khí**. Nên H1 chưa quyết định được từ bất kỳ artifact nào
   đang có, và aggregator báo đúng điều đó thay vì lặng lẽ trình bày MAE khí 12 h
   (thứ Amendment 1 đã hạ xuống thứ yếu) như thể là chỉ số chính.

Cả ba đều là việc phải làm trước khi ma trận cho ra kết luận, không phải lý do
mở lại quyết định nào. Xem PORT_LOG J-94.

**ĐÓNG CẢ BA 2026-08-06. Xem PORT_LOG J-95.**

1. **Ô 2 của PI-DeepONet: xong.** `configs/matrix/pideeponet_monolithic.yaml`.
   Không cần code — `MonolithicFair` đã nhận `use_baseline` và `run.py` đã
   truyền nó. Đo: cả hai ô monolithic của PI-DeepONet đều **156.498 tham số** và
   đều thoả IC chính xác, nên baseline là biến duy nhất. Cả hai mang J-8 **như
   nhau**, đó là lý do không sửa J-8 ở một ô. Góc phần tư PI-DeepONet lần đầu
   tiên đầy đủ, tức main effect và interaction lần đầu tiên **tách được**.
2. **COD vào runbook: xong.** Nguyên nhân đáng ghi: hai vòng lặp duyệt
   `configs/matrix/*.yaml`, mà config của COD là `configs/example_cod_seed1.yaml`
   — **không nằm trong tập được duyệt**. Khác hình dạng với J-93: ở đó biến vòng
   lặp tới được đường dẫn mà không tới được phép tính; ở đây phần tử chưa bao giờ
   ở trong vòng lặp. Cả hai để lại một bộ kết quả trông đầy đủ.
3. **Chỉ số chính của Amendment 1: có rồi.** `24_rollout_thermal_error.py` phát
   ra `gas_end` — nồng độ khí ppm cuối rollout, cho cả model lẫn tham chiếu — và
   `scripts/aggregate_results.py` đọc `<run>/rollout.json` rồi áp hai thanh chắn
   §3 lên nó ở mục **§3 PRIMARY**, hạ MAE khí 12 h xuống **§4 SECONDARY**. Cổng
   `--exact` kiểm cả hai chiều: `--inject-gas-bias 0.5` cho **+19,97 ppm** trên
   cả năm khí sau 40 cửa sổ và trả exit 1.

   Đại lượng này **vốn đã được tính rồi vứt đi**: rollout mang cả sáu state qua
   từng cửa sổ, chỉ `theta_TO` được giữ lại.

   Kèm theo, hai lỗi phải sửa để chỉ số này dùng được cho ma trận:
   `24` chỉ dựng được `CODOperator` (tức dùng được cho 1 trên 17 ô), và toàn bộ
   nhánh xuất markdown của nó chết vì `NameError`.

**Seed đã có, không chạy lại** (`38_reusable_seed_audit.py`,
`audit_port/REUSABLE_SEEDS.md`): `cod` seed 1 (`artifacts/o5`) và
`cod_no_baseline` seed 1 (`artifacts/o12`). Cả hai train bằng `train_v34`, mà
`14cd674` chỉ đụng `ode_physics_loss_shared`. `artifacts/fno` và `artifacts/fno2`
**không dùng lại được**: không có `model.pt`, nên không sinh được chỉ số chính.

Còn lại: **117 lần chạy** (17 ô × 7 seed − 2). Danh sách ở
`audit_port/LAUNCH_LIST.md`, sinh bởi `scripts/make_launch_list.py`.

**N-13. Bước CHẤM ĐIỂM đắt hơn kế hoạch 20 lần, và đó là lỗi cache chứ không
phải lỗi kế hoạch.** Đo được 2026-08-06: `24_rollout_thermal_error.py --exact`
mất **57 phút** cho 2 kịch bản × 730 cửa sổ trên CPU này. Gần như toàn bộ chi phí
là rollout **tham chiếu** và burn-in điểm cuối tuần hoàn — cả hai **không phụ
thuộc model**, chúng là tính chất của forcing — nhưng script tính lại từ đầu mỗi
lần gọi. Với 117 run: **~107 giờ CPU**, và ở chân trời 2 năm của Amendment 1 thì
gấp đôi.

Amendment 1 dự toán **4,6 giờ GPU** cho đúng việc này, với đúng lý do: "reference
rollout và cyclic burn-in không phụ thuộc model, nên tính một lần mỗi kịch bản và
dùng lại cho cả 112 cell-seed". Kế hoạch đúng, script chưa làm.

**Không chặn việc train** — train phải xảy ra trước dù thế nào. Việc cần làm
trước **bước chấm điểm**: lưu lại reference rollout và cyclic cache theo
`(K, max_windows)` rồi dùng lại giữa các checkpoint.