# COD benchmark

Code cho manuscript Cascaded Operator Decomposition. Toàn bộ logic khoa học nằm
trong package `cod/`. Notebook chỉ dùng để gọi, không chứa code.

## Quy trình

Trên máy local:

```bash
git clone https://github.com/<user>/cod-paper.git
cd cod-paper
pip install -e .
python scripts/run.py --config configs/example_cod_seed1.yaml
```

Trên Colab hoặc Kaggle, notebook chỉ cần bốn dòng:

```python
!git clone https://github.com/<user>/cod-paper.git
%cd cod-paper
!pip install -e . --quiet
!python scripts/run.py --config configs/example_cod_seed1.yaml --out /content/drive/MyDrive/cod_results
```

Sửa code thì sửa trong `cod/`, commit, push. Lần chạy sau trên Colab tự có bản
mới. Đổi tài khoản Google không mất gì vì bản gốc nằm trên GitHub.

## Cấu trúc

```
cod/
  config.py        đọc YAML, băm config, cơ chế đóng băng phân phối
  provenance.py    ghi commit hash, môi trường, GPU vào mỗi kết quả
  data/
    physics.py     ODE RHS (numpy + torch), hot-spot, Arrhenius, pd_factor, DP
    steady_state.py  formula_A/B/C cạnh nhau + true_fixed_point()
    profiles.py    sample_consistent_ic, make_sensor_profile
    generate.py    sinh dataset, test set seed 999, ground truth RK45
  models/
    blocks.py      ModifiedMLP, trunk features 32 chiều, interp_sensors
    cod.py         COD (giữ .detach() trên thermal grid — cố ý)
    monolithic.py  PI-DeepONet monolithic: Fair, MultiHead, SoftIC
  training/
    losses.py      ode_physics_loss (+ biến thể của shared trainer), CHI losses
    train.py       CODTrainer (train_v34), SharedPhysicsTrainer (train_physics)
    harness.py     vòng train dùng chung, tiêu chí hội tụ, phát hiện bệnh lý
  eval/
    metrics.py     đơn vị vật lý trước, NMAE kèm tỷ lệ chạm sàn
    benchmark.py   NMAE đúng như notebook — chỉ để tái hiện số cũ
    rollout.py     CHI/DP lifetime rollout (đã tách khỏi hàm vẽ hình)
configs/           mỗi thí nghiệm một file YAML
scripts/
  run.py               điểm vào duy nhất
  verify_phase1.py     ba cổng xác minh Phase 1
  measure_fix_effects.py  đo tác động từng fix của Phase 2
audit_port/scripts/  script kiểm chứng, chạy lại được
```

## Trạng thái

**Phase 1 — port trung thực: xong, cả ba cổng đạt.** Xem
`PHASE1_VERIFICATION.md`. Không train lại gì.

| cổng | nội dung | kết quả |
|---|---|---|
| 1 | Table 2 từ `transformer_pideepOnet_v57.pt` | đạt, lệch ≤ 0.05 pp mọi state |
| 2 | capacity sweep, 10 checkpoint | đạt |
| 3 | monolithic headline 13199.7% | đạt, khớp tuyệt đối |

Bằng chứng mạnh nhất: sinh lại 8000 IC với seed 42 tái hiện
`transformer_training_v57.npz` **đúng từng byte**, và test set seed 999 giống
bit-for-bit với bản tái hiện của chính audit (`06_test_ranges.npy`, sai khác
tương đối 0.000e+00). Cả 13 checkpoint load được với `strict=True`.

**Phase 2 — sáu fix, mỗi fix một commit: xong.** Xem `PHASE2_EFFECTS.md` để có
số Gate 1 trước/sau từng fix.

Fix 6 (2026-07-28) đóng DECISIONS N-1: `fast_rhs_torch` và `_gas_integral` bỏ
`V_arr.clamp(max=1e4)` và lấy đúng bao nhiệt độ `[313.15, 573.15]` K của
`fast_rhs_np`. Trước đó reference và model giải hai động học khác nhau — 996
trên 4000 trạng thái ngẫu nhiên lệch nhau, tối đa 100% của đạo hàm; sau fix là
0. Gate 1 overall 1,49% → 1,26%, `c_C2H2` MAE 0,5926 → 0,1138 ppm.

Mặc định của package bây giờ là hành vi ĐÃ SỬA. Hành vi v57 vẫn gọi được bằng
tham số tường minh, nên `scripts/verify_phase1.py` vẫn tái hiện đủ ba cổng — đó
là regression test.

**Checkpoint hiện tại không còn hợp lệ.** Fix 1, 4 và 5 đều đổi training
distribution, nên phải lập lại hash đóng băng trong `DISTRIBUTION_FREEZE.md`
trước lần retrain đầu tiên. Fix 6 vô hiệu checkpoint vì lý do khác: nó đổi
physics residual, tức mục tiêu huấn luyện, mà không đổi phân phối lấy mẫu.

## Ba cơ chế xử lý nguyên nhân gốc từ audit

**Đóng băng phân phối.** `config.py` băm riêng khối `distribution`. Hash được
ghi vào `DISTRIBUTION_FREEZE.md` trước khi train model đầu tiên.
`assert_distribution_unchanged()` sẽ báo lỗi nếu ai nới một dải lấy mẫu. Điều
này chặn đúng việc đã xảy ra ở v57, khi biên độ sinusoidal được nới với mục tiêu
ghi rõ trong changelog là `target TV gap: 2.7% → <2.0%`.

**Truy vết.** Mỗi lần chạy ghi `run.json` chứa commit hash, config hash, seed,
phiên bản thư viện, tên GPU. Câu hỏi "con số 0,6% từ lần chạy nào" trả lời được
bằng cách tra file.

**Không hội tụ thì không được báo cáo thành số.** `harness.py` áp cùng một tiêu
chí hội tụ cho mọi model, dùng ngân sách theo wall-clock song song với epoch, và
trả về `converged=False` kèm `stop_reason` nếu model dừng vì hết ngân sách. Nó
cũng phát hiện causal weight underflow. Cơ chế này đã bắt được lỗi thật ngay
trong lúc kiểm thử: `SharedPhysicsTrainer` trên model monolithic trả về
`causal_weight_min = 0.0` ngay bước đầu tiên, tái hiện đúng `wm=0.000` mà audit
B-1 mô tả. Sau fix 3 giá trị này là 1e-8, không còn bằng 0.

## Metric

`eval/metrics.py` báo cáo MAE tuyệt đối theo đơn vị vật lý làm chỉ số chính,
NMAE làm phụ, và luôn kèm tỷ lệ số case mà mẫu số chuẩn hóa chạm sàn. Với state
gần hằng trong cửa sổ đánh giá, hàm `report()` in cảnh báo rõ rằng NMAE không
phải thước đo sai số vật lý.

`eval/benchmark.py` là NMAE đúng như notebook cũ, sàn mẫu số 1e-4, chỉ dùng để
tái hiện số cũ. Đo trực tiếp cho thấy sai số 34,558% trên acetylene của
monolithic thực chất là 0.705 ppm, tức 2.0% ngưỡng IEC 60599; sai số lớn thật là
nhiệt, 13.4 °C.

Ba tầng test không bao giờ được gộp: T1 trong phân phối, T2 ngoại suy tham số,
T3 ngoài họ phân phối. `report()` bắt buộc khai báo tầng. Lưu ý: test set hiện
tại **chỉ có T1**. Chưa có thí nghiệm ngoài phân phối thật sự nào trong repo.

## Quy tắc

1. Không viết code khoa học trong notebook.
2. Commit trước mỗi lần chạy sẽ báo cáo số. `provenance.warn_if_dirty()` cảnh
   báo nếu quên.
3. Không sửa khối `distribution` sau khi đóng băng. Nếu buộc phải sửa, ghi vào
   `CHANGELOG_DISTRIBUTION.md` kèm ngày và lý do, và công bố trong bài.
4. Không nhìn kết quả tầng T2 và T3 cho tới khi model đã train xong theo tiêu
   chí định trước. Mọi điều chỉnh chỉ dựa trên validation tách từ phân phối
   huấn luyện.
5. Model không hội tụ được báo cáo là không hội tụ, kèm learning curve.
6. `.detach()` trên thermal grid trong `cod.py` là hành vi cố ý của cascade.
   Không xóa. Nếu §6/§7/Appendix C mô tả gas residual là tín hiệu huấn luyện thì
   phần chữ sai, không phải code sai.
