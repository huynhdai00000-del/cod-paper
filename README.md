# COD benchmark

Code cho manuscript Cascaded Operator Decomposition. Toàn bộ logic khoa học nằm trong package `cod/`. Notebook chỉ dùng để gọi, không chứa code.

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

Sửa code thì sửa trong `cod/`, commit, push. Lần chạy sau trên Colab tự có bản mới. Đổi tài khoản Google không mất gì vì bản gốc nằm trên GitHub.

## Cấu trúc

```
cod/
  config.py        đọc YAML, băm config, cơ chế đóng băng phân phối
  provenance.py    ghi commit hash, môi trường, GPU vào mỗi kết quả
  data/            sinh dữ liệu và ground truth
  models/          COD, PI-DeepONet, FNO, MIONet
  training/
    harness.py     vòng train dùng chung, tiêu chí hội tụ, phát hiện bệnh lý
  eval/
    metrics.py     đơn vị vật lý trước, NMAE kèm tỷ lệ chạm sàn
configs/           mỗi thí nghiệm một file YAML
scripts/run.py     điểm vào duy nhất
```

## Ba cơ chế xử lý nguyên nhân gốc từ audit

**Đóng băng phân phối.** `config.py` băm riêng khối `distribution`. Hash được ghi vào `DISTRIBUTION_FREEZE.md` trước khi train model đầu tiên. `assert_distribution_unchanged()` sẽ báo lỗi nếu ai nới một dải lấy mẫu. Điều này chặn đúng việc đã xảy ra ở v57, khi biên độ sinusoidal được nới với mục tiêu ghi rõ trong changelog là `target TV gap: 2.7% → <2.0%`.

**Truy vết.** Mỗi lần chạy ghi `run.json` chứa commit hash, config hash, seed, phiên bản thư viện, tên GPU. Câu hỏi "con số 0,6% từ lần chạy nào" trả lời được bằng cách tra file.

**Không hội tụ thì không được báo cáo thành số.** `harness.py` áp cùng một tiêu chí hội tụ cho mọi model, dùng ngân sách theo wall-clock song song với epoch, và trả về `converged=False` kèm `stop_reason` nếu model dừng vì hết ngân sách. Nó cũng phát hiện causal weight underflow, tức tình huống baseline Mono Fair có `wm=0.000` trong khi COD có `wm=0.988`, một bất đối xứng khiến so sánh mất giá trị.

## Metric

`eval/metrics.py` báo cáo MAE tuyệt đối theo đơn vị vật lý làm chỉ số chính, NMAE làm phụ, và luôn kèm tỷ lệ số case mà mẫu số chuẩn hóa chạm sàn. Với state gần hằng trong cửa sổ đánh giá, hàm `report()` in cảnh báo rõ rằng NMAE không phải thước đo sai số vật lý.

Ba tầng test không bao giờ được gộp: T1 trong phân phối, T2 ngoại suy tham số, T3 ngoài họ phân phối. `report()` bắt buộc khai báo tầng.

## Quy tắc

1. Không viết code khoa học trong notebook.
2. Commit trước mỗi lần chạy sẽ báo cáo số. `provenance.warn_if_dirty()` cảnh báo nếu quên.
3. Không sửa khối `distribution` sau khi đóng băng. Nếu buộc phải sửa, ghi vào `CHANGELOG_DISTRIBUTION.md` kèm ngày và lý do, và công bố trong bài.
4. Không nhìn kết quả tầng T2 và T3 cho tới khi model đã train xong theo tiêu chí định trước. Mọi điều chỉnh chỉ dựa trên validation tách từ phân phối huấn luyện.
5. Model không hội tụ được báo cáo là không hội tụ, kèm learning curve.
