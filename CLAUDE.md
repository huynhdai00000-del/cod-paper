# COD benchmark

Đọc README.md trước khi làm gì.
Đọc DECISIONS.md trước khi đề xuất bất kỳ hướng đi nào.
Không mở lại mục CLOSED. Phát hiện mâu thuẫn thì ghi vào mục "Bằng chứng mới".

## Chế độ làm việc

Làm việc tự chủ. Không dừng lại hỏi những quyết định thuộc thẩm quyền của bạn.

- Tài liệu tham chiếu nằm ở `reference/`. Đọc `reference/audit/AUDIT_REPORT.md`
  trước khi port. Source notebook đã trích sẵn ở `reference/audit/extracted/`.
- Gặp mơ hồ thì chọn phương án hợp lý nhất, ghi lại lựa chọn kèm lý do vào
  `PORT_LOG.md`, rồi đi tiếp. Không dừng để hỏi.
- Commit sau mỗi đơn vị công việc. Không commit trạng thái hỏng.
- Chạy test tự do, đó là cách tự kiểm chứng.

CHỈ dừng lại và báo cáo trong ba trường hợp:
1. Cổng xác minh Phase 1 thất bại (không tái hiện được Table 2).
2. Phát hiện điều mâu thuẫn với audit, tức audit có thể đã sai.
3. Cần thao tác nằm trong denylist.

## Quy tắc bắt buộc

- Không viết code khoa học trong notebook. Mọi logic nằm trong package `cod/`.
- Mỗi hàm định nghĩa đúng một lần. Repo cũ có `ode_physics_loss` ba bản khác nhau.
- Không sửa khối `distribution` trong config sau khi đã đóng băng hash.
- Model không hội tụ thì báo cáo là không hội tụ, không quy thành con số hiệu năng.
- `.detach()` trên thermal grid là hành vi cố ý của cascade, không phải bug.
- Thư mục `reference/` là chỉ đọc. Không sửa gì trong đó.

### Mọi phát biểu định lượng phải có script kiểm chứng

Không khẳng định một con số trong docstring, PORT_LOG hay báo cáo nếu chưa có
script đo nó. Viết script trước, chạy, rồi mới viết con số — và giữ script lại
trong `audit_port/scripts/` để chạy lại được.

Áp dụng cho: "hai cách này khớp nhau đến X", "sai số dưới Y", "nhanh hơn Z lần",
"không đổi", "tương đương". Nếu định viết một con số kèm đơn vị hoặc một phát
biểu về độ khớp, câu hỏi là: script nào in ra nó?

Lý do, từ một lần đã xảy ra (PORT_LOG J-70, fix 8): docstring của
`cyclic_endpoint_theta` viết rằng nghiệm giải tích và RK45 "khớp nhau đến dưới
0,01 °C" — viết trước khi đo. Đo thật thì lệch **0,11 đến 0,30 °C**, tăng theo
tải, vì recurrence dạng đóng đẩy theta_TO về một `theta_ss` cố định trong khi
`fac_n` của ODE thật lại phụ thuộc chính theta_TO qua `Rf`. Con số sai đó nằm
dưới một thước đo có nhiệm vụ phân giải sai số cỡ 0,5 °C, tức đúng loại lỗi mà
O-9 vừa gỡ, ở một tầng sâu hơn và khó thấy hơn. Viết `19_verify_bias_fix.py` là
thứ bắt được nó.

Hệ quả kèm theo: script kiểm chứng phải **fail được**. Đặt ngưỡng và trả exit
code khác 0, đừng chỉ in số ra rồi tự đọc.

## Quy ước hình vẽ

Mọi hình trong repo theo đúng bộ quy ước này. Cài đặt nằm ở `scripts/figures.py`
(`RC`, `FORMATS`); hình viết ở chỗ khác phải khớp.

- **Font serif.** `font.family: serif`, mathtext `dejavuserif`.
- **Xuất cả PDF và SVG**, cùng tên, vào thư mục con `figures/` của thư mục kết
  quả của lần chạy.
- **Không có suptitle.** Thông tin toàn hình đặt ở nhãn trục hoặc annotation góc.
- **Có title cho từng subplot.** Đây là chỗ ghi đại lượng và đơn vị.
- Lưới mờ (`alpha 0.3`), `bbox_inches="tight"`, dpi 150.

Code vẽ hình **không nằm trong package** `cod/`. Đây là quy tắc cũ đã ghi ở
`cod/eval/rollout.py`: rollout trả về dữ liệu, việc vẽ tách ra ngoài. Hình nào
cần một con số thì con số đó phải có sẵn trong `run.json`, `predictions.npz` hoặc
`clamp_history.json` trước — hàm vẽ không được tự tính.

**Cái gì không dựng lại được từ `predictions.npz`** thì phải lưu riêng, vì
`predictions.npz` chỉ chứa phần đánh giá (`pred`, `gt`, `t_eval`, `x0`, sensors,
`kind`, `family`). Toàn bộ phần *huấn luyện* không nằm trong đó:

- đường loss huấn luyện → `loss_history.json`
- đường validation, thứ mà tiêu chí plateau thực sự quyết định trên đó →
  `run.json` (`outcome.val_history`)
- quỹ đạo clamp và quỹ đạo causal weight, trả lời câu "nó kích hoạt lúc nào" mà
  không thống kê tóm tắt nào trả lời được (J-85, J-89) → `clamp_history.json`

Còn một chuỗi **chưa** lưu, ghi ra đây để nó là lỗ hổng đã biết chứ không phải
lỗ hổng im lặng: chuẩn gradient theo epoch (`PathologyReport` chỉ giữ
`grad_norm_final`).

## Bối cảnh

Bản thảo bị JCP desk-reject vì originality. Audit tìm ra 6 vấn đề blocking:
baseline chưa train, test set nằm trong training distribution, timing claims
không có artifact, PINN baseline không tồn tại. Đang xây lại toàn bộ phần
thực nghiệm với protocol đóng băng trước khi train.

## Lệnh shell

Tránh `find -exec`, nó luôn kích hoạt prompt xin phép và không thể auto-allow.
Dùng thay thế:
- Đếm dòng: `wc -l $(find . -name '*.py')` hoặc `wc -l cod/**/*.py`
- Lọc file: `find ... | xargs ...` chỉ khi thực sự cần, ưu tiên glob
Gộp nhiều thao tác vào một script Python rồi chạy một lần, thay vì nhiều lệnh shell nhỏ.

## Lệnh shell: dùng gì và tránh gì

TRÁNH những lệnh sau, chúng luôn kích hoạt prompt xin phép vì vừa đọc
vừa ghi được, và không thể auto-allow bằng bất kỳ cấu hình nào:
  sed, awk, perl, find -exec, xargs

DÙNG thay thế (đã nằm trong allowlist, chạy không hỏi):
  cat, head, tail, wc, ls, grep, python

Ví dụ tương đương:
  sed -n '/A/,/B/p' f.md        ->  grep -A 200 'A' f.md | grep -B 200 'B'
  sed -n '100,200p' f.md        ->  head -200 f.md | tail -100
  find . -exec wc -l {} \;      ->  wc -l cod/**/*.py

QUAN TRỌNG: với bất kỳ thao tác kiểm tra nhiều bước nào, viết một script
Python vào audit_port/scripts/ rồi chạy một lần bằng `python`, thay vì
chuỗi nhiều lệnh shell nhỏ. Đây là cách hiệu quả nhất để làm việc liên tục
không bị ngắt, và script còn lưu lại được để chạy lại.

Git identity đã cấu hình sẵn trong repo. Dùng `git commit` trực tiếp,
không cần `git -c user.name=... commit`.

KHÔNG dùng heredoc Python inline (`python - <<'EOF'`). Mỗi lần như vậy đều
kích hoạt prompt xin phép vì bộ phân tích tĩnh không đọc được f-string và
dấu chấm phẩy, và không thể auto-allow bằng bất kỳ cấu hình nào.

Thay vào đó: ghi file vào `audit_port/scripts/` rồi chạy bằng
`python audit_port/scripts/NN_ten.py`. Lệnh này đã nằm trong allowlist nên
chạy không hỏi, và script còn lưu lại được để chạy lại.

Cũng áp dụng cho việc sửa file: dùng công cụ Edit thay vì `cat >>` hay `sed`.