# COD benchmark

Đọc README.md trước khi làm gì.

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

## Bối cảnh

Bản thảo bị JCP desk-reject vì originality. Audit tìm ra 6 vấn đề blocking:
baseline chưa train, test set nằm trong training distribution, timing claims
không có artifact, PINN baseline không tồn tại. Đang xây lại toàn bộ phần
thực nghiệm với protocol đóng băng trước khi train.
