# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** Không được ghi trong notebook/output đã cung cấp.

**Runtime Colab:** CPU.

**Python / PyTorch / Ultralytics:** Python `3.13.18` / PyTorch `2.11.0+cpu` / Ultralytics `8.4.145`.

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`.

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`.

**Thay đổi so với notebook nguồn:** Không ghi nhận thay đổi đối với checkpoint hoặc ngưỡng dùng để tạo ba tệp JSON. Notebook có chạy thêm phép so sánh detection tại các ngưỡng `0.20`, `0.35`, `0.60`; phép so sánh này không thay đổi ngưỡng `0.35` được lưu trong JSON.

> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.

**Ảnh dùng:** `sample_id = traffic`, `coco_image_id = 210273`, kích thước `640 × 428` pixel. Lấy dòng `rank = 1`.

1. Record hạng 1: `class_id` = **468**; `class_name` = **cab**; `rank` = **1**; `score` = **0.510915**; `taxonomy_name` = **ImageNet-1K**.

2. Record này nói về **cả bức ảnh**, không phải một vật thể riêng. Đây là tác vụ image classification: model xếp hạng các lớp đại diện cho nội dung toàn ảnh. JSON classification không có tọa độ `bbox` hoặc polygon để xác định vị trí từng vật thể.

3. Danh sách tên lớp không phải do model tự nghĩ ra. Các tên lớp thuộc taxonomy đi kèm checkpoint; ở đây là `ImageNet-1K` của `yolo11n-cls.pt`.

4. Cần ghi cả `class_id`, `class_name` và `taxonomy_name`: `class_id` là mã lớp, `class_name` là tên dễ đọc, còn `taxonomy_name` xác định bộ danh mục mà mã/tên đó thuộc về. Chỉ ghi tên lớp sẽ khó đối chiếu chính xác giữa các taxonomy hoặc checkpoint khác nhau.

5. Khi ảnh có nhiều vật hoặc nhiều chủ đề, guideline cần quy định tiêu chí chọn **một lớp đại diện cho toàn ảnh** (ví dụ chủ thể chính theo mục tiêu bộ dữ liệu, mức độ nổi bật hoặc diện tích), đồng thời nêu cách xử lý ảnh có nhiều chủ đề ngang nhau hoặc không có chủ đề rõ ràng. Annotator phải áp dụng cùng một tiêu chí.

6. Không được lấy `score` làm ground truth vì score chỉ thể hiện mức độ mô hình tin vào dự đoán, không chứng minh dự đoán đúng. Với ảnh này, lớp top-1 `cab` có score `0.510915`; ground truth vẫn phải do annotator xác định theo guideline và được reviewer kiểm tra.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.

**Ảnh dùng theo output mới:** `sample_id = traffic`, `coco_image_id = 210273`, kích thước `640 × 428` pixel. Record được cung cấp có `score_threshold = 0.35`, `taxonomy_name = COCO-80`.

1. Record `person` trong output mới:
   - `class_id` = **0**
   - `class_name` = **person**
   - `score` = **0.769676**
   - `bbox_xyxy` = **[0.08, 256.79, 18.39, 313.12]**
   - `bbox_width` = **18.32**
   - `bbox_height` = **56.33**
   - `coordinate_unit` = **pixel**
   - `bbox_format` = **xyxy**

2. Theo định dạng `xyxy`, góc trên-trái của box ở tọa độ **(0.08, 256.79)** pixel tính từ góc trên-trái ảnh; góc dưới-phải ở **(18.39, 313.12)**. Box rộng **18.32 pixel** và cao **56.33 pixel**. Box nằm sát mép trái ảnh vì tọa độ x-min là `0.08` pixel.

3. Record mới xác nhận đây là prediction của ảnh `traffic` tại ngưỡng `0.35`. Không dùng record `person` của ảnh `kitchen` trong phiên bản report này. Số lượng prediction ở các ngưỡng `0.20` và `0.60` cho ảnh `traffic` **chưa thể xác nhận từ record JSON đơn lẻ được cung cấp**. Các số `17/11/6` trong bản report trước được ghi cho ảnh `kitchen`, nên không được chuyển sang ảnh `traffic`.

4. Khi hạ ngưỡng, nhiều hộp có score thấp hơn có thể được giữ lại; reviewer cần kiểm tra false positive, lớp dự đoán sai, box không ôm đúng vật thể và box trùng lặp. Số prediction nhiều hơn không đồng nghĩa chất lượng nhãn cao hơn. Cần dùng kết quả chạy threshold đúng trên ảnh `traffic` để định lượng sự thay đổi.

5. Quy tắc box: **Vẽ hình chữ nhật nhỏ nhất có thể nhưng vẫn bao trọn phần nhìn thấy của đúng một vật thể; hộp song song với trục ảnh, không tự ước lượng phần bị che khuất và không lấy thêm nền không cần thiết.** Áp dụng thống nhất cho tất cả ảnh.

6. Prediction `person` có box sát mép trái ảnh (`x_min = 0.08`). Đây là tình huống cần chú ý khi xây dựng guideline về vật thể bị cắt bởi mép khung hình. Nếu guideline chưa quy định cách xử lý hoặc mức độ nhìn thấy tối thiểu để gán lớp, cần **escalation lên Lab Coach** để thống nhất quy tắc; không tự đặt ngưỡng.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.

**Ảnh dùng:** `sample_id = kitchen`, `coco_image_id = 397133`, kích thước `640 × 427` pixel. Chọn record `kitchen-002`.

1. Record:
   - `instance_id` = **kitchen-002**
   - `class_name` = **bowl**
   - `score` = **0.735743**
   - `polygon_point_count` = **67**
   - 5 điểm đầu của `polygon_xy` = **[[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0], [48.0, 346.0]]**

2. Box chỉ cho biết hình chữ nhật bao quanh vật thể. Polygon mô tả đường biên theo hình dạng thực tế của vật thể, gồm các đoạn cong/lồi lõm và vùng không chiếm hết hình chữ nhật; do đó polygon/mask cung cấp thông tin hình dạng chi tiết hơn box.

3. `instance_id` là mã phân biệt từng đối tượng riêng trong output. Ví dụ `kitchen-002` là một instance cụ thể thuộc lớp `bowl`; một bowl khác có thể có `instance_id` khác dù cùng `class_name`. Nó không phải class ID (mã chung của lớp) và cũng không phải tracking ID (mã theo dõi cùng một vật thể qua nhiều frame/video).

4. Quy tắc vẽ biên mask: **Vẽ theo đường ranh giới nhìn thấy của đúng một instance, bám sát biên vật thể, không lấn sang nền hoặc vật thể bên cạnh; với phần bị che khuất, chỉ xử lý theo quy tắc che khuất đã được guideline quy định, không tự đoán đường biên không nhìn thấy.**

5. Trên ảnh segmentation `kitchen`, mask `dining table` bao phủ vùng mặt bàn lớn và nằm sát các mask `bowl` trên bàn; ranh giới giữa mặt bàn và vật đặt trên bàn là vị trí cần chú ý khi xác định pixel thuộc instance nào. Ngoài ra, mask `potted plant` và `oven` nằm sát mép trái ảnh. Với ranh giới nhìn thấy rõ, annotator áp dụng guideline; tại vùng hai vật chạm nhau hoặc bị che mà không thể xác định chắc đường biên, cần escalation lên Lab Coach nếu guideline chưa có quy tắc phân xử.

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một class label cho toàn ảnh, kèm class ID/name theo taxonomy đã chọn. | `traffic` trong `classification_predictions.json`: top-1 `cab`, class ID `468`, score `0.510915`, taxonomy `ImageNet-1K`; top-2 `minibus`, score `0.164284`. Đây là dự đoán cấp ảnh, không phải nhãn chuẩn. | Xem toàn ảnh, chọn lớp theo tiêu chí chủ đề chính trong guideline; ghi đúng taxonomy và class; escalation nếu ảnh có nhiều chủ đề ngang nhau hoặc không rõ lớp đại diện. | Kiểm tra nhãn có mô tả toàn ảnh và đúng taxonomy không. Nếu chọn lớp không phù hợp guideline, trả lại để annotator xem xét/sửa; không chọn lớp chỉ vì model có score cao. |
| Phát hiện vật thể | Với mỗi vật thể: class label và bounding box `xyxy` theo pixel. | Output Detection mới của `traffic`: `person`, class ID `0`, score `0.769676`, box `[0.08, 256.79, 18.39, 313.12]`; box sát mép trái ảnh 640 × 428. Record này là prediction của model tại threshold `0.35`. | Gán từng vật thể nhìn thấy theo guideline; vẽ box cho đúng một instance; xử lý vật thể cắt mép theo quy tắc đã được xác nhận, không tự đặt ngưỡng nhìn thấy. | Kiểm tra lớp, box có bao đúng phần nhìn thấy không, box có quá rộng/hẹp hoặc trùng lặp không, có vật thể bị bỏ sót không. Với box sát mép trái, xác minh cách áp dụng quy tắc truncated object; nếu guideline chưa rõ thì escalation, không tự kết luận. |
| Instance segmentation | Với mỗi instance: class label, mask/polygon theo pixel và `instance_id` riêng. | `segmentation_predictions.json`: record `kitchen-002` là `bowl`, score `0.735743`, polygon có 67 điểm; `kitchen-005` là `dining table` với 598 điểm polygon. PNG segmentation cho thấy vùng bàn và các vật đặt trên bàn nằm gần nhau; các mask sát mép ảnh cũng cần được quan sát cẩn thận. | Vẽ mask riêng cho từng vật thể, bám biên nhìn thấy; giữ ID riêng cho từng instance; không gộp hai vật chỉ vì cùng lớp; escalation khi biên bị che hoặc không thể phân định. | Kiểm tra mask có lấn nền/vật thể khác không, biên có bám đúng vật thể không, instance có bị gộp/tách sai không và ID có phân biệt đúng instance không. Trả lại mask lệch biên để sửa; ca mơ hồ chuyển Lab Coach quyết định và cập nhật guideline. |

## 5. An toàn dữ liệu

1. Chỉ sử dụng ảnh và dữ liệu công khai đúng phạm vi bài lab; không đưa thông tin cá nhân hoặc dữ liệu nhạy cảm vào notebook, JSON, ảnh, báo cáo hay repository.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng xử lý và báo **Lab Coach/Giảng viên phụ trách**, không tự ý tiếp tục hoặc chia sẻ dữ liệu đó.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã đưa các tệp vào `report/` trên GitHub và kiểm tra repository
- [ ] Đã nộp **link repository** trên LMS
