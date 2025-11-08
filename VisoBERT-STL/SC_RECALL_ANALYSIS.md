# Phân Tích Lỗi Recall - Sentiment Classification (SC)

## Tổng Quan

Recall tổng thể: **91.04%**

### Recall theo Aspect (từ thấp đến cao):

1. **Price**: 80.70% ⚠️ (thấp nhất)
2. **Design**: 88.16%
3. **Performance**: 89.83%
4. **General**: 91.46%
5. **Camera**: 93.55%
6. **Shop_Service**: 94.16%
7. **Display**: 95.71%
8. **Shipping**: 96.49%
9. **Packaging**: 96.59%
10. **Battery**: 98.12% ✅ (cao nhất)

---

## Phân Tích Chi Tiết Các Aspect Có Recall Thấp

### 1. PRICE (Recall: 80.70%) - VẤN ĐỀ NGHIÊM TRỌNG NHẤT

**Tổng số samples có label thật**: 57
- True Positive (TP): 46
- False Negative (FN): 11
- **Recall: 80.70%**

#### Phân tích lỗi:

1. **Predict Neutral nhưng True là Positive** (54.5% lỗi)
   - 6/11 samples
   - Ví dụ:
     - Sample 467: "Mua xong thì giá lại giảm hơn rất nhiều tiếc ơi là tiếc luôn..."
     - Sample 562: "Hàng nguyên seal. Mới khui dùng thấy ok. Ngon. không mua được giá sale. Tiếc :(("
   - **Vấn đề**: Model không nhận diện được các câu tích cực về giá (ví dụ: giá tốt, giá sale, giá rẻ)

2. **Predict Neutral nhưng True là Negative** (27.3% lỗi)
   - 3/11 samples
   - Ví dụ:
     - Sample 138: "Nhóm màu: đen sản phẩm đúng như trình bày dong gói kỹ giao hàng nhiệt tình chất lượng máy hoạt động..."
     - Sample 809: "Giá bán của sản phẩm tương xứng với cấu hình được trang bị..."
   - **Vấn đề**: Model không nhận diện được các câu tiêu cực về giá

3. **Predict Positive nhưng True là Negative** (18.2% lỗi)
   - 2/11 samples
   - Ví dụ:
     - Sample 1116: "áp xu và voucher luôn thì săn được giá 1260k sử dụng được mấy ngày thì bị sọc ngang dọc rồi tắt nguồn..."
     - Sample 1337: "máy đã qua sử dụng, màn hình đã thay, không còn cảm ứng vân tay. dẫu biết rằng tiền nào của nấy..."
   - **Vấn đề**: Model nhầm lẫn giữa positive và negative về giá

#### Nguyên nhân:
- Model quá conservative, thường predict Neutral thay vì Positive/Negative
- Các từ khóa về giá (giá, giá bán, giá sale, tiền) có thể không được model học tốt
- Context về giá có thể bị nhầm với các aspect khác

---

### 2. PERFORMANCE (Recall: 89.83%)

**Tổng số samples có label thật**: 118
- True Positive (TP): 106
- False Negative (FN): 12
- **Recall: 89.83%**

#### Phân tích lỗi:

1. **Predict Positive nhưng True là Negative** (41.7% lỗi)
   - 5/12 samples
   - Ví dụ:
     - Sample 79: "Camera selfie tệ. Ảnh selfie khác ảnh quảng cáo. Camera thiếu chi tiết + màu giả. Selfie bị mịn quá..."
     - Sample 297: "chất lượng sản phẩm chất lượng kém so với giá. đúng với mô tả thiếu 1 số phụ kiện..."
   - **Vấn đề**: Model nhầm lẫn giữa positive và negative về performance
   - **Nguyên nhân có thể**: Câu có nhiều aspect, model bị confuse

2. **Predict Neutral nhưng True là Negative** (33.3% lỗi)
   - 4/12 samples
   - Ví dụ:
     - Sample 133: "Sản phẩm thiết kế đẹp, mạnh mẽ với giá cả phải chăng, tuy nhiên hệ điều hành hơi ít tùy biến..."
     - Sample 392: "Điện thoại thiết kế màn cong, cầm trên tay cho cảm giác rất mỏng nhẹ chắc chắn. Chưa dùng nhiều nên..."
   - **Vấn đề**: Model không nhận diện được negative sentiment về performance

3. **Predict Neutral nhưng True là Positive** (25.0% lỗi)
   - 3/12 samples
   - Ví dụ:
     - Sample 380: "Sau 1 thời gian trải nghiệm thì mình nhận thấy máy bắt wifi không tốt lắm..."
     - Sample 583: "Hiệu năng nhanh và mượt mà., độ phân giải hd cho hình ảnh sắc nét..."
   - **Vấn đề**: Model không nhận diện được positive sentiment về performance

#### Nguyên nhân:
- Performance là aspect phức tạp, có thể liên quan đến nhiều thành phần (CPU, RAM, wifi, hiệu năng)
- Model có thể bị confuse khi câu có nhiều aspect cùng lúc
- Các từ khóa về performance có thể không rõ ràng

---

### 3. DESIGN (Recall: 88.16%)

**Tổng số samples có label thật**: 76
- True Positive (TP): 67
- False Negative (FN): 9
- **Recall: 88.16%**

#### Phân tích lỗi:

1. **Predict Neutral nhưng True là Positive** (77.8% lỗi)
   - 7/9 samples
   - Ví dụ:
     - Sample 174: "cửa hàng đóng gói rất cẩn thận, tem niêm phong, hoá đơn đầy đủ. có điều máy cực kì ám vàng. không bi..."
     - Sample 190: "Đó h xài ip, nhỏ em gái mua khen dv bảo hành bên Oppo xịn nên đua theoooo. Màu vàng đồng chuẩn thì s..."
   - **Vấn đề**: Model không nhận diện được positive sentiment về design
   - **Nguyên nhân**: Các từ về màu sắc, thiết kế có thể không được model học tốt

2. **Predict Neutral nhưng True là Negative** (22.2% lỗi)
   - 2/9 samples
   - Ví dụ:
     - Sample 785: "điện thoại ngon, chơi game không lag, ngoại hình như mới có điều khay sim hơi cũ..."
     - Sample 1348: "Pin 80, máy sử dụng tốt nhận sim nhưng không bật được internet ngoại hình ổn..."
   - **Vấn đề**: Model không nhận diện được negative sentiment về design

#### Nguyên nhân:
- Design thường được đề cập gián tiếp (màu sắc, ngoại hình, thiết kế)
- Model có thể không liên kết được các từ khóa về design với sentiment

---

## Pattern Chung Của Lỗi

### 1. **Predict Neutral thay vì Positive/Negative** (Vấn đề chính)
- Chiếm phần lớn lỗi recall
- Model quá conservative, không tự tin predict positive/negative
- **Giải pháp**: 
  - Giảm threshold cho positive/negative classification
  - Tăng weight cho positive/negative samples trong training
  - Sử dụng focal loss với alpha cao hơn cho positive/negative classes

### 2. **Nhầm lẫn giữa Positive và Negative**
- Đặc biệt với Performance và Price
- **Giải pháp**:
  - Tăng số lượng training samples cho các cặp positive/negative
  - Sử dụng contrastive learning
  - Data augmentation cho các câu positive/negative

### 3. **Aspect-specific issues**
- **Price**: Model không nhận diện tốt các từ khóa về giá
- **Performance**: Model bị confuse khi có nhiều aspect trong cùng câu
- **Design**: Model không liên kết được các từ khóa về design với sentiment

---

## Khuyến Nghị

### Ngắn hạn:
1. **Giảm threshold** cho positive/negative classification
2. **Tăng weight** cho positive/negative classes trong loss function
3. **Data augmentation** cho Price và Performance aspects

### Dài hạn:
1. **Fine-tune** model với focus vào Price và Performance
2. **Collect more data** cho Price và Performance với negative samples
3. **Feature engineering**: Thêm các từ khóa đặc trưng cho từng aspect
4. **Ensemble methods**: Kết hợp nhiều model cho các aspect khó

---

## Kết Luận

Recall thấp chủ yếu do:
1. Model quá conservative → predict Neutral thay vì Positive/Negative
2. Nhầm lẫn giữa Positive và Negative (đặc biệt Performance)
3. Không nhận diện tốt các từ khóa đặc trưng cho từng aspect (đặc biệt Price)

**Aspect cần cải thiện ưu tiên**: Price (80.70%) > Performance (89.83%) > Design (88.16%)

