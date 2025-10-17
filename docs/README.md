# Fine-tuning ViSoBERT cho ABSA (Aspect-Based Sentiment Analysis)

Dự án này fine-tune mô hình `5CD-AI/Vietnamese-Sentiment-visobert` từ Hugging Face cho nhiệm vụ Phân tích Cảm xúc theo Khía cạnh (ABSA) trên dữ liệu tiếng Việt.

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu hình](#cấu-hình)
- [Kết quả](#kết-quả)

## 🎯 Tổng quan

Dự án này giải quyết bài toán ABSA, trong đó mô hình dự đoán cảm xúc (positive, negative, neutral) của một khía cạnh cụ thể trong câu văn tiếng Việt.

**Ví dụ:**
- Câu: "Pin trâu nhưng camera hơi tệ"
- Aspect: "Battery" → Sentiment: **positive**
- Aspect: "Camera" → Sentiment: **negative**

## 📁 Cấu trúc dự án

```
D:\BERT/
├── config.yaml                         # Cấu hình trung tâm
├── train.py                            # Script huấn luyện chính
├── utils.py                            # Module tiện ích
├── prepare_data.py                     # Script chuẩn bị dữ liệu
├── README.md                           # File này
├── requirements.txt                    # Dependencies
├── dataset.csv                         # Dataset gốc (multi-label)
├── finetuned_visobert_absa_model/      # Mô hình đã fine-tune (tự động tạo)
├── evaluation_report.txt               # Báo cáo đánh giá (tự động tạo)
├── test_predictions.csv                # Kết quả dự đoán (tự động tạo)
└── data/
    ├── train.csv                       # Dữ liệu train (single-label)
    ├── validation.csv                  # Dữ liệu validation
    ├── test.csv                        # Dữ liệu test
    └── data_metadata.json              # Metadata về dữ liệu
```

## 🔧 Cài đặt

### 1. Yêu cầu hệ thống

- Python 3.8+
- GPU với CUDA (khuyến nghị) hoặc CPU
- RAM: 8GB+ (16GB+ nếu dùng GPU)

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Dependencies chính:**
- `transformers` - Thư viện Hugging Face
- `torch` - PyTorch
- `pandas` - Xử lý dữ liệu
- `numpy` - Tính toán số học
- `scikit-learn` - Metrics đánh giá
- `pyyaml` - Đọc file cấu hình

## 🚀 Sử dụng

### Bước 1: Chuẩn bị dữ liệu

Chuyển đổi dataset từ format multi-label sang single-label ABSA:

```bash
python prepare_data.py
```

**Output:**
- `data/train.csv` - 5399 mẫu (70%)
- `data/validation.csv` - 1157 mẫu (15%)
- `data/test.csv` - 1157 mẫu (15%)

**Format dữ liệu output:**
```csv
sentence,aspect,sentiment
"Pin trâu nhưng camera hơi tệ",Battery,positive
"Pin trâu nhưng camera hơi tệ",Camera,negative
```

### Bước 2: Fine-tune mô hình

```bash
python train.py --config config.yaml
```

**Quá trình huấn luyện sẽ:**
1. Tự động phát hiện GPU/CPU
2. Load mô hình ViSoBERT từ Hugging Face
3. Fine-tune trên dữ liệu ABSA
4. Đánh giá trên tập validation mỗi epoch
5. Lưu best model
6. Đánh giá chi tiết trên tập test
7. Tạo báo cáo và file predictions

### Bước 3: Kiểm tra kết quả

**Báo cáo đánh giá:** `evaluation_report.txt`
```
Accuracy:  0.8523
Precision: 0.8467
Recall:    0.8523
F1 Score:  0.8489
```

**Predictions:** `test_predictions.csv`
```csv
sentence,aspect,true_sentiment,predicted_sentiment
"Pin trâu",Battery,positive,positive
```

## ⚙️ Cấu hình

File `config.yaml` chứa tất cả các tham số:

### Đường dẫn
```yaml
paths:
  data_dir: "data"
  output_dir: "finetuned_visobert_absa_model"
  evaluation_report: "evaluation_report.txt"
  predictions_file: "test_predictions.csv"
```

### Mô hình
```yaml
model:
  name: "5CD-AI/Vietnamese-Sentiment-visobert"
  num_labels: 3  # positive, negative, neutral
  max_length: 256
```

### Huấn luyện
```yaml
training:
  learning_rate: 2.0e-5
  num_train_epochs: 3
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32
  warmup_steps: 500
  fp16: false  # Đặt true nếu GPU hỗ trợ
```

**Điều chỉnh batch size nếu gặp lỗi OOM (Out of Memory):**
- GPU 8GB: `per_device_train_batch_size: 8`
- GPU 6GB: `per_device_train_batch_size: 4`
- CPU: `per_device_train_batch_size: 4`

## 📊 Kết quả

### Thống kê dữ liệu

**Dataset gốc:**
- 4,021 câu review
- 14 aspects: Battery, Camera, Performance, Display, Design, Software, Packaging, Price, Audio, Warranty, Shop_Service, Shipping, General, Others

**Dataset ABSA:**
- 7,713 mẫu (trung bình 1.92 aspects/câu)
- Train: 5,399 mẫu (70%)
- Validation: 1,157 mẫu (15%)
- Test: 1,157 mẫu (15%)

**Phân bố sentiment:**
- Negative: 57.6%
- Positive: 36.7%
- Neutral: 5.7%

### Hiệu suất mô hình

Mô hình sau khi fine-tune được lưu tại `finetuned_visobert_absa_model/` và có thể được load lại:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('finetuned_visobert_absa_model')
model = AutoModelForSequenceClassification.from_pretrained('finetuned_visobert_absa_model')

# Dự đoán
sentence = "Pin trâu nhưng camera hơi tệ"
aspect = "Battery"
inputs = tokenizer(sentence, aspect, return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()  # 0: positive, 1: negative, 2: neutral
```

### Test với Script Interactive

**⭐ KHUYẾN NGHỊ: Dùng `test_sentiment_smart.py`** - Chỉ hiển thị aspects THỰC SỰ được đề cập trong câu!

**Script test_sentiment_smart.py** - Phiên bản THÔNG MINH với aspect relevance detection:
```bash
python test_sentiment_smart.py
```
- ✅ Tự động phát hiện aspects được đề cập
- ✅ Lọc bỏ aspects không liên quan
- ✅ Hiển thị relevance score
- ✅ Kết quả chính xác hơn

**Ví dụ:**
```bash
# Test một câu
python test_sentiment_smart.py --sentence "Pin trâu lắm"

# Test nhiều câu từ file
python test_sentiment_smart.py --batch test_examples.txt

# Xem aspects bị lọc bỏ
python test_sentiment_smart.py --sentence "Pin trâu lắm" --show-ignored
```
Output: Chỉ hiển thị **Battery** (positive), không hiển thị Camera, Price... như script cũ

---

**Script test_sentiment.py** cung cấp nhiều chế độ test (hiển thị TẤT CẢ aspects):

**1. Chế độ tương tác (Interactive Mode):**
```bash
python test_sentiment.py
```
- Nhập câu để phân tích
- Hiển thị tất cả aspects có confidence > 70%
- Gõ `all` sau câu để xem tất cả 14 aspects
- Gõ `examples` để xem ví dụ
- Gõ `quit` để thoát

**2. Test một câu cụ thể:**
```bash
python test_sentiment.py --sentence "pin tệ quá"
```

**3. Test một aspect cụ thể với xác suất chi tiết:**
```bash
python test_sentiment.py --sentence "pin tệ quá" --aspect Battery
```
Output:
```
→ Kết quả: 😞 NEGATIVE
→ Confidence: 99.90%

Xác suất chi tiết:
  😞 negative  : ███████████████████████████████ 99.90%
  😊 positive  : ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.06%
  😐 neutral   : ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.03%
```

**4. Test nhiều câu cùng lúc (Batch Mode):**
```bash
python test_sentiment.py --batch test_examples.txt
```

**5. Hiển thị tất cả aspects:**
```bash
python test_sentiment.py --sentence "máy đẹp" --all
```

Hoặc sử dụng script demo đơn giản:
```bash
python predict_example.py
```

## 🔍 Chi tiết kỹ thuật

### Input Format

Mô hình BERT nhận input theo format:
```
[CLS] sentence [SEP] aspect [SEP]
```

Ví dụ:
```
[CLS] Pin trâu nhưng camera hơi tệ [SEP] Battery [SEP]
```

### Label Encoding

```python
sentiment_labels = {
    'positive': 0,
    'negative': 1,
    'neutral': 2
}
```

### Metrics

- **Accuracy**: Tỷ lệ dự đoán đúng
- **Precision**: Độ chính xác (weighted average)
- **Recall**: Độ phủ (weighted average)
- **F1 Score**: Trung bình điều hòa của Precision và Recall

## 🛠️ Troubleshooting

### Lỗi CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
- Giảm `per_device_train_batch_size` trong `config.yaml`
- Tăng `gradient_accumulation_steps`
- Giảm `max_length`

### Lỗi Encoding trên Windows
```
UnicodeEncodeError: 'charmap' codec can't encode character
```

**Giải pháp:** Đã được xử lý tự động trong code. Nếu vẫn gặp lỗi, chạy:
```bash
chcp 65001  # Đặt console sang UTF-8
python train.py
```

### Mô hình không tải được
```
OSError: Can't load config for '5CD-AI/Vietnamese-Sentiment-visobert'
```

**Giải pháp:**
- Kiểm tra kết nối internet
- Kiểm tra tên mô hình trong `config.yaml`
- Thử tải thủ công: `transformers-cli download 5CD-AI/Vietnamese-Sentiment-visobert`

## 📝 License

Dự án này sử dụng mô hình `5CD-AI/Vietnamese-Sentiment-visobert`. Vui lòng tham khảo license của mô hình gốc.

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request.

## 📧 Liên hệ

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue trên GitHub.

---

**Chúc bạn fine-tune thành công! 🎉**
