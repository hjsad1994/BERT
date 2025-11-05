# Backend API cho Dual-Task ABSA Model

## Cài đặt

Cài đặt các dependencies:
```bash
pip install -r requirements.txt
```

## Cách sử dụng

### 1. Chạy FastAPI Backend

**Cách 1: Sử dụng script helper (Khuyên dùng)**
```bash
# Từ thư mục gốc dự án
python dual-task-learning/start_api.py
```

**Cách 2: Chạy trực tiếp**
```bash
# Từ thư mục dual-task-learning
cd dual-task-learning
python api.py
```

**Cách 3: Sử dụng uvicorn**
```bash
cd dual-task-learning
uvicorn api:app --host 0.0.0.0 --port 8000
```

API sẽ chạy tại: `http://localhost:8000`

- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`
- Model Info: `http://localhost:8000/model/info`

### 2. Chạy Gradio Demo

**Cách 1: Sử dụng script helper (Khuyên dùng)**
```bash
# Từ thư mục gốc dự án
python dual-task-learning/start_demo.py
```

**Cách 2: Chạy trực tiếp**
```bash
# Từ thư mục dual-task-learning
cd dual-task-learning
python demo_gradio.py
```

Demo sẽ chạy tại: `http://localhost:7860`

## Dual-Task Learning Approach

Model này sử dụng **Dual-Task Learning**:
- **Task 1: Aspect Detection** - Binary classification (aspect có mặt hay không)
- **Task 2: Sentiment Classification** - 3-class classification (positive/negative/neutral) cho các aspects được phát hiện

## API Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Predict Single Text
```bash
POST /predict
Content-Type: application/json

{
    "text": "Pin trâu camera xấu",
    "min_aspect_confidence": 0.5,
    "filter_absent": true,
    "min_sentiment_confidence": 0.5,
    "top_k": 3
}
```

Response:
```json
{
    "text": "Pin trâu camera xấu",
    "predictions": {
        "Battery": {
            "present": true,
            "present_confidence": 0.95,
            "sentiment": "positive",
            "sentiment_confidence": 0.88,
            "probabilities": {
                "positive": 0.88,
                "negative": 0.05,
                "neutral": 0.07
            }
        },
        "Camera": {
            "present": true,
            "present_confidence": 0.92,
            "sentiment": "negative",
            "sentiment_confidence": 0.91,
            "probabilities": {
                "positive": 0.03,
                "negative": 0.91,
                "neutral": 0.06
            }
        }
    }
}
```

### 3. Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
    "texts": [
        "Pin trâu camera xấu",
        "Màn hình đẹp giá rẻ"
    ],
    "filter_absent": true,
    "top_k": 3
}
```

### 4. Get Aspects
```bash
GET /aspects
```

### 5. Get Sentiments
```bash
GET /sentiments
```

### 6. Get Model Info
```bash
GET /model/info
```

## Ví dụ sử dụng với Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "Pin trâu camera xấu",
        "filter_absent": True,
        "top_k": 3
    }
)
result = response.json()
print(result)

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "texts": [
            "Pin trâu camera xấu",
            "Màn hình đẹp giá rẻ"
        ],
        "filter_absent": True,
        "top_k": 3
    }
)
results = response.json()
print(results)
```

## Ví dụ sử dụng với JavaScript/Frontend

```javascript
// Single prediction
fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        text: 'Pin trâu camera xấu',
        filter_absent: true,
        top_k: 3
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Filtering Parameters

- `min_aspect_confidence` (0.0-1.0): Ngưỡng confidence tối thiểu để aspect được coi là "present"
- `filter_absent` (bool): Chỉ trả về các aspects được phát hiện (present=true)
- `min_sentiment_confidence` (0.0-1.0): Ngưỡng confidence tối thiểu cho sentiment prediction
- `top_k` (int, optional): Chỉ trả về top K aspects có aspect confidence cao nhất

## Cấu hình

Model được load từ:
- Config: `dual-task-learning/config_multi.yaml`
- Model checkpoint: `dual-task-learning/models/dual_task_learning/best_model.pt`

Có thể thay đổi bằng cách chỉnh sửa trong `model_service.py` hoặc truyền tham số khi khởi tạo.

## So sánh với Multi-Label Approach

| Feature | Multi-Label | Dual-Task |
|---------|-------------|-----------|
| Aspect Detection | ❌ (predict tất cả) | ✅ (binary detection) |
| Sentiment Classification | ✅ (softmax per aspect) | ✅ (softmax per aspect) |
| Filtering | Dựa trên confidence | Dựa trên aspect detection |
| Output | Sentiment cho tất cả aspects | Chỉ sentiment cho aspects được detect |

