# Backend API cho Multi-Label ABSA Model

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
python multi_label/start_api.py
```

**Cách 2: Chạy trực tiếp**
```bash
# Từ thư mục multi_label
cd multi_label
python api.py
```

**Cách 3: Sử dụng uvicorn**
```bash
cd multi_label
uvicorn api:app --host 0.0.0.0 --port 8000
```

API sẽ chạy tại: `http://localhost:8000`

- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 2. Chạy Gradio Demo

**Cách 1: Sử dụng script helper (Khuyên dùng)**
```bash
# Từ thư mục gốc dự án
python multi_label/start_demo.py
```

**Cách 2: Chạy trực tiếp**
```bash
# Từ thư mục multi_label
cd multi_label
python demo_gradio.py
```

Demo sẽ chạy tại: `http://localhost:7860`

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
    "text": "Pin trâu camera xấu"
}
```

Response:
```json
{
    "text": "Pin trâu camera xấu",
    "predictions": {
        "Battery": {
            "sentiment": "positive",
            "confidence": 0.95,
            "probabilities": {
                "positive": 0.95,
                "negative": 0.03,
                "neutral": 0.02
            }
        },
        "Camera": {
            "sentiment": "negative",
            "confidence": 0.88,
            "probabilities": {
                "positive": 0.05,
                "negative": 0.88,
                "neutral": 0.07
            }
        },
        ...
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
    ]
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

## Ví dụ sử dụng với Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Pin trâu camera xấu"}
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
        ]
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
        text: 'Pin trâu camera xấu'
    })
})
.then(response => response.json())
.then(data => console.log(data));

// Batch prediction
fetch('http://localhost:8000/predict/batch', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        texts: [
            'Pin trâu camera xấu',
            'Màn hình đẹp giá rẻ'
        ]
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Cấu hình

Model được load từ:
- Config: `multi_label/config_multi.yaml`
- Model checkpoint: `multi_label/models/multilabel_focal/best_model.pt`

Có thể thay đổi bằng cách chỉnh sửa trong `model_service.py` hoặc truyền tham số khi khởi tạo.

