"""
Script Ví Dụ Dự Đoán ABSA
=========================
Demo cách sử dụng mô hình đã fine-tune để dự đoán sentiment cho aspect trong câu mới

Usage:
    python predict_example.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def predict_sentiment(model, tokenizer, sentence, aspect, device='cpu'):
    """
    Dự đoán sentiment cho một cặp (sentence, aspect)
    
    Args:
        model: Mô hình đã fine-tune
        tokenizer: Tokenizer
        sentence: Câu văn
        aspect: Khía cạnh cần phân tích
        device: 'cuda' hoặc 'cpu'
        
    Returns:
        tuple: (predicted_sentiment, confidence_score)
    """
    # Label mapping
    id2label = {
        0: 'positive',
        1: 'negative',
        2: 'neutral'
    }
    
    # Xử lý VNCoreNLP segmentation nếu có
    sentence = sentence.replace('_', ' ')
    
    # Tokenize input
    inputs = tokenizer(
        sentence,
        aspect,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding=True
    )
    
    # Di chuyển inputs sang device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Dự đoán
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()
    
    predicted_sentiment = id2label[predicted_class]
    
    return predicted_sentiment, confidence


def main():
    """Hàm main"""
    print("\n" + "="*70)
    print("DỰ ĐOÁN ABSA VỚI VISOBERT")
    print("="*70)
    
    # Đường dẫn đến mô hình đã fine-tune
    model_path = "finetuned_visobert_absa_model"
    
    # Kiểm tra device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load mô hình và tokenizer
    print(f"\nĐang load mô hình từ: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        print(f"Load mô hình thành công!")
    except Exception as e:
        print(f"\nLỗi: {str(e)}")
        print(f"\nGợi ý: Hãy chạy 'python train.py' để fine-tune mô hình trước")
        return
    
    # Các ví dụ test
    test_examples = [
        {
            'sentence': 'Pin trâu lắm, dùng cả ngày không lo hết pin. Camera chụp hơi tối.',
            'aspects': ['Battery', 'Camera']
        },
        {
            'sentence': 'Màn hình đẹp, hiển thị sắc nét. Hiệu năng mượt mà, chơi game không lag.',
            'aspects': ['Display', 'Performance']
        },
        {
            'sentence': 'Giá hơi cao so với mặt bằng chung. Thiết kế đẹp, sang trọng.',
            'aspects': ['Price', 'Design']
        },
        {
            'sentence': 'Giao hàng nhanh, đóng gói cẩn thận. Shop tư vấn nhiệt tình.',
            'aspects': ['Shipping', 'Packaging', 'Shop_Service']
        }
    ]
    
    print(f"\n{'='*70}")
    print("KẾT QUẢ DỰ ĐOÁN")
    print(f"{'='*70}\n")
    
    # Dự đoán cho từng ví dụ
    for idx, example in enumerate(test_examples, 1):
        sentence = example['sentence']
        aspects = example['aspects']
        
        print(f"Ví dụ {idx}:")
        print(f"Câu: {sentence}\n")
        
        for aspect in aspects:
            sentiment, confidence = predict_sentiment(
                model, tokenizer, sentence, aspect, device
            )
            
            print(f"  • {aspect:>15}: {sentiment:>10} (confidence: {confidence:.2%})")
        
        print()
    
    # Interactive mode
    print(f"{'='*70}")
    print("CHẾ ĐỘ TƯƠNG TÁC")
    print(f"{'='*70}\n")
    print("Nhập câu và aspect để dự đoán (hoặc 'quit' để thoát)\n")
    
    while True:
        try:
            sentence = input("Câu: ").strip()
            if sentence.lower() in ['quit', 'exit', 'q']:
                break
            
            if not sentence:
                continue
            
            aspect = input("Aspect: ").strip()
            if not aspect:
                continue
            
            sentiment, confidence = predict_sentiment(
                model, tokenizer, sentence, aspect, device
            )
            
            print(f"\n→ Kết quả: {sentiment.upper()} (confidence: {confidence:.2%})\n")
            print("-" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nThoát chương trình.")
            break
        except Exception as e:
            print(f"\nLỗi: {str(e)}\n")
    
    print("\n" + "="*70)
    print("Cảm ơn bạn đã sử dụng!")
    print("="*70 + "\n")


if __name__ == '__main__':
    # Fix encoding cho Windows
    import sys
    import io
    
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    main()
