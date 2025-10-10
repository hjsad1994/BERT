"""
Script Test Sentiment SMART với Aspect Relevance Detection
==========================================================
Chỉ hiển thị các aspect THỰC SỰ được đề cập trong câu
Sử dụng keyword matching + confidence filtering

Usage:
    python test_sentiment_smart.py
    python test_sentiment_smart.py --sentence "pin tệ quá"
    python test_sentiment_smart.py --sentence "pin tệ quá" --show-ignored
    python test_sentiment_smart.py --batch test_examples.txt
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from colorama import init, Fore, Style

init(autoreset=True)

# Configuration
MODEL_PATH = "finetuned_visobert_absa_model"
MAX_LENGTH = 256

# Aspect keywords (từ khóa tiếng Việt cho mỗi aspect)
ASPECT_KEYWORDS = {
    'Battery': [
        'pin', 'sạc', 'xạc', 'dung lượng pin', 'mah', 'battery', 'trâu pin',
        'hết pin', 'tụt pin', 'chai pin', 'phình pin', 'tốn pin', 'ngốn pin'
    ],
    'Camera': [
        'camera', 'cam', 'chụp', 'quay', 'ảnh', 'hình', 'selfie', 'tele',
        'zoom', 'góc rộng', 'chân dung', 'macro', 'ống kính', 'megapixel', 'mp'
    ],
    'Performance': [
        'hiệu năng', 'mượt', 'lag', 'giật', 'đơ', 'nhanh', 'chậm', 'mạnh',
        'yếu', 'chip', 'cpu', 'gpu', 'ram', 'snapdragon', 'mediatek', 'dimensity',
        'xử lý', 'đa nhiệm', 'gaming', 'game', 'fps'
    ],
    'Display': [
        'màn hình', 'màn', 'screen', 'display', 'hiển thị', 'độ phân giải',
        'amoled', 'lcd', 'oled', 'ips', 'tấm nền', 'hz', 'refresh rate',
        'sáng', 'tối', 'nắng', 'xem phim', 'sắc nét', 'mờ', 'trầy màn'
    ],
    'Design': [
        'thiết kế', 'đẹp', 'xấu', 'sang', 'bóng', 'nhám', 'vuông', 'tròn',
        'kim loại', 'nhựa', 'kính', 'vỏ', 'khung', 'cầm', 'nắm', 'ergonomic',
        'màu', 'color', 'style', 'kiểu dáng', 'ngoại hình', 'bề ngoài'
    ],
    'Software': [
        'phần mềm', 'software', 'hệ điều hành', 'os', 'android', 'ios',
        'miui', 'one ui', 'coloros', 'funtouch', 'realme ui', 'oxygen',
        'update', 'cập nhật', 'app', 'ứng dụng', 'bloatware', 'quảng cáo'
    ],
    'Packaging': [
        'đóng gói', 'hộp', 'box', 'bao bì', 'packaging', 'gói', 'niêm phong',
        'seal', 'tem', 'fullbox', 'nguyên seal'
    ],
    'Price': [
        'giá', 'price', 'tiền', 'rẻ', 'đắt', 'mắc', 'phải chăng', 'hợp lý',
        'cost', 'budget', 'túi tiền', 'khoản', 'vnđ', 'triệu', 'nghìn',
        'giá trị', 'value', 'đáng giá'
    ],
    'Audio': [
        'âm thanh', 'audio', 'loa', 'speaker', 'tiếng', 'nghe', 'nghe nhạc',
        'âm', 'bass', 'treble', 'stereo', 'mono', 'volume', 'âm lượng',
        'to', 'nhỏ', 'rè', 'rít', 'dolby', 'hifi', 'chất lượng âm thanh'
    ],
    'Warranty': [
        'bảo hành', 'warranty', 'bh', 'đổi trả', 'guarantee', 'hỗ trợ',
        'claim', 'sửa chữa', 'service center', 'trung tâm bảo hành'
    ],
    'Shop_Service': [
        'shop', 'cửa hàng', 'store', 'seller', 'người bán', 'chủ shop',
        'tư vấn', 'hỗ trợ', 'nhiệt tình', 'thái độ', 'phục vụ', 'service',
        'chăm sóc', 'trả lời', 'chat'
    ],
    'Shipping': [
        'giao hàng', 'ship', 'vận chuyển', 'delivery', 'shipper', 'giao',
        'nhận hàng', 'ship nhanh', 'ship chậm', 'giao nhanh', 'giao chậm',
        'shipper', 'đóng gói giao hàng'
    ],
    'General': [
        'máy', 'điện thoại', 'phone', 'smartphone', 'device', 'sản phẩm',
        'product', 'hàng', 'overall', 'tổng thể', 'chung', 'nói chung',
        'tốt', 'xấu', 'ok', 'oke', 'ổn', 'được'
    ],
    'Others': [
        'khác', 'other', 'phụ kiện', 'accessory', 'tai nghe', 'sạc dự phòng'
    ]
}

ID2LABEL = {0: 'positive', 1: 'negative', 2: 'neutral'}
LABEL2ID = {'positive': 0, 'negative': 1, 'neutral': 2}


def check_aspect_relevance(sentence, aspect):
    """
    Kiểm tra xem aspect có được đề cập trong câu không
    
    Returns:
        float: relevance score (0.0 - 1.0)
    """
    sentence_lower = sentence.lower()
    keywords = ASPECT_KEYWORDS.get(aspect, [])
    
    if not keywords:
        return 0.0
    
    # Đếm số keywords xuất hiện
    matches = 0
    for keyword in keywords:
        if keyword.lower() in sentence_lower:
            matches += 1
    
    # Relevance score = số keywords match / tổng số keywords
    # Bonus nếu có ít nhất 1 match
    if matches > 0:
        base_score = matches / len(keywords)
        # Boost score nếu có match (ít nhất 0.3)
        relevance_score = max(0.3, min(1.0, base_score * 5))
        return relevance_score
    
    return 0.0


class SmartSentimentPredictor:
    """Predictor thông minh với aspect relevance detection"""
    
    def __init__(self, model_path=MODEL_PATH):
        """Khởi tạo predictor"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Đang load mô hình từ: {model_path}")
        print(f"🔧 Device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Load mô hình thành công!\n")
        except Exception as e:
            print(f"❌ Lỗi khi load mô hình: {str(e)}")
            sys.exit(1)
    
    def predict_single(self, sentence, aspect):
        """Dự đoán sentiment cho một aspect"""
        # Xử lý VNCoreNLP segmentation nếu có
        sentence = sentence.replace('_', ' ')
        
        inputs = self.tokenizer(
            sentence, aspect,
            return_tensors='pt',
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        predicted_idx = np.argmax(probs)
        predicted_sentiment = ID2LABEL[predicted_idx]
        confidence = probs[predicted_idx]
        
        probabilities = {
            ID2LABEL[i]: float(probs[i]) for i in range(len(probs))
        }
        
        return {
            'aspect': aspect,
            'sentiment': predicted_sentiment,
            'confidence': float(confidence),
            'probabilities': probabilities
        }
    
    def predict_smart(self, sentence, confidence_threshold=0.7, relevance_threshold=0.3):
        """
        Dự đoán THÔNG MINH - chỉ aspects được đề cập
        
        Args:
            sentence: Câu cần phân tích
            confidence_threshold: Ngưỡng confidence tối thiểu
            relevance_threshold: Ngưỡng relevance tối thiểu (aspect được đề cập)
            
        Returns:
            dict: {
                'relevant_aspects': list (aspects được đề cập với sentiment),
                'all_results': list (tất cả aspects),
                'ignored_aspects': list (aspects bị ignore vì không relevance)
            }
        """
        all_results = []
        relevant_aspects = []
        ignored_aspects = []
        
        print(f"\n{Fore.CYAN}🔍 Đang phân tích aspect relevance...{Style.RESET_ALL}")
        
        for aspect in ASPECT_KEYWORDS.keys():
            # Check relevance
            relevance_score = check_aspect_relevance(sentence, aspect)
            
            # Predict sentiment
            result = self.predict_single(sentence, aspect)
            result['relevance_score'] = relevance_score
            
            all_results.append(result)
            
            # Chỉ thêm vào relevant nếu:
            # 1. Có relevance score > threshold
            # 2. Confidence > threshold
            # 3. Không phải neutral (hoặc neutral với confidence thấp)
            if relevance_score >= relevance_threshold:
                if result['confidence'] >= confidence_threshold:
                    relevant_aspects.append(result)
                elif result['sentiment'] != 'neutral':
                    # Aspect được đề cập nhưng confidence thấp - vẫn thêm vào
                    relevant_aspects.append(result)
            else:
                ignored_aspects.append({
                    'aspect': aspect,
                    'reason': f'Not mentioned (relevance: {relevance_score:.2f})'
                })
        
        # Sort by relevance score
        relevant_aspects = sorted(relevant_aspects, 
                                 key=lambda x: (x['relevance_score'], x['confidence']), 
                                 reverse=True)
        
        return {
            'relevant_aspects': relevant_aspects,
            'all_results': all_results,
            'ignored_aspects': ignored_aspects
        }


def format_sentiment_emoji(sentiment):
    """Emoji cho sentiment"""
    return {'positive': '😊', 'negative': '😞', 'neutral': '😐'}.get(sentiment, '')


def format_confidence_bar(confidence, width=20):
    """Progress bar"""
    filled = int(confidence * width)
    return '█' * filled + '░' * (width - filled)


def print_smart_results(sentence, results, show_ignored=False):
    """In kết quả thông minh"""
    print(f"\n{'='*80}")
    print(f"📝 Câu phân tích: {Fore.CYAN}{sentence}{Style.RESET_ALL}")
    print(f"{'='*80}\n")
    
    relevant = results['relevant_aspects']
    ignored = results['ignored_aspects']
    
    if relevant:
        print(f"{Fore.GREEN}🎯 CÁC ASPECT ĐƯỢC ĐỀ CẬP:{Style.RESET_ALL}\n")
        
        for idx, result in enumerate(relevant, 1):
            aspect = result['aspect']
            sentiment = result['sentiment']
            confidence = result['confidence']
            relevance = result['relevance_score']
            emoji = format_sentiment_emoji(sentiment)
            
            # Màu theo sentiment
            color = {
                'positive': Fore.GREEN,
                'negative': Fore.RED,
                'neutral': Fore.YELLOW
            }.get(sentiment, '')
            
            print(f"{idx}. {Fore.CYAN}{aspect:<15}{Style.RESET_ALL} → "
                  f"{emoji} {color}{sentiment.upper():<10}{Style.RESET_ALL}")
            print(f"   Confidence: {confidence:.1%} | Relevance: {relevance:.1%}")
            
            # Progress bars
            conf_bar = format_confidence_bar(confidence)
            rel_bar = format_confidence_bar(relevance)
            print(f"   Conf: {conf_bar} {confidence:.2%}")
            print(f"   Relv: {rel_bar} {relevance:.2%}\n")
    else:
        print(f"{Fore.YELLOW}⚠️  Không phát hiện aspect cụ thể nào được đề cập{Style.RESET_ALL}\n")
    
    # Hiển thị aspects bị ignore
    if show_ignored and ignored:
        print(f"\n{Fore.BLUE}🚫 ASPECTS BỊ BỎ QUA (không được đề cập):{Style.RESET_ALL}")
        for item in ignored[:5]:  # Chỉ hiển thị 5 cái đầu
            print(f"   • {item['aspect']}: {item['reason']}")
        if len(ignored) > 5:
            print(f"   ... và {len(ignored) - 5} aspects khác")


def print_compact_results(sentence, results):
    """In kết quả dạng compact cho batch mode"""
    relevant = results['relevant_aspects']
    
    print(f"\n📝 \"{sentence}\"")
    
    if relevant:
        print(f"→ Phát hiện {len(relevant)} aspect(s) được đề cập:")
        for r in relevant:
            emoji = format_sentiment_emoji(r['sentiment'])
            print(f"   • {r['aspect']}: {emoji} {r['sentiment']} "
                  f"(conf: {r['confidence']:.1%}, rel: {r['relevance_score']:.1%})")
    else:
        print(f"→ Không phát hiện aspect cụ thể nào")
    print()


def batch_mode(predictor, sentences):
    """Chế độ batch - test nhiều câu từ file"""
    print(f"\n{'='*80}")
    print(f"{Fore.GREEN}📦 CHẾ ĐỘ BATCH - Test {len(sentences)} câu{Style.RESET_ALL}")
    print(f"{'='*80}\n")
    
    for idx, sentence in enumerate(sentences, 1):
        print(f"{Fore.BLUE}[{idx}/{len(sentences)}]{Style.RESET_ALL}", end=" ")
        
        # Predict
        results = predictor.predict_smart(sentence)
        
        # Print compact
        print_compact_results(sentence, results)


def interactive_mode(predictor):
    """Chế độ tương tác"""
    print(f"\n{'='*80}")
    print(f"{Fore.GREEN}🔮 CHẾ ĐỘ TƯƠNG TÁC THÔNG MINH - SMART ABSA{Style.RESET_ALL}")
    print(f"{'='*80}")
    print(f"\nTính năng:")
    print(f"  • Tự động phát hiện aspects được đề cập trong câu")
    print(f"  • Lọc bỏ aspects không liên quan")
    print(f"  • Hiển thị relevance score cho mỗi aspect")
    print(f"\nHướng dẫn:")
    print(f"  • Nhập câu để phân tích")
    print(f"  • Gõ 'ignored' sau câu để xem aspects bị bỏ qua")
    print(f"  • Gõ 'quit' để thoát")
    print(f"\n{'-'*80}\n")
    
    while True:
        try:
            user_input = input(f"{Fore.YELLOW}Nhập câu:{Style.RESET_ALL} ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Fore.CYAN}👋 Tạm biệt!{Style.RESET_ALL}\n")
                break
            
            # Check show ignored
            show_ignored = False
            if user_input.lower().endswith(' ignored'):
                show_ignored = True
                user_input = user_input[:-8].strip()
            
            # Predict
            print(f"\n{Fore.CYAN}⏳ Đang phân tích...{Style.RESET_ALL}")
            results = predictor.predict_smart(user_input)
            
            # Print results
            print_smart_results(user_input, results, show_ignored=show_ignored)
            
            print(f"\n{'-'*80}\n")
            
        except KeyboardInterrupt:
            print(f"\n\n{Fore.CYAN}👋 Tạm biệt!{Style.RESET_ALL}\n")
            break
        except Exception as e:
            print(f"\n{Fore.RED}❌ Lỗi: {str(e)}{Style.RESET_ALL}\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Test sentiment THÔNG MINH với aspect relevance detection'
    )
    parser.add_argument('--sentence', '-s', type=str, help='Câu cần test')
    parser.add_argument('--batch', '-b', type=str, 
                       help='File chứa danh sách câu (mỗi dòng một câu)')
    parser.add_argument('--show-ignored', action='store_true', 
                       help='Hiển thị aspects bị ignore')
    
    args = parser.parse_args()
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n{Fore.RED}❌ Không tìm thấy mô hình tại: {MODEL_PATH}{Style.RESET_ALL}\n")
        return
    
    # Initialize predictor
    predictor = SmartSentimentPredictor(MODEL_PATH)
    
    # Batch mode
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"\n{Fore.RED}❌ Không tìm thấy file: {args.batch}{Style.RESET_ALL}\n")
            return
        
        with open(args.batch, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        batch_mode(predictor, sentences)
        return
    
    # Test sentence
    if args.sentence:
        results = predictor.predict_smart(args.sentence)
        print_smart_results(args.sentence, results, show_ignored=args.show_ignored)
        return
    
    # Interactive mode
    interactive_mode(predictor)


if __name__ == '__main__':
    main()
