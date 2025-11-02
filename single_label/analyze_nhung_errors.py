"""
Phân tích các errors có từ "nhưng" (adversative conjunction)
Tìm patterns và đề xuất giải pháp
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import os

def analyze_nhung_errors():
    # Change to BERT directory
    os.chdir('D:/BERT')
    
    print("="*80)
    print("PHAN TICH ERRORS CO TU 'NHUNG' (Adversative Conjunction)")
    print("="*80)
    
    # Load all errors1
    errors_file = "error_analysis_results/all_errors_detailed.csv"
    
    if not os.path.exists(errors_file):
        print(f"File không tồn tại: {errors_file}")
        print("Vui lòng chạy: python tests/error_analysis.py")
        return
    
    df = pd.read_csv(errors_file, encoding='utf-8-sig')
    
    print(f"\nTổng số errors: {len(df)}")
    
    # Find errors with "nhưng"
    nhung_errors = df[df['sentence'].str.contains('nhưng', case=False, na=False)]
    
    print(f"Số errors có từ 'nhưng': {len(nhung_errors)} ({len(nhung_errors)/len(df)*100:.1f}%)")
    
    if len(nhung_errors) == 0:
        print("\nKhông có errors nào chứa từ 'nhưng'")
        return
    
    # Analyze confusion patterns
    print(f"\n{'='*80}")
    print("CONFUSION PATTERNS CHO ERRORS CO 'NHUNG':")
    print(f"{'='*80}")
    
    confusion_stats = nhung_errors.groupby('confusion_type').size().reset_index(name='count')
    confusion_stats = confusion_stats.sort_values('count', ascending=False)
    
    for _, row in confusion_stats.iterrows():
        pct = row['count'] / len(nhung_errors) * 100
        print(f"   • {row['confusion_type']:<30} {row['count']:>3} cases ({pct:>5.1f}%)")
    
    # Analyze by aspect
    print(f"\n{'='*80}")
    print("ERRORS CO 'NHUNG' BY ASPECT:")
    print(f"{'='*80}")
    
    aspect_stats = nhung_errors.groupby('aspect').size().reset_index(name='count')
    aspect_stats = aspect_stats.sort_values('count', ascending=False)
    
    for _, row in aspect_stats.iterrows():
        pct = row['count'] / len(nhung_errors) * 100
        print(f"   • {row['aspect']:<20} {row['count']:>3} cases ({pct:>5.1f}%)")
    
    # Show examples
    print(f"\n{'='*80}")
    print("TOP 10 ERRORS CO 'NHUNG' (Examples):")
    print(f"{'='*80}\n")
    
    for i, row in nhung_errors.head(10).iterrows():
        # Highlight "nhưng" in sentence
        sentence = row['sentence']
        # Find position of "nhưng"
        import re
        nhung_match = re.search(r'nhưng', sentence, re.IGNORECASE)
        if nhung_match:
            pos = nhung_match.start()
            # Show context around "nhưng"
            start = max(0, pos - 40)
            end = min(len(sentence), pos + 60)
            context = sentence[start:end]
            if start > 0:
                context = "..." + context
            if end < len(sentence):
                context = context + "..."
        else:
            context = sentence[:100] + "..." if len(sentence) > 100 else sentence
        
        print(f"{i+1}. Aspect: {row['aspect']}")
        print(f"   True: {row['sentiment']:>8} → Predicted: {row['predicted_sentiment']:<8}")
        print(f"   Text: {context}")
        print()
    
    # Save nhung errors to separate file
    output_file = "error_analysis_results/nhung_errors_detailed.csv"
    nhung_errors.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved {len(nhung_errors)} errors có 'nhưng' to: {output_file}")
    
    # Also check all data (not just errors)
    print(f"\n{'='*80}")
    print("PHAN TICH TOAN BO DATASET (bao gồm cả đúng + sai):")
    print(f"{'='*80}")
    
    # Load test set
    test_file = "data/test.csv"
    pred_file = "test_predictions.csv"
    
    if os.path.exists(test_file) and os.path.exists(pred_file):
        test_df = pd.read_csv(test_file, encoding='utf-8-sig')
        pred_df = pd.read_csv(pred_file, encoding='utf-8-sig')
        
        # Merge
        full_df = test_df.copy()
        full_df['predicted_sentiment'] = pred_df['predicted_sentiment']
        full_df['correct'] = full_df['sentiment'] == full_df['predicted_sentiment']
        
        # Find all "nhưng" sentences
        nhung_all = full_df[full_df['sentence'].str.contains('nhưng', case=False, na=False)]
        
        print(f"\nTổng số samples có 'nhưng' trong test set: {len(nhung_all)}")
        print(f"Số samples đúng: {nhung_all['correct'].sum()} ({nhung_all['correct'].sum()/len(nhung_all)*100:.1f}%)")
        print(f"Số samples sai: {(~nhung_all['correct']).sum()} ({(~nhung_all['correct']).sum()/len(nhung_all)*100:.1f}%)")
        
        print(f"\nSo sánh với overall performance:")
        overall_acc = full_df['correct'].mean()
        nhung_acc = nhung_all['correct'].mean()
        print(f"   • Overall accuracy: {overall_acc:.2%}")
        print(f"   • Accuracy on 'nhưng' sentences: {nhung_acc:.2%}")
        print(f"   • Difference: {(nhung_acc - overall_acc)*100:+.2f}%")
        
        if nhung_acc < overall_acc:
            print(f"\nModel performs WORSE on sentences with 'nhưng'!")
        else:
            print(f"\nModel performs OK on sentences with 'nhưng'")
    
    # Generate solutions
    print(f"\n{'='*80}")
    print("GIAI PHAP DE XUAT:")
    print(f"{'='*80}\n")
    
    solutions = [
        "1. DATA AUGMENTATION với từ chuyển ý:",
        "   • Tạo thêm samples có 'nhưng', 'tuy nhiên', 'mặc dù', 'song'",
        "   • Oversampling các samples có từ chuyển ý bị sai",
        "   • Synthetic data: đảo ngược câu có 'nhưng' để tạo thêm data",
        "",
        "2. FEATURE ENGINEERING:",
        "   • Thêm special token [ADV] trước 'nhưng' khi tokenize",
        "   • Ví dụ: 'pin tốt nhưng camera xấu' → 'pin tốt [ADV] nhưng camera xấu'",
        "   • Model sẽ học được rằng [ADV] là signal quan trọng",
        "",
        "3. ATTENTION MECHANISM:",
        "   • Fine-tune thêm epochs với focus vào adversative conjunctions",
        "   • Tăng weight cho tokens xung quanh 'nhưng' trong loss function",
        "",
        "4. RULE-BASED POST-PROCESSING:",
        "   • Detect câu có 'nhưng' → split thành 2 phần",
        "   • Phần SAU 'nhưng' thường quan trọng hơn",
        "   • Ví dụ: 'Pin tốt nhưng camera tệ' → Focus vào 'camera tệ'",
        "",
        "5. ENSEMBLE METHOD:",
        "   • Train model riêng cho sentences có adversative conjunctions",
        "   • Combine predictions với main model",
        "",
        "6. CONTEXT WINDOW:",
        "   • Tăng max_length để model thấy được full context",
        "   • Hiện tại: 256 tokens → có thể tăng lên 384",
        "",
        "7. HARD NEGATIVE MINING:",
        "   • Tập trung train lại trên những samples có 'nhưng' bị sai",
        "   • Tăng weight của những samples này trong training",
        "",
        "8. PREPROCESSING:",
        "   • Tách câu phức thành câu đơn tại vị trí 'nhưng'",
        "   • Mỗi phần được phân tích riêng",
        "   • Ví dụ: 'Pin tốt nhưng camera tệ' →",
        "     - 'Pin tốt' (positive)",
        "     - 'camera tệ' (negative)",
    ]
    
    for solution in solutions:
        print(solution)
    
    print(f"\n{'='*80}")
    print("HANH DONG DE XUAT NGAY:")
    print(f"{'='*80}\n")
    
    print("OPTION 1: DATA AUGMENTATION (Dễ nhất, hiệu quả cao)")
    print("   → Oversampling các errors có 'nhưng' trong training data")
    print("   → Tạo script để duplicate và augment những samples này")
    print()
    print("OPTION 2: SPECIAL TOKEN (Cần retrain)")
    print("   → Thêm [ADV] token vào tokenizer")
    print("   → Retrain model với special token này")
    print()
    print("OPTION 3: RULE-BASED (Không cần retrain)")
    print("   → Post-processing: split câu tại 'nhưng'")
    print("   → Phân tích mỗi phần riêng, ưu tiên phần SAU 'nhưng'")
    print()
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    analyze_nhung_errors()
