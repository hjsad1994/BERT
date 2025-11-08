"""
Phan tich loi recall cho Sentiment Classification (SC)
Tim hieu tai sao recall thap va predict sai o dau
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import sys
from sklearn.metrics import precision_recall_fscore_support

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def analyze_recall_errors():
    """Phân tích các lỗi recall trong SC"""
    
    # Đường dẫn files (relative to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'VisoBERT-STL' else script_dir
    
    predictions_file = os.path.join(script_dir, "models/sentiment_classification/test_predictions_detailed.csv")
    test_file = os.path.join(script_dir, "data/test_multilabel.csv")
    output_file = os.path.join(script_dir, "models/sentiment_classification/recall_errors_all_samples.txt")
    
    if not os.path.exists(predictions_file):
        print(f"ERROR: Khong tim thay file predictions: {predictions_file}")
        return
    
    if not os.path.exists(test_file):
        print(f"ERROR: Khong tim thay file test data: {test_file}")
        return
    
    # Load data
    print("Dang load predictions va test data...")
    pred_df = pd.read_csv(predictions_file, encoding='utf-8-sig')
    test_df = pd.read_csv(test_file, encoding='utf-8-sig')
    
    # Đảm bảo có cột 'data' để hiển thị text
    if 'data' not in pred_df.columns and 'data' in test_df.columns:
        pred_df['data'] = test_df['data'].values
    
    aspects = ['Battery', 'Camera', 'Performance', 'Display', 'Design', 
               'Packaging', 'Price', 'Shop_Service', 'Shipping', 'General']
    
    # Tạo labeled_mask giống như code training (chỉ tính các samples có label trong CSV gốc)
    print("Tao labeled_mask tu CSV goc...")
    labeled_mask = {}
    for aspect in aspects:
        if aspect in test_df.columns:
            # Chỉ tính các samples có label (không phải NaN) trong CSV gốc
            labeled_mask[aspect] = test_df[aspect].notna().values
        else:
            labeled_mask[aspect] = np.zeros(len(test_df), dtype=bool)
    
    # Mapping: 0 = Positive, 1 = Negative, 2 = Neutral (theo dataset_visobert_sc.py)
    label_map = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
    
    # List để lưu tất cả các samples bị lỗi
    all_errors = []
    
    print("\n" + "="*80)
    print("PHAN TICH LOI RECALL - SENTIMENT CLASSIFICATION")
    print("="*80)
    
    # Phân tích từng aspect
    for aspect in aspects:
        pred_col = f"{aspect}_pred"
        true_col = f"{aspect}_true"
        correct_col = f"{aspect}_correct"
        
        if pred_col not in pred_df.columns or true_col not in pred_df.columns:
            continue
        
        # Loc cac samples co label that (theo CSV goc - giong code training)
        # Chỉ tính các samples có label trong CSV gốc (không phải NaN)
        if aspect not in labeled_mask:
            continue
        
        has_label = labeled_mask[aspect]
        
        if not has_label.any():
            continue
        
        # Tính recall: dùng macro recall giống như code training
        # Macro recall = average của recall cho từng class (Positive, Negative, Neutral)
        
        # Lọc các samples có label trong CSV gốc
        aspect_preds = pred_df.loc[has_label, pred_col].values
        aspect_labels = pred_df.loc[has_label, true_col].values
        
        # Tính macro recall (giống như evaluate_sc trong train_visobert_stl.py)
        if len(aspect_preds) > 0:
            _, recall_macro, _, _ = precision_recall_fscore_support(
                aspect_labels, aspect_preds, average='macro', zero_division=0
            )
            recall = recall_macro
        else:
            recall = 0.0
        
        # Tính overall recall để so sánh
        correct_mask = pred_df[correct_col] == 1
        true_positive = (has_label & correct_mask).sum()
        false_negative = (has_label & ~correct_mask).sum()
        total_positive = has_label.sum()
        recall_overall = true_positive / total_positive if total_positive > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"ASPECT: {aspect}")
        print(f"{'='*80}")
        print(f"Total samples co label that: {total_positive}")
        print(f"True Positive (TP): {true_positive}")
        print(f"False Negative (FN): {false_negative}")
        print(f"Recall (Macro): {recall:.2%} (theo code training)")
        print(f"Recall (Overall): {recall_overall:.2%} (TP/(TP+FN))")
        
        # Phan tich cac loi FN
        fn_mask = has_label & ~correct_mask
        fn_samples = pred_df[fn_mask].copy()
        
        if len(fn_samples) > 0:
            print(f"\n--- PHAN TICH FALSE NEGATIVES ({len(fn_samples)} samples) ---")
            
            # Thong ke loai loi
            error_types = defaultdict(int)
            error_details = []
            
            for idx, row in fn_samples.iterrows():
                pred_label = int(row[pred_col]) if pd.notna(row[pred_col]) else 0
                true_label = int(row[true_col]) if pd.notna(row[true_col]) else 0
                
                pred_str = label_map.get(pred_label, f'Unknown({pred_label})')
                true_str = label_map.get(true_label, f'Unknown({true_label})')
                
                error_type = f"Predict {pred_str} but True is {true_str}"
                error_types[error_type] += 1
                
                # Luu chi tiet
                text = row.get('data', 'N/A') if 'data' in row else 'N/A'
                text_short = text[:100] if isinstance(text, str) else 'N/A'
                error_details.append({
                    'sample_id': idx,
                    'text': text_short,
                    'text_full': text,
                    'predicted': pred_str,
                    'true': true_str,
                    'error_type': error_type
                })
                
                # Luu vao all_errors để xuất file
                all_errors.append({
                    'aspect': aspect,
                    'sample_id': idx,
                    'text': text if isinstance(text, str) else 'N/A',
                    'predicted': pred_str,
                    'predicted_num': pred_label,
                    'true': true_str,
                    'true_num': true_label,
                    'error_type': error_type
                })
            
            print("\nThong ke loai loi:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count} samples ({count/len(fn_samples):.1%})")
            
            # Hien thi mot so vi du loi
            print(f"\n--- VI DU LOI (top 10) ---")
            error_df = pd.DataFrame(error_details)
            
            # Nhom theo loai loi va lay vi du
            for error_type in sorted(error_types.keys(), key=lambda x: error_types[x], reverse=True)[:5]:
                examples = error_df[error_df['error_type'] == error_type].head(2)
                print(f"\n{error_type}:")
                for _, ex in examples.iterrows():
                    print(f"  Sample {ex['sample_id']}: {ex['text']}")
        
        # Phan tich Precision (de so sanh)
        pred_positive = (pred_df[pred_col].notna() & (pred_df[pred_col] != 0)).sum()
        if pred_positive > 0:
            precision = true_positive / pred_positive
            print(f"\nPrecision: {precision:.2%} (TP: {true_positive}, Predicted Positive: {pred_positive})")
    
    # Tong ket
    print(f"\n{'='*80}")
    print("TONG KET")
    print(f"{'='*80}")
    
    # Tinh recall cho tung aspect (dùng labeled_mask giống phần phân tích chi tiết)
    aspect_recalls = {}
    for aspect in aspects:
        pred_col = f"{aspect}_pred"
        true_col = f"{aspect}_true"
        
        if pred_col not in pred_df.columns:
            continue
        
        # Dùng labeled_mask từ CSV gốc (giống code training)
        if aspect not in labeled_mask:
            continue
        
        has_label = labeled_mask[aspect]
        
        if not has_label.any():
            continue
        
        # Tính macro recall (giống code training)
        aspect_preds = pred_df.loc[has_label, pred_col].values
        aspect_labels = pred_df.loc[has_label, true_col].values
        
        if len(aspect_preds) > 0:
            _, recall_macro, _, _ = precision_recall_fscore_support(
                aspect_labels, aspect_preds, average='macro', zero_division=0
            )
            recall = recall_macro
        else:
            recall = 0.0
        aspect_recalls[aspect] = recall
    
    # Sap xep theo recall
    sorted_recalls = sorted(aspect_recalls.items(), key=lambda x: x[1])
    
    print("\nRecall theo aspect (tu thap den cao):")
    for aspect, recall in sorted_recalls:
        print(f"  {aspect:15s}: {recall:.2%}")
    
    print(f"\nAspect co recall thap nhat: {sorted_recalls[0][0]} ({sorted_recalls[0][1]:.2%})")
    print(f"Aspect co recall cao nhat: {sorted_recalls[-1][0]} ({sorted_recalls[-1][1]:.2%})")
    
    # Xuat tat ca cac samples bi loi ra file txt
    if all_errors:
        print(f"\n{'='*80}")
        print(f"Dang xuat {len(all_errors)} samples bi loi ra file: {output_file}")
        print(f"{'='*80}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TAT CA CAC SAMPLES BI LOI RECALL - SENTIMENT CLASSIFICATION\n")
            f.write("="*80 + "\n\n")
            f.write(f"Tong so samples bi loi: {len(all_errors)}\n")
            f.write(f"Ngay tao: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Nhom theo aspect
            errors_by_aspect = defaultdict(list)
            for error in all_errors:
                errors_by_aspect[error['aspect']].append(error)
            
            # Xuat theo aspect
            for aspect in aspects:
                if aspect not in errors_by_aspect:
                    continue
                
                aspect_errors = errors_by_aspect[aspect]
                
                # Tinh recall cho aspect nay (dùng labeled_mask và macro recall)
                pred_col = f"{aspect}_pred"
                true_col = f"{aspect}_true"
                correct_col = f"{aspect}_correct"
                
                # Dùng labeled_mask từ CSV gốc (giống code training)
                if aspect not in labeled_mask:
                    continue
                
                has_label = labeled_mask[aspect]
                
                # Tính macro recall (giống code training)
                aspect_preds = pred_df.loc[has_label, pred_col].values
                aspect_labels = pred_df.loc[has_label, true_col].values
                
                if len(aspect_preds) > 0:
                    _, recall_macro, _, _ = precision_recall_fscore_support(
                        aspect_labels, aspect_preds, average='macro', zero_division=0
                    )
                    recall = recall_macro
                else:
                    recall = 0.0
                
                correct_mask = pred_df[correct_col] == 1
                true_positive = (has_label & correct_mask).sum()
                total_positive = has_label.sum()
                
                f.write("\n" + "="*80 + "\n")
                f.write(f"ASPECT: {aspect}\n")
                f.write(f"Recall Score: {recall:.2%}\n")
                f.write(f"So luong loi: {len(aspect_errors)}\n")
                f.write(f"Total samples co label: {total_positive}\n")
                f.write(f"True Positive (TP): {true_positive}\n")
                f.write(f"False Negative (FN): {len(aspect_errors)}\n")
                f.write("="*80 + "\n\n")
                
                # Thong ke loai loi
                error_types_count = defaultdict(int)
                for err in aspect_errors:
                    error_types_count[err['error_type']] += 1
                
                f.write("Thong ke loai loi:\n")
                for err_type, count in sorted(error_types_count.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  - {err_type}: {count} samples\n")
                f.write("\n")
                
                # Xuat chi tiet tung sample
                for i, err in enumerate(aspect_errors, 1):
                    f.write(f"\n--- Sample {i}/{len(aspect_errors)} ---\n")
                    f.write(f"Sample ID: {err['sample_id']}\n")
                    f.write(f"Aspect: {err['aspect']}\n")
                    f.write(f"Predicted: {err['predicted']} ({err['predicted_num']})\n")
                    f.write(f"True Label: {err['true']} ({err['true_num']})\n")
                    f.write(f"Error Type: {err['error_type']}\n")
                    f.write(f"Text:\n{err['text']}\n")
                    f.write("-" * 80 + "\n")
            
            # Tong ket
            f.write("\n\n" + "="*80 + "\n")
            f.write("TONG KET\n")
            f.write("="*80 + "\n\n")
            f.write("So luong loi va Recall theo aspect:\n")
            f.write(f"{'Aspect':<15s} {'Recall':<12s} {'So loi':<10s} {'Total samples':<15s}\n")
            f.write("-" * 80 + "\n")
            
            # Sap xep theo recall (tu thap den cao)
            aspect_summary = []
            for aspect in aspects:
                if aspect not in errors_by_aspect:
                    continue
                
                # Tinh recall (macro recall giống code training)
                pred_col = f"{aspect}_pred"
                true_col = f"{aspect}_true"
                correct_col = f"{aspect}_correct"
                
                # Dùng labeled_mask từ CSV gốc (giống code training)
                if aspect not in labeled_mask:
                    continue
                
                has_label = labeled_mask[aspect]
                
                # Tính macro recall
                aspect_preds = pred_df.loc[has_label, pred_col].values
                aspect_labels = pred_df.loc[has_label, true_col].values
                
                if len(aspect_preds) > 0:
                    _, recall_macro, _, _ = precision_recall_fscore_support(
                        aspect_labels, aspect_preds, average='macro', zero_division=0
                    )
                    recall = recall_macro
                else:
                    recall = 0.0
                
                correct_mask = pred_df[correct_col] == 1
                true_positive = (has_label & correct_mask).sum()
                total_positive = has_label.sum()
                
                aspect_summary.append({
                    'aspect': aspect,
                    'recall': recall,
                    'errors': len(errors_by_aspect[aspect]),
                    'total': total_positive
                })
            
            # Sap xep theo recall
            aspect_summary.sort(key=lambda x: x['recall'])
            
            for summary in aspect_summary:
                f.write(f"{summary['aspect']:<15s} {summary['recall']:<12.2%} {summary['errors']:<10d} {summary['total']:<15d}\n")
            
            f.write("\n")
            f.write(f"Aspect co recall thap nhat: {aspect_summary[0]['aspect']} ({aspect_summary[0]['recall']:.2%})\n")
            f.write(f"Aspect co recall cao nhat: {aspect_summary[-1]['aspect']} ({aspect_summary[-1]['recall']:.2%})\n")
        
        print(f"Da xuat thanh cong {len(all_errors)} samples bi loi ra file!")
        print(f"File: {output_file}")
    else:
        print("\nKhong co loi nao de xuat!")


if __name__ == "__main__":
    analyze_recall_errors()


