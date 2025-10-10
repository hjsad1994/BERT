"""
Error Analysis trên TẬP TRAIN - để debug training process
"""
import sys
sys.path.append('.')

from error_analysis import ErrorAnalyzer

if __name__ == '__main__':
    # Phân tích trên TRAIN set
    analyzer = ErrorAnalyzer(
        test_file='data/train.csv',  # ← Đổi sang TRAIN
        predictions_file='train_predictions.csv'  # Cần có file này
    )
    
    print("="*70)
    print("⚠️  ĐANG PHÂN TÍCH TRÊN TẬP TRAIN (không phải TEST)")
    print("="*70)
    
    analyzer.run_full_analysis()
