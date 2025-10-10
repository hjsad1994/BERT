"""
Error Analysis trên TẬP VALIDATION - để tune hyperparameters
"""
import sys
sys.path.append('.')

from error_analysis import ErrorAnalyzer

if __name__ == '__main__':
    # Phân tích trên VALIDATION set
    analyzer = ErrorAnalyzer(
        test_file='data/validation.csv',  # ← Đổi sang VALIDATION
        predictions_file='validation_predictions.csv'  # Cần có file này
    )
    
    print("="*70)
    print("⚠️  ĐANG PHÂN TÍCH TRÊN TẬP VALIDATION (không phải TEST)")
    print("="*70)
    
    analyzer.run_full_analysis()
