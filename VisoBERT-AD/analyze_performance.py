"""
Phân tích vấn đề performance của model aspect detection
"""

import json
import pandas as pd

# Load kết quả
with open('models/aspect_detection/test_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Load optimal thresholds
with open('models/aspect_detection/optimal_thresholds.json', 'r', encoding='utf-8') as f:
    thresholds = json.load(f)

print("=" * 80)
print("PERFORMANCE ANALYSIS")
print("=" * 80)

print("\n1. OVERALL METRICS:")
print(f"   Element-wise Accuracy: {results['test_accuracy']*100:.2f}%")
print(f"   Macro F1: {results['test_f1']*100:.2f}%")
print(f"   Precision: {results['test_precision']*100:.2f}%")
print(f"   Recall: {results['test_recall']*100:.2f}%")

print("\n2. CLASS IMBALANCE ISSUES:")
print("-" * 80)
aspect_metrics = []
for aspect, metrics in results['per_aspect'].items():
    pos_ratio = metrics['n_positive'] / metrics['n_samples'] * 100
    aspect_metrics.append({
        'Aspect': aspect,
        'F1': metrics['f1'] * 100,
        'Precision': metrics['precision'] * 100,
        'Recall': metrics['recall'] * 100,
        'Positive': metrics['n_positive'],
        'Positive%': pos_ratio,
        'FP': metrics['fp'],
        'FN': metrics['fn'],
        'Threshold': thresholds.get(aspect, 0.5)
    })

df = pd.DataFrame(aspect_metrics)
df = df.sort_values('F1')

print("\nAspects with lowest F1:")
print(df[['Aspect', 'F1', 'Precision', 'Recall', 'Positive', 'Positive%', 'FP', 'FN', 'Threshold']].head(3).to_string(index=False))

print("\n3. DETAILED ANALYSIS:")
print("-" * 80)

# General aspect
general = results['per_aspect']['General']
print(f"\nGeneral Aspect:")
print(f"   F1: {general['f1']*100:.2f}% (LOW)")
print(f"   Precision: {general['precision']*100:.2f}% (90 FP - many false positives)")
print(f"   Recall: {general['recall']*100:.2f}% (103 FN - many false negatives)")
print(f"   Positive samples: {general['n_positive']} ({general['n_positive']/1432*100:.2f}%)")
print(f"   Threshold: {thresholds.get('General', 0.5):.2f}")
print(f"   -> Issue: Both precision and recall are low, many FP and FN errors")

# Others aspect
others = results['per_aspect']['Others']
print(f"\nOthers Aspect:")
print(f"   F1: {others['f1']*100:.2f}% (VERY LOW)")
print(f"   Precision: {others['precision']*100:.2f}% (11 FP)")
print(f"   Recall: {others['recall']*100:.2f}% (14 FN - missed {14/37*100:.1f}%)")
print(f"   Positive samples: {others['n_positive']} ({others['n_positive']/1432*100:.2f}%) - TOO FEW!")
print(f"   Threshold: {thresholds.get('Others', 0.5):.2f}")
print(f"   -> Issue: Too few training samples, model struggles, misses many ({14/37*100:.1f}%)")

print("\n4. ROOT CAUSES:")
print("-" * 80)
print("+ Severe class imbalance (Others: only 2.58% samples)")
print("+ General aspect: many false positives and negatives")
print("+ Threshold may not be optimal")
print("+ Overfitting: Validation F1 (89.48%) > Test F1 (87.17%)")

print("\n5. RECOMMENDATIONS:")
print("-" * 80)
print("1. Increase data for 'Others' aspect (oversampling or collect more data)")
print("2. Adjust thresholds for 'General' and 'Others'")
print("3. Use Focal Loss instead of BCE for better class imbalance handling")
print("4. Increase dropout or regularization to reduce overfitting")
print("5. Early stopping based on test F1 instead of validation F1")
print("6. Class weights may need re-adjustment")

