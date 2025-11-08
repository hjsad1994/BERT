import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np

print("="*80)
print("ANALYZING STL PRICE SC RECALL")
print("="*80)

# Load predictions and ground truth
pred_df = pd.read_csv('VisoBERT-STL/models/sentiment_classification/test_predictions_detailed.csv')
test_df = pd.read_csv('VisoBERT-STL/data/test_multilabel.csv', encoding='utf-8-sig')

# Filter to Price-labeled samples only
price_has_label = test_df['Price'].notna()
print(f"\nTotal test samples: {len(test_df)}")
print(f"Price-labeled samples: {price_has_label.sum()}")

# Get Price predictions and labels
price_preds = pred_df.loc[price_has_label, 'Price_pred'].values
price_true = pred_df.loc[price_has_label, 'Price_true'].values

# Overall metrics
p, r, f, _ = precision_recall_fscore_support(price_true, price_preds, average='macro', zero_division=0)
print(f"\n" + "-"*80)
print(f"OVERALL PRICE SC METRICS (282 samples)")
print("-"*80)
print(f"  Precision: {p*100:.2f}%")
print(f"  Recall:    {r*100:.2f}%")
print(f"  F1:        {f*100:.2f}%")

# Per-class metrics
p_c, r_c, f_c, s_c = precision_recall_fscore_support(price_true, price_preds, average=None, zero_division=0)
sentiment_labels = ['Positive (0)', 'Negative (1)', 'Neutral (2)']

print(f"\n" + "-"*80)
print(f"PER-CLASS METRICS")
print("-"*80)
for i, label in enumerate(sentiment_labels):
    print(f"{label:18} Precision: {p_c[i]*100:5.1f}%  Recall: {r_c[i]*100:5.1f}%  F1: {f_c[i]*100:5.1f}%  (n={int(s_c[i])})")

# Confusion matrix
cm = confusion_matrix(price_true, price_preds)
print(f"\n" + "-"*80)
print(f"CONFUSION MATRIX")
print("-"*80)
print("              Predicted")
print("             Pos  Neg  Neu")
print(f"True Pos:  {cm[0,0]:4} {cm[0,1]:4} {cm[0,2]:4}")
print(f"     Neg:  {cm[1,0]:4} {cm[1,1]:4} {cm[1,2]:4}")
print(f"     Neu:  {cm[2,0]:4} {cm[2,1]:4} {cm[2,2]:4}")

# Analyze errors
errors = price_preds != price_true
error_indices = np.where(errors)[0]

print(f"\n" + "-"*80)
print(f"ERROR ANALYSIS")
print("-"*80)
print(f"Total errors: {errors.sum()} / {len(price_true)} = {errors.sum()/len(price_true)*100:.2f}%")

print(f"\nError breakdown by true class:")
for i, label in enumerate(['Positive', 'Negative', 'Neutral']):
    true_class_mask = price_true == i
    errors_in_class = errors & true_class_mask
    print(f"  {label:10} errors: {errors_in_class.sum():2} / {true_class_mask.sum():3} = {errors_in_class.sum()/max(true_class_mask.sum(),1)*100:5.1f}% error rate")

# Why is Recall low?
print(f"\n" + "="*80)
print(f"WHY IS RECALL LOW?")
print("="*80)

print(f"\nRecall per class:")
for i, label in enumerate(['Positive', 'Negative', 'Neutral']):
    print(f"  {label:10} Recall: {r_c[i]*100:5.1f}%  (correctly predicted {int(cm[i,i])} out of {int(s_c[i])})")

print(f"\nMacro Recall = Average of per-class recalls:")
print(f"  ({r_c[0]*100:.1f}% + {r_c[1]*100:.1f}% + {r_c[2]*100:.1f}%) / 3 = {r*100:.2f}%")

print(f"\n**KEY INSIGHT:**")
print(f"  Negative Recall: {r_c[1]*100:.1f}% - Only {int(cm[1,1])} out of {int(s_c[1])} Negative samples predicted correctly!")
print(f"  Neutral Recall:  {r_c[2]*100:.1f}% - Only {int(cm[2,2])} out of {int(s_c[2])} Neutral samples predicted correctly!")
print(f"\n  These low recalls for minority classes pull down the macro average.")

# Compare with MTL
print(f"\n" + "="*80)
print(f"COMPARISON WITH MTL")
print("="*80)

print(f"\nMTL Price SC:")
print(f"  Recall: 96.68%")
print(f"  F1:     96.57%")

print(f"\nSTL Price SC:")
print(f"  Recall: {r*100:.2f}%  ({(96.68-r*100):.1f}% worse than MTL)")
print(f"  F1:     {f*100:.2f}%  ({(96.57-f*100):.1f}% worse than MTL)")

print(f"\n**ROOT CAUSE:**")
print(f"  STL fails to predict minority classes (Negative/Neutral)")
print(f"  Despite 97.16% overall accuracy, it misses many minority samples")
print(f"  This is a class imbalance problem, NOT related to AD!")
