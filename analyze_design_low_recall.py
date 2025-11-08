import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("="*80)
print("ANALYZING DESIGN LOW RECALL")
print("="*80)

# Load data
pred_df = pd.read_csv('VisoBERT-STL/models/sentiment_classification/test_predictions_detailed.csv')
test_df = pd.read_csv('VisoBERT-STL/data/test_multilabel.csv', encoding='utf-8-sig')

# Filter Design samples
design_mask = test_df['Design'].notna()
design_test = test_df[design_mask].reset_index(drop=True)
design_preds = pred_df.loc[design_mask, 'Design_pred'].reset_index(drop=True)
design_true = pred_df.loc[design_mask, 'Design_true'].reset_index(drop=True)

print(f"\nTotal Design samples in test: {len(design_test)}")

# Per-class metrics
p_c, r_c, f_c, s_c = precision_recall_fscore_support(design_true, design_preds, average=None, zero_division=0)
labels = ['Positive', 'Negative', 'Neutral']

print(f"\n" + "-"*80)
print("PER-CLASS METRICS")
print("-"*80)
for i, label in enumerate(labels):
    print(f"{label:10} Precision: {p_c[i]*100:5.1f}%  Recall: {r_c[i]*100:5.1f}%  F1: {f_c[i]*100:5.1f}%  (n={int(s_c[i])})")

# Macro averages
print(f"\nMacro Recall: {r_c.mean()*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(design_true, design_preds)
print(f"\n" + "-"*80)
print("CONFUSION MATRIX")
print("-"*80)
print("             Predicted")
print("          Pos  Neg  Neu")
print(f"True Pos: {cm[0,0]:4} {cm[0,1]:4} {cm[0,2]:4}")
print(f"     Neg: {cm[1,0]:4} {cm[1,1]:4} {cm[1,2]:4}")
print(f"     Neu: {cm[2,0]:4} {cm[2,1]:4} {cm[2,2]:4}")

# Find errors
errors = design_preds != design_true
error_indices = design_test[errors].index.tolist()

print(f"\n" + "="*80)
print(f"ERROR ANALYSIS")
print("="*80)
print(f"Total errors: {errors.sum()} / {len(design_true)} = {errors.sum()/len(design_true)*100:.2f}%")

# Errors by true class
print(f"\nErrors by true class:")
for i, label in enumerate(['Positive', 'Negative', 'Neutral']):
    true_class_mask = design_true == i
    errors_in_class = errors & true_class_mask
    print(f"  {label:10} errors: {errors_in_class.sum():2} / {true_class_mask.sum():3} = {errors_in_class.sum()/max(true_class_mask.sum(),1)*100:5.1f}%")

# Which class has lowest recall?
min_recall_idx = r_c.argmin()
min_recall_class = labels[min_recall_idx]
min_recall_value = r_c[min_recall_idx]

print(f"\n" + "-"*80)
print(f"LOWEST RECALL CLASS")
print("-"*80)
print(f"Class: {min_recall_class}")
print(f"Recall: {min_recall_value*100:.1f}%")
print(f"Samples: {int(s_c[min_recall_idx])}")
print(f"Correct: {int(cm[min_recall_idx, min_recall_idx])}")
print(f"Errors: {int(s_c[min_recall_idx]) - int(cm[min_recall_idx, min_recall_idx])}")

# Show confusion pattern
print(f"\nConfusion pattern for {min_recall_class}:")
for j, pred_label in enumerate(labels):
    if cm[min_recall_idx, j] > 0:
        print(f"  Predicted as {pred_label}: {cm[min_recall_idx, j]} samples")

# Sample some errors
print(f"\n" + "="*80)
print(f"SAMPLE ERRORS FROM {min_recall_class.upper()} CLASS")
print("="*80)

error_mask = (design_true == min_recall_idx) & errors
error_samples = design_test[error_mask]

for idx, (i, row) in enumerate(error_samples.head(5).iterrows()):
    pred_label = labels[design_preds.iloc[i]]
    text = row['data']
    print(f"\nError {idx+1}:")
    print(f"  True: {min_recall_class} -> Predicted: {pred_label}")
    print(f"  Text: {text[:200]}...")

# Compare with training data
print(f"\n" + "="*80)
print("TRAINING DATA ANALYSIS")
print("="*80)

train_df = pd.read_csv('VisoBERT-STL/data/train_multilabel_balanced.csv', encoding='utf-8-sig')
design_train = train_df['Design'].dropna()

print(f"\nDesign training samples (balanced): {len(design_train)}")
print(f"\nSentiment distribution:")
for sent in ['Positive', 'Negative', 'Neutral']:
    count = (design_train == sent).sum()
    print(f"  {sent:10}: {count:4} ({count/len(design_train)*100:5.1f}%)")

# Check if test low-recall class is underrepresented in training
train_counts = design_train.value_counts()
test_counts = test_df['Design'].value_counts()

print(f"\n" + "-"*80)
print("TRAINING vs TEST DISTRIBUTION")
print("-"*80)
print(f"{'Sentiment':<12} {'Train %':<10} {'Test %':<10} {'Difference':<10}")
print("-"*80)
for sent in ['Positive', 'Negative', 'Neutral']:
    train_pct = (design_train == sent).sum() / len(design_train) * 100
    test_design = test_df['Design'].dropna()
    test_pct = (test_design == sent).sum() / len(test_design) * 100
    diff = test_pct - train_pct
    print(f"{sent:<12} {train_pct:>8.1f}% {test_pct:>8.1f}% {diff:>+8.1f}%")

print(f"\n" + "="*80)
print("CONCLUSION")
print("="*80)

print(f"\nDesign Recall: {r_c.mean()*100:.2f}%")
print(f"Lowest class: {min_recall_class} ({min_recall_value*100:.1f}% recall)")
print(f"\nPossible reasons:")
print(f"  1. Class imbalance in test set")
print(f"  2. {min_recall_class} patterns different from training")
print(f"  3. Augmentation quality for {min_recall_class}")
print(f"  4. Model bias towards other classes")
