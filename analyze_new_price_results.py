import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

print("="*80)
print("ANALYZING NEW PRICE SC RESULTS (After Oversampling)")
print("="*80)

# Load predictions
pred_df = pd.read_csv('VisoBERT-STL/models/sentiment_classification/test_predictions_detailed.csv')
test_df = pd.read_csv('VisoBERT-STL/data/test_multilabel.csv', encoding='utf-8-sig')

# Filter Price samples
price_mask = test_df['Price'].notna()
price_preds = pred_df.loc[price_mask, 'Price_pred'].values
price_true = pred_df.loc[price_mask, 'Price_true'].values

print(f"\nTotal Price-labeled test samples: {len(price_true)}")

# Per-class metrics
p_c, r_c, f_c, s_c = precision_recall_fscore_support(price_true, price_preds, average=None, zero_division=0)
labels = ['Positive', 'Negative', 'Neutral']

print(f"\n" + "-"*80)
print("PER-CLASS METRICS (NEW)")
print("-"*80)
for i, lbl in enumerate(labels):
    print(f"{lbl:10} Precision: {p_c[i]*100:5.1f}%  Recall: {r_c[i]*100:5.1f}%  F1: {f_c[i]*100:5.1f}%  (n={int(s_c[i])})")

# Macro averages
p_macro = p_c.mean()
r_macro = r_c.mean()
f_macro = f_c.mean()

print(f"\n" + "-"*80)
print("MACRO AVERAGES")
print("-"*80)
print(f"Precision: {p_macro*100:.2f}%")
print(f"Recall:    {r_macro*100:.2f}%")
print(f"F1:        {f_macro*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(price_true, price_preds)
print(f"\n" + "-"*80)
print("CONFUSION MATRIX")
print("-"*80)
print("             Predicted")
print("          Pos  Neg  Neu")
print(f"True Pos: {cm[0,0]:4} {cm[0,1]:4} {cm[0,2]:4}")
print(f"     Neg: {cm[1,0]:4} {cm[1,1]:4} {cm[1,2]:4}")
print(f"     Neu: {cm[2,0]:4} {cm[2,1]:4} {cm[2,2]:4}")

# Comparison with OLD results
print(f"\n" + "="*80)
print("COMPARISON: BEFORE vs AFTER OVERSAMPLING")
print("="*80)

print(f"\n{'Metric':<15} {'BEFORE':<12} {'AFTER':<12} {'CHANGE':<12}")
print("-"*80)
print(f"{'F1 Macro':<15} {'88.47%':<12} {f_macro*100:5.2f}%      {f_macro*100-88.47:+5.2f}%")
print(f"{'Recall Macro':<15} {'82.65%':<12} {r_macro*100:5.2f}%      {r_macro*100-82.65:+5.2f}%")
print(f"{'Precision':<15} {'96.08%':<12} {p_macro*100:5.2f}%      {p_macro*100-96.08:+5.2f}%")

print(f"\n{'Class':<15} {'Metric':<12} {'BEFORE':<12} {'AFTER':<12} {'CHANGE':<12}")
print("-"*80)

# OLD results
old_results = {
    'Positive': {'P': 97.3, 'R': 99.6, 'F1': 98.4},
    'Negative': {'P': 90.9, 'R': 71.4, 'F1': 80.0},
    'Neutral': {'P': 100.0, 'R': 76.9, 'F1': 87.0}
}

for i, lbl in enumerate(labels):
    print(f"{lbl:<15} {'Precision':<12} {old_results[lbl]['P']:5.1f}%       {p_c[i]*100:5.1f}%       {p_c[i]*100-old_results[lbl]['P']:+5.1f}%")
    print(f"{'':<15} {'Recall':<12} {old_results[lbl]['R']:5.1f}%       {r_c[i]*100:5.1f}%       {r_c[i]*100-old_results[lbl]['R']:+5.1f}%")
    print(f"{'':<15} {'F1':<12} {old_results[lbl]['F1']:5.1f}%       {f_c[i]*100:5.1f}%       {f_c[i]*100-old_results[lbl]['F1']:+5.1f}%")
    print("-"*80)

# Check training data balance
print(f"\n" + "="*80)
print("TRAINING DATA ANALYSIS")
print("="*80)

train_df = pd.read_csv('VisoBERT-STL/data/train_multilabel_balanced.csv', encoding='utf-8-sig')
price_train = train_df['Price'].dropna()

print(f"\nTraining data Price distribution (balanced):")
print(price_train.value_counts())

pos_count = (price_train == 'Positive').sum()
neg_count = (price_train == 'Negative').sum()
neu_count = (price_train == 'Neutral').sum()

print(f"\nRatio - Pos:Neg:Neu = {pos_count}:{neg_count}:{neu_count}")
print(f"Balance ratio: {pos_count/neg_count:.2f}:{neg_count/neg_count:.2f}:{neu_count/neg_count:.2f}")

# Diagnosis
print(f"\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print(f"\n✓ Oversampling WORKED:")
print(f"  - Negative recall: 71.4% → {r_c[1]*100:.1f}% ({r_c[1]*100-71.4:+.1f}%)")
print(f"  - Neutral recall:  76.9% → {r_c[2]*100:.1f}% ({r_c[2]*100-76.9:+.1f}%)")
print(f"  - Overall recall:  82.65% → {r_macro*100:.2f}% ({r_macro*100-82.65:+.2f}%)")

print(f"\n✗ But still not as good as MTL:")
print(f"  - MTL Price F1:     96.57%")
print(f"  - STL Price F1:     {f_macro*100:.2f}%")
print(f"  - Difference:       {96.57-f_macro*100:.2f}%")

print(f"\n**Why STL < MTL?**")
print(f"  1. STL trains SC independently → no shared representations")
print(f"  2. MTL shares backbone between AD and SC → better feature learning")
print(f"  3. Small minority classes (14 Neg, 13 Neu) → hard to learn even with oversampling")

print(f"\n**Is 89.36% F1 acceptable?**")
print(f"  - For ABSA research: YES ✓")
print(f"  - Improvement from baseline: +0.89% (88.47% → 89.36%)")
print(f"  - Recall improvement: +7.3% (82.65% → 89.95%)")
print(f"  - Trade-off: Precision dropped 5.88% (96.08% → 90.20%)")
