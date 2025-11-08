import pandas as pd
import numpy as np
import sys

# Set encoding for output
sys.stdout.reconfigure(encoding='utf-8')

print("="*80)
print("ANALYZING PRICE NEGATIVE CLASS ERRORS")
print("="*80)

# Load data
pred_df = pd.read_csv('VisoBERT-STL/models/sentiment_classification/test_predictions_detailed.csv')
test_df = pd.read_csv('VisoBERT-STL/data/test_multilabel.csv', encoding='utf-8-sig')

# Get Price samples
price_mask = test_df['Price'].notna()
price_test = test_df[price_mask].reset_index(drop=True)
price_preds = pred_df.loc[price_mask, 'Price_pred'].reset_index(drop=True)
price_true = pred_df.loc[price_mask, 'Price_true'].reset_index(drop=True)

# Find Negative errors
neg_mask = (price_true == 1)
neg_correct = (price_preds == 1) & neg_mask
neg_errors = (price_preds != 1) & neg_mask

print(f"\nNegative class performance:")
print(f"  Total Negative samples: {neg_mask.sum()}")
print(f"  Correct: {neg_correct.sum()}")
print(f"  Errors: {neg_errors.sum()}")
print(f"  Recall: {neg_correct.sum() / neg_mask.sum() * 100:.1f}%")

print(f"\n" + "-"*80)
print(f"THE 4 NEGATIVE ERRORS:")
print("-"*80)

error_indices = price_test[neg_errors].index.tolist()
for i, idx in enumerate(error_indices):
    text = price_test.iloc[idx]['data']
    pred_label = ['Positive', 'Negative', 'Neutral'][price_preds.iloc[idx]]
    
    print(f"\nError {i+1} (Sample index {idx}):")
    print(f"  True: Negative -> Predicted: {pred_label}")
    print(f"  Text: {text[:300]}...")

# Analyze training data for Negative
print(f"\n" + "="*80)
print("TRAINING DATA ANALYSIS - NEGATIVE CLASS")
print("="*80)

train_df = pd.read_csv('VisoBERT-STL/data/train_multilabel_balanced.csv', encoding='utf-8-sig')
price_train_neg = train_df[train_df['Price'] == 'Negative']

print(f"\nNegative Price samples in training (balanced):")
print(f"  Count: {len(price_train_neg)}")
print(f"\nSample negative reviews:")
for i in range(min(3, len(price_train_neg))):
    print(f"\n  {i+1}. {price_train_neg.iloc[i]['data'][:200]}...")

# Check if test negative samples are similar to training
print(f"\n" + "-"*80)
print("HYPOTHESIS: Why Negative not improving?")
print("-"*80)

print(f"\n1. Data size:")
print(f"   - Training: {len(price_train_neg)} Negative samples")
print(f"   - Test: {neg_mask.sum()} Negative samples")
print(f"   - Ratio: {len(price_train_neg) / neg_mask.sum():.1f}x training data")

print(f"\n2. Class confusion:")
print(f"   - All 4 errors predicted as: {['Positive']*4}")
print(f"   - Model biased towards Positive class")

print(f"\n3. Possible reasons:")
print(f"   a) Negative reviews mention positive aspects first")
print(f"   b) Implicit negative sentiment (e.g., 'Chưa bao giờ nghĩ là...')")
print(f"   c) Mixed sentiment with negative price comment")
print(f"   d) Model focuses on dominant sentiment in text")

# Compare with original training data
train_orig = pd.read_csv('VisoBERT-STL/data/train_multilabel.csv', encoding='utf-8-sig')
price_train_neg_orig = train_orig[train_orig['Price'] == 'Negative']

print(f"\n" + "-"*80)
print("ORIGINAL vs BALANCED TRAINING DATA")
print("-"*80)
print(f"Original Negative samples: {len(price_train_neg_orig)}")
print(f"Balanced Negative samples: {len(price_train_neg)}")
print(f"Augmentation factor: {len(price_train_neg) / len(price_train_neg_orig):.1f}x")

# Check for duplicate augmentation
if len(price_train_neg) > len(price_train_neg_orig):
    print(f"\nNote: {len(price_train_neg) - len(price_train_neg_orig)} samples were augmented")
    print("If augmentation is just duplication, model won't learn better!")
