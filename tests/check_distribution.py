import pandas as pd

# Load validation data
df_val = pd.read_csv('data/validation.csv', encoding='utf-8-sig')
df_test = pd.read_csv('data/test.csv', encoding='utf-8-sig')

print("="*70)
print("VALIDATION SET DISTRIBUTION")
print("="*70)
print(f"Total samples: {len(df_val)}")
print("\nSentiment distribution:")
print(df_val['sentiment'].value_counts())
print("\nPercentages:")
print(df_val['sentiment'].value_counts(normalize=True) * 100)

print("\n" + "="*70)
print("TEST SET DISTRIBUTION")
print("="*70)
print(f"Total samples: {len(df_test)}")
print("\nSentiment distribution:")
print(df_test['sentiment'].value_counts())
print("\nPercentages:")
print(df_test['sentiment'].value_counts(normalize=True) * 100)

# Check imbalance ratio
val_counts = df_val['sentiment'].value_counts()
test_counts = df_test['sentiment'].value_counts()

print("\n" + "="*70)
print("IMBALANCE ANALYSIS")
print("="*70)
val_ratio = val_counts.max() / val_counts.min()
test_ratio = test_counts.max() / test_counts.min()
print(f"Validation imbalance ratio: {val_ratio:.2f}:1")
print(f"Test imbalance ratio: {test_ratio:.2f}:1")

if val_ratio > 2 or test_ratio > 2:
    print("\n⚠️  DATA IS IMBALANCED (ratio > 2:1)")
    print("👉 Recommendation: Use eval_f1 instead of eval_accuracy")
else:
    print("\n✅ DATA IS RELATIVELY BALANCED")
    print("👉 Recommendation: eval_accuracy or eval_f1 are both good")
