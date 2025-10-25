import pandas as pd

aspects = ['Battery', 'Camera', 'Performance', 'Display', 'Design', 
           'Packaging', 'Price', 'Shop_Service', 'Shipping', 'General', 'Others']

# Analyze TEST set
print("=" * 80)
print("TEST SET DISTRIBUTION")
print("=" * 80)

df_test = pd.read_csv('multi_label/data/test_multilabel.csv', encoding='utf-8-sig')
test_total = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

print(f"\n{'Aspect':<15} {'Positive':<10} {'Negative':<10} {'Neutral':<10}")
print("-" * 50)

for aspect in aspects:
    counts = df_test[aspect].value_counts()
    pos = counts.get('Positive', 0)
    neg = counts.get('Negative', 0)
    neu = counts.get('Neutral', 0)
    
    print(f"{aspect:<15} {pos:<10} {neg:<10} {neu:<10}")
    
    test_total['Positive'] += pos
    test_total['Negative'] += neg
    test_total['Neutral'] += neu

total_samples = sum(test_total.values())
print(f"\n{'TOTAL':<15} {test_total['Positive']:<10} {test_total['Negative']:<10} {test_total['Neutral']:<10}")
print(f"\nPercentage:")
print(f"  Positive: {test_total['Positive']/total_samples*100:.1f}%")
print(f"  Negative: {test_total['Negative']/total_samples*100:.1f}%")
print(f"  Neutral:  {test_total['Neutral']/total_samples*100:.1f}%")

# Analyze TRAIN set (balanced)
print("\n" + "=" * 80)
print("TRAIN SET (BALANCED) DISTRIBUTION")
print("=" * 80)

df_train = pd.read_csv('multi_label/data/train_multilabel_balanced.csv', encoding='utf-8-sig')
train_total = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

print(f"\n{'Aspect':<15} {'Positive':<10} {'Negative':<10} {'Neutral':<10}")
print("-" * 50)

for aspect in aspects:
    counts = df_train[aspect].value_counts()
    pos = counts.get('Positive', 0)
    neg = counts.get('Negative', 0)
    neu = counts.get('Neutral', 0)
    
    print(f"{aspect:<15} {pos:<10} {neg:<10} {neu:<10}")
    
    train_total['Positive'] += pos
    train_total['Negative'] += neg
    train_total['Neutral'] += neu

total_samples = sum(train_total.values())
print(f"\n{'TOTAL':<15} {train_total['Positive']:<10} {train_total['Negative']:<10} {train_total['Neutral']:<10}")
print(f"\nPercentage:")
print(f"  Positive: {train_total['Positive']/total_samples*100:.1f}%")
print(f"  Negative: {train_total['Negative']/total_samples*100:.1f}%")
print(f"  Neutral:  {train_total['Neutral']/total_samples*100:.1f}%")
