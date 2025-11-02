"""
Analyze overfitting for Others aspect
"""

import json
import pandas as pd

# Load results
results = json.load(open('models/aspect_detection/test_results.json', encoding='utf-8'))
history = pd.read_csv('models/aspect_detection/training_history.csv')

print("=" * 80)
print("PHAN TICH OVERFITTING CHO OTHERS ASPECT")
print("=" * 80)

print("\n1. Validation Others F1 qua cac epochs:")
print("-" * 80)
for i, row in history.iterrows():
    others_f1 = row.get('Others_f1', 0)
    if others_f1 > 0:
        print(f"  Epoch {int(row['epoch'])}: {others_f1*100:.2f}%")

best_val_f1 = history['Others_f1'].max() if 'Others_f1' in history.columns else 0
test_f1 = results['per_aspect']['Others']['f1']
gap = (best_val_f1 - test_f1) * 100

print("\n2. So sanh Validation vs Test:")
print("-" * 80)
print(f"  Best Validation F1: {best_val_f1*100:.2f}%")
print(f"  Test F1:            {test_f1*100:.2f}%")
print(f"  Gap:                {gap:.2f}%")

if gap > 10:
    print(f"\n  -> OVERFITTING NGHIEM TRONG! Gap > 10%")
elif gap > 5:
    print(f"\n  -> Co overfitting nhe. Gap > 5%")
else:
    print(f"\n  -> Khong co overfitting ro ret")

print("\n3. Test Others metrics chi tiet:")
print("-" * 80)
o = results['per_aspect']['Others']
print(f"  F1:        {o['f1']*100:.2f}%")
print(f"  Precision: {o['precision']*100:.2f}%")
print(f"  Recall:    {o['recall']*100:.2f}%")
print(f"  TP: {o['tp']}, FP: {o['fp']}, FN: {o['fn']}, TN: {o['tn']}")
print(f"  Positive samples: {o['n_positive']} ({o['n_positive']/1432*100:.2f}%)")

print("\n4. Nguyen nhan co the:")
print("-" * 80)
print(f"  - Others co qua it du lieu (chi {o['n_positive']} samples = {o['n_positive']/1432*100:.2f}%)")
print(f"  - Class weight rat cao (22.904) -> model overfocus vao positive class")
print(f"  - Model hoc qua tot tren validation nhung khong generalize sang test")
print(f"  - Threshold 0.17 rat thap -> co the khong optimal cho test set")

print("\n5. De xuat cai thien:")
print("-" * 80)
print("  1. Giam class weight cho Others (tren 22.904)")
print("  2. Tang dropout hoac regularization")
print("  3. Early stopping khi validation F1 khong cai thien")
print("  4. Oversampling cho Others aspect")
print("  5. Dieu chinh threshold lai cho test set")
print("  6. Su dung data augmentation cho Others")


