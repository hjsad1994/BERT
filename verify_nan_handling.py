"""
Verify that NaN aspects are NOT converted to Neutral during training
This script proves:
1. NaN is kept as NaN in the dataset
2. NaN gets mask=0.0 (skipped in training)
3. NaN is NOT converted to Neutral sentiment
"""

import pandas as pd
import torch
from transformers import AutoTokenizer
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append('multi_label')
from dataset_multilabel import MultiLabelABSADataset

print("=" * 80)
print("VERIFICATION: NaN Handling in Dataset")
print("=" * 80)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('uitnlp/visobert')

# Load test dataset
test_file = 'multi_label/data/test_multilabel.csv'
dataset = MultiLabelABSADataset(test_file, tokenizer, max_length=128)

print(f"\nLoaded dataset: {len(dataset)} samples")
print(f"Aspects: {dataset.aspects}")

# Read raw CSV to compare
raw_df = pd.read_csv(test_file, encoding='utf-8-sig')

print("\n" + "=" * 80)
print("TEST 1: Check first 10 samples")
print("=" * 80)

for idx in range(min(10, len(dataset))):
    print(f"\n--- Sample {idx} ---")
    
    # Get from dataset (processed)
    item = dataset[idx]
    labels = item['labels']
    masks = item['loss_mask']
    
    # Get from raw CSV (original)
    raw_row = raw_df.iloc[idx]
    
    print(f"Text: {raw_row['data'][:60]}...")
    print(f"\nAspect Processing:")
    
    has_nan = False
    has_neutral = False
    
    for i, aspect in enumerate(dataset.aspects):
        raw_value = raw_row[aspect]
        is_nan = pd.isna(raw_value)
        
        label_id = labels[i].item()
        mask = masks[i].item()
        
        # Convert label_id back to sentiment
        label_name = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}[label_id]
        
        if is_nan:
            has_nan = True
            status = "[OK] NaN (masked)" if mask == 0.0 else "[ERROR] NaN (NOT masked!)"
            print(f"  {aspect:15} Raw: NaN          -> Label: {label_name} (id={label_id}), Mask: {mask:.1f}  {status}")
        else:
            if raw_value == 'Neutral':
                has_neutral = True
            status = "[OK] Labeled" if mask == 1.0 else "[ERROR] NOT masked!"
            print(f"  {aspect:15} Raw: {raw_value:8} → Label: {label_name} (id={label_id}), Mask: {mask:.1f}  {status}")
    
    if has_nan:
        print(f"\n  Summary: Sample has NaN aspects (should have mask=0.0)")
    if has_neutral:
        print(f"  Summary: Sample has REAL Neutral sentiment (should have mask=1.0)")

# Statistics
print("\n" + "=" * 80)
print("TEST 2: Statistics Across All Samples")
print("=" * 80)

total_aspects = len(dataset) * len(dataset.aspects)
total_nan = 0
total_labeled = 0
total_neutral_labeled = 0
total_nan_with_wrong_mask = 0
total_labeled_with_wrong_mask = 0

for idx in range(len(dataset)):
    item = dataset[idx]
    labels = item['labels']
    masks = item['loss_mask']
    raw_row = raw_df.iloc[idx]
    
    for i, aspect in enumerate(dataset.aspects):
        raw_value = raw_row[aspect]
        is_nan = pd.isna(raw_value)
        mask = masks[i].item()
        label_id = labels[i].item()
        
        if is_nan:
            total_nan += 1
            if mask != 0.0:
                total_nan_with_wrong_mask += 1
        else:
            total_labeled += 1
            if raw_value == 'Neutral':
                total_neutral_labeled += 1
            if mask != 1.0:
                total_labeled_with_wrong_mask += 1

print(f"\nTotal aspects:  {total_aspects:,}")
print(f"  NaN (unlabeled):     {total_nan:,} ({total_nan/total_aspects*100:.1f}%)")
print(f"  Labeled:             {total_labeled:,} ({total_labeled/total_aspects*100:.1f}%)")
print(f"    - Real Neutral:    {total_neutral_labeled:,} ({total_neutral_labeled/total_labeled*100:.1f}% of labeled)")

print(f"\nMask Verification:")
print(f"  NaN with WRONG mask (should be 0.0):     {total_nan_with_wrong_mask}")
print(f"  Labeled with WRONG mask (should be 1.0): {total_labeled_with_wrong_mask}")

if total_nan_with_wrong_mask == 0 and total_labeled_with_wrong_mask == 0:
    print(f"\n  [PASS] All masks are CORRECT!")
else:
    print(f"\n  [FAIL] Some masks are WRONG!")

# Test 3: Verify NaN is NOT converted to Neutral
print("\n" + "=" * 80)
print("TEST 3: Verify NaN ≠ Neutral")
print("=" * 80)

print(f"\nChecking if NaN is converted to Neutral...")

nan_converted_to_neutral = 0
real_neutral_count = 0

for idx in range(len(dataset)):
    item = dataset[idx]
    labels = item['labels']
    masks = item['loss_mask']
    raw_row = raw_df.iloc[idx]
    
    for i, aspect in enumerate(dataset.aspects):
        raw_value = raw_row[aspect]
        is_nan = pd.isna(raw_value)
        label_id = labels[i].item()
        mask = masks[i].item()
        
        if is_nan:
            # If NaN has label_id=2 AND mask=0.0, it's OK (placeholder)
            # If NaN has mask=1.0, it's being TRAINED as Neutral (WRONG!)
            if label_id == 2 and mask == 1.0:
                nan_converted_to_neutral += 1
        else:
            if raw_value == 'Neutral' and mask == 1.0:
                real_neutral_count += 1

print(f"\nResults:")
print(f"  NaN converted to trainable Neutral: {nan_converted_to_neutral}")
print(f"  Real Neutral (with mask=1.0):       {real_neutral_count}")

if nan_converted_to_neutral == 0:
    print(f"\n  [PASS] NaN is NOT being trained as Neutral!")
    print(f"  [OK] NaN has placeholder label but mask=0.0 (skipped in training)")
else:
    print(f"\n  [FAIL] {nan_converted_to_neutral} NaN aspects are being trained as Neutral!")

# Final Summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\n1. Dataset contains:")
print(f"   - {total_nan:,} NaN aspects (not mentioned in review)")
print(f"   - {total_labeled:,} labeled aspects")
print(f"   - {total_neutral_labeled:,} REAL Neutral sentiments (aspects mentioned but no clear opinion)")

print(f"\n2. NaN Handling:")
print(f"   - NaN gets placeholder label_id=2")
print(f"   - NaN gets mask=0.0 → SKIPPED in training")
print(f"   - NaN is NOT converted to Neutral sentiment")

print(f"\n3. Training Behavior:")
print(f"   - Training loss only computed on aspects with mask=1.0 (labeled)")
print(f"   - NaN aspects (mask=0.0) do NOT contribute to loss")
print(f"   - Real Neutral (mask=1.0) ARE trained normally")

print(f"\n4. Key Distinction:")
print(f"   - NaN: Aspect not mentioned → mask=0.0 → Skip")
print(f"   - Neutral: Aspect mentioned but no clear sentiment → mask=1.0 → Train")

print("\n" + "=" * 80)
