import pandas as pd
import json

print("="*80)
print("COMPARING PRICE PERFORMANCE: STL vs MTL")
print("="*80)

# Load STL results
with open('VisoBERT-STL/models/sentiment_classification/test_results.json', 'r') as f:
    stl_results = json.load(f)

# Load MTL results  
with open('VisoBERT-MTL/models/mtl/test_results.json', 'r') as f:
    mtl_results = json.load(f)

# Get Price metrics
stl_price = stl_results['per_aspect']['Price']
mtl_price_sc = mtl_results['sc']['per_aspect']['Price']
mtl_price_ad = mtl_results['ad']['per_aspect']['Price']

print("\n" + "-"*80)
print("STAGE 2: SENTIMENT CLASSIFICATION")
print("-"*80)

print(f"\nSTL Price (SC):")
print(f"  Accuracy:  {stl_price['accuracy']*100:.2f}%")
print(f"  F1 Score:  {stl_price['f1']*100:.2f}%")
print(f"  Precision: {stl_price['precision']*100:.2f}%")
print(f"  Recall:    {stl_price['recall']*100:.2f}%")

print(f"\nMTL Price (SC):")
print(f"  Accuracy:  {mtl_price_sc['accuracy']*100:.2f}%")
print(f"  F1 Score:  {mtl_price_sc['f1']*100:.2f}%")
print(f"  Precision: {mtl_price_sc['precision']*100:.2f}%")
print(f"  Recall:    {mtl_price_sc['recall']*100:.2f}%")

print("\n" + "-"*80)
print("COMPARISON:")
print("-"*80)
print(f"  Accuracy:  STL {stl_price['accuracy']*100:.2f}% vs MTL {mtl_price_sc['accuracy']*100:.2f}% (Diff: {(stl_price['accuracy']-mtl_price_sc['accuracy'])*100:+.2f}%)")
print(f"  F1 Score:  STL {stl_price['f1']*100:.2f}% vs MTL {mtl_price_sc['f1']*100:.2f}% (Diff: {(stl_price['f1']-mtl_price_sc['f1'])*100:+.2f}%)")
print(f"  Precision: STL {stl_price['precision']*100:.2f}% vs MTL {mtl_price_sc['precision']*100:.2f}% (Diff: {(stl_price['precision']-mtl_price_sc['precision'])*100:+.2f}%)")
print(f"  Recall:    STL {stl_price['recall']*100:.2f}% vs MTL {mtl_price_sc['recall']*100:.2f}% (Diff: {(stl_price['recall']-mtl_price_sc['recall'])*100:+.2f}%)")

print("\n" + "-"*80)
print("STAGE 1: ASPECT DETECTION")
print("-"*80)

# Load STL AD results
with open('VisoBERT-STL/models/aspect_detection/test_results.json', 'r') as f:
    stl_ad_results = json.load(f)

stl_price_ad = stl_ad_results['per_aspect']['Price']

print(f"\nSTL Price (AD):")
print(f"  Accuracy:  {stl_price_ad['accuracy']*100:.2f}%")
print(f"  F1 Score:  {stl_price_ad['f1']*100:.2f}%")
print(f"  Precision: {stl_price_ad['precision']*100:.2f}%")
print(f"  Recall:    {stl_price_ad['recall']*100:.2f}%")

print(f"\nMTL Price (AD):")
print(f"  Accuracy:  {mtl_price_ad['accuracy']*100:.2f}%")
print(f"  F1 Score:  {mtl_price_ad['f1']*100:.2f}%")
print(f"  Precision: {mtl_price_ad['precision']*100:.2f}%")
print(f"  Recall:    {mtl_price_ad['recall']*100:.2f}%")

print("\n" + "-"*80)
print("AD COMPARISON:")
print("-"*80)
print(f"  Accuracy:  STL {stl_price_ad['accuracy']*100:.2f}% vs MTL {mtl_price_ad['accuracy']*100:.2f}% (Diff: {(stl_price_ad['accuracy']-mtl_price_ad['accuracy'])*100:+.2f}%)")
print(f"  F1 Score:  STL {stl_price_ad['f1']*100:.2f}% vs MTL {mtl_price_ad['f1']*100:.2f}% (Diff: {(stl_price_ad['f1']-mtl_price_ad['f1'])*100:+.2f}%)")
print(f"  Precision: STL {stl_price_ad['precision']*100:.2f}% vs MTL {mtl_price_ad['precision']*100:.2f}% (Diff: {(stl_price_ad['precision']-mtl_price_ad['precision'])*100:+.2f}%)")
print(f"  Recall:    STL {stl_price_ad['recall']*100:.2f}% vs MTL {mtl_price_ad['recall']*100:.2f}% (Diff: {(stl_price_ad['recall']-mtl_price_ad['recall'])*100:+.2f}%)")

print("\n" + "="*80)
print("ERROR ANALYSIS")
print("="*80)

# Count Price errors in MTL
mtl_errors = pd.read_csv('VisoBERT-MTL/error_analysis_results/all_errors_detailed.csv', encoding='utf-8-sig')
price_errors_mtl = mtl_errors[mtl_errors['aspect'] == 'Price']

print(f"\nMTL Price errors: {len(price_errors_mtl)} out of {mtl_price_sc['n_samples']} samples")
print(f"MTL Price error rate: {len(price_errors_mtl)/mtl_price_sc['n_samples']*100:.2f}%")

print("\nMTL Price error breakdown:")
for confusion_type, count in price_errors_mtl['confusion_type'].value_counts().items():
    print(f"  {confusion_type}: {count} samples")

# Count Price errors in STL
stl_predictions = pd.read_csv('VisoBERT-STL/models/sentiment_classification/test_predictions_detailed.csv')
price_stl_errors = stl_predictions[stl_predictions['Price_correct'] == 0]

print(f"\nSTL Price errors: {len(price_stl_errors)} out of {len(stl_predictions)} samples")
print(f"STL Price error rate: {len(price_stl_errors)/len(stl_predictions)*100:.2f}%")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)

print("\n**Why STL Price SC has lower recall than MTL:**")
print(f"  - STL Recall: {stl_price['recall']*100:.2f}%")
print(f"  - MTL Recall: {mtl_price_sc['recall']*100:.2f}%")
print(f"  - Difference: {(mtl_price_sc['recall']-stl_price['recall'])*100:.2f}% better in MTL")

print("\n**But STL has better precision:**")
print(f"  - STL Precision: {stl_price['precision']*100:.2f}%")
print(f"  - MTL Precision: {mtl_price_sc['precision']*100:.2f}%")
print(f"  - Difference: {(stl_price['precision']-mtl_price_sc['precision'])*100:.2f}% better in STL")

print("\n**Overall F1 (what matters):**")
print(f"  - STL F1: {stl_price['f1']*100:.2f}%")
print(f"  - MTL F1: {mtl_price_sc['f1']*100:.2f}%")
print(f"  - Difference: {(stl_price['f1']-mtl_price_sc['f1'])*100:.2f}% {'better in STL' if stl_price['f1'] > mtl_price_sc['f1'] else 'better in MTL'}")
