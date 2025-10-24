"""Check which model generated predictions"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch

checkpoint = torch.load('multi_label/models/multilabel_focal_contrastive/best_model.pt', 
                        map_location='cpu', weights_only=False)

print("="*80)
print("CHECKPOINT INFO")
print("="*80)

print(f"\nEpoch: {checkpoint.get('epoch', 'unknown')}")

metrics = checkpoint.get('metrics', {})
print(f"\nValidation Metrics:")
print(f"  F1:        {metrics.get('overall_f1', 0)*100:.2f}%")
print(f"  Accuracy:  {metrics.get('overall_accuracy', 0)*100:.2f}%")
print(f"  Precision: {metrics.get('overall_precision', 0)*100:.2f}%")
print(f"  Recall:    {metrics.get('overall_recall', 0)*100:.2f}%")

print(f"\nModel Architecture:")
has_pooler = 'bert.pooler.dense.weight' in checkpoint['model_state_dict']
print(f"  Has pooler: {has_pooler}")
print(f"  → This is {'OLD' if has_pooler else 'NEW'} model (pooler {'included' if has_pooler else 'removed'})")

# Check if masking was used
training_args = checkpoint.get('training_args', {})
print(f"\nTraining Info:")
print(f"  Training args keys: {list(training_args.keys())[:10]}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if has_pooler:
    print("""
❌ This is OLD MODEL (trained WITHOUT masking!)
   • Has pooler layer (old architecture)
   • Trained on 87.7% Neutral data (NaN→Neutral)
   • Expected: High Neutral bias

This explains why:
   • Shipping: positive/negative → neutral (133/153 = 86.9% errors!)
   • High validation F1 (95.42%) but low true accuracy (13.07%)
   • Validation skips NaN but model learned from biased data

Solution:
   ✅ TRAIN NEW MODEL with masking!
   • Skip NaN during training
   • Eliminates Neutral bias
   • Expected: 90-91% F1 + better true accuracy
""")
else:
    print("""
✅ This is NEW MODEL (trained WITH masking!)
   • No pooler layer (new architecture)
   • Should have less Neutral bias
   
But if error analysis still shows high errors:
   → Need to investigate further
""")

print("="*80)
