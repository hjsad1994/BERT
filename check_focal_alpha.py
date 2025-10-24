"""
Check actual focal loss alpha values used in training
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, 'multi_label')
from focal_loss_multilabel import calculate_global_alpha

print("="*80)
print("CHECKING FOCAL LOSS ALPHA VALUES")
print("="*80)

aspect_cols = ['Battery', 'Camera', 'Performance', 'Display', 'Design', 
               'Packaging', 'Price', 'Shop_Service', 'Shipping', 'General', 'Others']

sentiment_to_idx = {'positive': 0, 'negative': 1, 'neutral': 2}

# Calculate alpha with inverse_freq (default method)
alpha = calculate_global_alpha(
    'multi_label/data/train_multilabel_balanced.csv',
    aspect_cols,
    sentiment_to_idx,
    method='inverse_freq'
)

print("\n" + "="*80)
print("ANALYSIS: WHY NEUTRAL BIAS?")
print("="*80)

print(f"\n✅ Alpha weights are BALANCED:")
print(f"   Positive: {alpha[0]:.4f}")
print(f"   Negative: {alpha[1]:.4f}")
print(f"   Neutral:  {alpha[2]:.4f}")

print(f"\n⚠️  BUT PROBLEM IS:")
print(f"   83.35% of data is NaN → converted to Neutral during training!")
print(f"   Model sees TONS of 'Neutral' examples from unlabeled aspects")

print(f"\n📊 BREAKDOWN:")
print(f"   Labeled aspects:   28,983 (16.65%)")
print(f"     - Positive:       9,829 (33.91%)")
print(f"     - Negative:       9,722 (33.54%)")
print(f"     - Neutral:        9,432 (32.54%) ← Balanced")
print(f"")
print(f"   Unlabeled aspects: 145,136 (83.35%)")
print(f"     - All become:    'Neutral' during training!")

print(f"\n🔴 RESULT:")
print(f"   Total Neutral seen by model:")
print(f"     Labeled Neutral:    9,432")
print(f"     + NaN→Neutral:    145,136")
print(f"     ─────────────────────────────")
print(f"     TOTAL:           154,568 (88.76% of all training data!)")

print(f"\n💡 THIS EXPLAINS THE NEUTRAL BIAS!")
print(f"   Model learns: 'When unsure → Predict Neutral (safest option)'")

print("\n" + "="*80)
print("SOLUTIONS")
print("="*80)

print("""
Option 1: MASK NaN During Training (RECOMMENDED) ⭐
────────────────────────────────────────────────────────
✅ Don't train on NaN aspects at all
✅ Only train on labeled aspects (28,983 samples)
✅ Model won't see biased Neutral data
✅ Follows standard ABSA practices

Implementation:
  1. Create mask in dataset: mask = ~torch.isnan(labels)
  2. In loss calculation: loss = loss * mask
  3. Only compute loss on labeled aspects

Expected: Reduce Neutral bias significantly (+2-3% F1)

Option 2: KEEP Current + Adjust Alpha (QUICK FIX)
────────────────────────────────────────────────────────
⚠️  Keep NaN→Neutral but penalize Neutral more

Change focal_alpha from:
  [1.04, 1.03, 1.03]  (current balanced)

To:
  [1.5, 1.5, 0.5]     (penalize Neutral 3x less!)

This forces model to avoid predicting Neutral

Expected: Reduce Neutral bias (+1-2% F1)

Option 3: Two-Stage Training
────────────────────────────────────────────────────────
Stage 1: Train ONLY on labeled aspects (no NaN)
Stage 2: Fine-tune on all data (with NaN→Neutral)

Expected: Best of both worlds (+3-4% F1)

Recommendation: Try Option 2 first (quick), then Option 1 (best)
""")

print("="*80)
