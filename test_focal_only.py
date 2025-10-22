"""
Test Focal Loss ONLY configuration (no contrastive)
Verify config is set correctly
"""

import yaml
import sys

print("=" * 80)
print("Testing Focal Loss Configuration (No Contrastive)")
print("=" * 80)

# Load config
config_path = 'multi_label/config_focal_only.yaml'
print(f"\nLoading config: {config_path}")

try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"ERROR: Cannot load config: {e}")
    sys.exit(1)

# Get multi_label settings
ml_config = config.get('multi_label', {})

print(f"\n{'='*80}")
print("Multi-Label Settings")
print(f"{'='*80}")

# Check loss type
loss_type = ml_config.get('loss_type', 'unknown')
print(f"\nLoss Type: {loss_type}")
if loss_type != 'focal':
    print("  WARNING: Expected 'focal'")

# Check weights
focal_weight = ml_config.get('focal_weight', 0)
contrastive_weight = ml_config.get('contrastive_weight', 0)

print(f"\nLoss Weights:")
print(f"  Focal:       {focal_weight}")
print(f"  Contrastive: {contrastive_weight}")

# Focal settings
focal_gamma = ml_config.get('focal_gamma', 0)
focal_alpha = ml_config.get('focal_alpha', 'none')

print(f"\nFocal Loss Settings:")
print(f"  Gamma: {focal_gamma}")
print(f"  Alpha: {focal_alpha}")

# Verify
print(f"\n{'='*80}")
print("Verification")
print(f"{'='*80}")

all_ok = True

if focal_weight == 1.0:
    print("\n[OK] Focal weight is 1.0 (100%)")
else:
    print(f"\n[WARN] Focal weight is {focal_weight} (expected 1.0)")
    all_ok = False

if contrastive_weight == 0.0:
    print("[OK] Contrastive weight is 0.0 (disabled)")
else:
    print(f"[WARN] Contrastive weight is {contrastive_weight} (expected 0.0)")
    all_ok = False

use_contrastive = ml_config.get('use_contrastive', True)
if not use_contrastive:
    print("[OK] use_contrastive is False (disabled)")
else:
    print("[WARN] use_contrastive is True (should be False)")
    all_ok = False

print(f"\n[OK] Focal gamma: {focal_gamma}")
print(f"[OK] Focal alpha: {focal_alpha}")

# Check output dir
output_dir = config['paths'].get('output_dir', '')
print(f"\n[OK] Output dir: {output_dir}")

# Check learning rate
learning_rate = config['training'].get('learning_rate', 0)
print(f"[OK] Learning rate: {learning_rate}")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

if all_ok:
    print("\n[SUCCESS] Config is set correctly for Focal Loss ONLY")
    print("\nSettings:")
    print(f"  - Loss: Focal (100%)")
    print(f"  - Contrastive: DISABLED (0%)")
    print(f"  - Focal gamma: {focal_gamma}")
    print(f"  - Focal alpha: {focal_alpha}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Output: {output_dir}")
    
    print(f"\nExpected results:")
    print(f"  - This is the baseline")
    print(f"  - Compare with:")
    print(f"    * Focal+Contrastive: 95.99% F1")
    print(f"    * GHM only: 96.0-96.5% F1")
    print(f"    * GHM+Contrastive: 96.5-97% F1")
    
    print(f"\nNext step:")
    print(f"  python multi_label\\train_multilabel_focal_contrastive.py --config multi_label/config_focal_only.yaml --epochs 15")
    
    print(f"\n{'='*80}")
    sys.exit(0)
else:
    print("\n[ERROR] Config has warnings. Please check above.")
    print("\nTo fix, edit multi_label/config_focal_only.yaml:")
    print("  focal_weight: 1.0")
    print("  contrastive_weight: 0.0")
    print("  use_contrastive: false")
    
    print(f"\n{'='*80}")
    sys.exit(1)
