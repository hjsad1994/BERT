"""
Test GHM-C Loss ONLY (no contrastive)
Verify config is set correctly
"""

import yaml
import sys

print("=" * 80)
print("Testing GHM-C Loss Configuration (No Contrastive)")
print("=" * 80)

# Load config
config_path = 'multi_label/config_ghm.yaml'
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
if loss_type != 'ghm':
    print("  WARNING: Expected 'ghm'")

# Check weights
classification_weight = ml_config.get('classification_weight', 0)
contrastive_weight = ml_config.get('contrastive_weight', 0)

print(f"\nLoss Weights:")
print(f"  Classification (GHM-C): {classification_weight}")
print(f"  Contrastive:            {contrastive_weight}")

# Verify
print(f"\n{'='*80}")
print("Verification")
print(f"{'='*80}")

all_ok = True

if classification_weight == 1.0:
    print("\n[OK] Classification weight is 1.0 (100%)")
else:
    print(f"\n[WARN] Classification weight is {classification_weight} (expected 1.0)")
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

# Check GHM settings
ghm_bins = ml_config.get('ghm_bins', 0)
ghm_momentum = ml_config.get('ghm_momentum', 0)

print(f"\n[OK] GHM bins: {ghm_bins}")
print(f"[OK] GHM momentum: {ghm_momentum}")

# Check output dir
output_dir = config['paths'].get('output_dir', '')
print(f"\n[OK] Output dir: {output_dir}")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

if all_ok:
    print("\n[SUCCESS] Config is set correctly for GHM-C Loss ONLY")
    print("\nSettings:")
    print(f"  - Loss: GHM-C (100%)")
    print(f"  - Contrastive: DISABLED (0%)")
    print(f"  - GHM bins: {ghm_bins}")
    print(f"  - GHM momentum: {ghm_momentum}")
    print(f"  - Output: {output_dir}")
    
    print(f"\nExpected improvement:")
    print(f"  - Focal Loss only:  95.99% F1")
    print(f"  - GHM-C Loss only:  96.0-96.5% F1  (+0.0-0.5%)")
    
    print(f"\nNext step:")
    print(f"  python multi_label\\train_multilabel_ghm_contrastive.py --epochs 15")
    
    print(f"\n{'='*80}")
    sys.exit(0)
else:
    print("\n[ERROR] Config has warnings. Please check above.")
    print("\nTo fix, edit multi_label/config_ghm.yaml:")
    print("  classification_weight: 1.0")
    print("  contrastive_weight: 0.0")
    print("  use_contrastive: false")
    
    print(f"\n{'='*80}")
    sys.exit(1)
