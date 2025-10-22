"""
Quick test to verify config loading for focal/contrastive weights
"""
import yaml
import argparse

# Simulate training script behavior
def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Simulate command line args
class Args:
    def __init__(self):
        self.config = 'multi_label/config_multi.yaml'
        self.focal_weight = None  # No command line override
        self.contrastive_weight = None  # No command line override
        self.temperature = None

args = Args()

# Load config (same as training script)
config = load_config(args.config)
multi_label_config = config.get('multi_label', {})

# Get weights (same logic as training script)
focal_weight = args.focal_weight if args.focal_weight is not None else multi_label_config.get('focal_weight', 0.7)
contrastive_weight = args.contrastive_weight if args.contrastive_weight is not None else multi_label_config.get('contrastive_weight', 0.3)
temperature = args.temperature if args.temperature is not None else multi_label_config.get('contrastive_temperature', 0.1)
focal_gamma = multi_label_config.get('focal_gamma', 2.0)
contrastive_base_weight = multi_label_config.get('contrastive_base_weight', 0.1)

print("="*60)
print("CONFIG VALUES TEST")
print("="*60)
print(f"\nConfig file: {args.config}")
print(f"\nLoss settings (from config):")
print(f"   Focal weight:            {focal_weight}")
print(f"   Contrastive weight:      {contrastive_weight}")
print(f"   Focal gamma:             {focal_gamma}")
print(f"   Contrastive temperature: {temperature}")
print(f"   Contrastive base weight: {contrastive_base_weight}")

print(f"\nVerification:")
if focal_weight == 0.95 and contrastive_weight == 0.05:
    print("   Status: Config loaded correctly from config_multi.yaml")
    print("   Values: 0.95 (focal) / 0.05 (contrastive)")
elif focal_weight == 0.7 and contrastive_weight == 0.3:
    print("   WARNING: Using default values (0.7/0.3)")
    print("   Config may not be loading properly")
else:
    print(f"   Using custom values: {focal_weight}/{contrastive_weight}")

print("\n" + "="*60)
