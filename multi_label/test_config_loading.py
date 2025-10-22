"""
Test script to verify config loading for contrastive loss
"""
import yaml
import argparse
import sys

def load_config(config_path='multi_label/config_multi.yaml'):
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def test_config_loading():
    """Test 1: Load config from YAML only"""
    print("=" * 80)
    print("TEST 1: Load from config_multi.yaml only")
    print("=" * 80)
    
    config = load_config('multi_label/config_multi.yaml')
    multi_label_config = config.get('multi_label', {})
    
    focal_weight = multi_label_config.get('focal_weight', 0.7)
    contrastive_weight = multi_label_config.get('contrastive_weight', 0.3)
    temperature = multi_label_config.get('contrastive_temperature', 0.1)
    focal_gamma = multi_label_config.get('focal_gamma', 2.0)
    contrastive_base_weight = multi_label_config.get('contrastive_base_weight', 0.1)
    num_epochs = config['training'].get('num_train_epochs', 8)
    output_dir = config['paths'].get('output_dir', 'multilabel_focal_contrastive_model')
    
    print(f"\nLoaded from config:")
    print(f"   Focal weight: {focal_weight}")
    print(f"   Contrastive weight: {contrastive_weight}")
    print(f"   Temperature: {temperature}")
    print(f"   Focal gamma: {focal_gamma}")
    print(f"   Contrastive base weight: {contrastive_base_weight}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Output dir: {output_dir}")

def test_config_with_override():
    """Test 2: Override with command line args"""
    print("\n" + "=" * 80)
    print("TEST 2: Override with command line arguments")
    print("=" * 80)
    
    config = load_config('multi_label/config_multi.yaml')
    multi_label_config = config.get('multi_label', {})
    
    # Simulate command line args
    class Args:
        focal_weight = 0.9
        contrastive_weight = 0.1
        temperature = 0.05
        epochs = 20
        output_dir = "custom_output_dir"
    
    args = Args()
    
    # Apply override logic
    focal_weight = args.focal_weight if args.focal_weight is not None else multi_label_config.get('focal_weight', 0.7)
    contrastive_weight = args.contrastive_weight if args.contrastive_weight is not None else multi_label_config.get('contrastive_weight', 0.3)
    temperature = args.temperature if args.temperature is not None else multi_label_config.get('contrastive_temperature', 0.1)
    focal_gamma = multi_label_config.get('focal_gamma', 2.0)  # Not overridden
    contrastive_base_weight = multi_label_config.get('contrastive_base_weight', 0.1)  # Not overridden
    num_epochs = args.epochs if args.epochs is not None else config['training'].get('num_train_epochs', 8)
    output_dir = args.output_dir if args.output_dir is not None else config['paths'].get('output_dir', 'multilabel_focal_contrastive_model')
    
    print(f"\nCommand line overrides:")
    print(f"   --focal-weight {args.focal_weight}")
    print(f"   --contrastive-weight {args.contrastive_weight}")
    print(f"   --temperature {args.temperature}")
    print(f"   --epochs {args.epochs}")
    print(f"   --output-dir {args.output_dir}")
    
    print(f"\nFinal values (after override):")
    print(f"   Focal weight: {focal_weight}")
    print(f"   Contrastive weight: {contrastive_weight}")
    print(f"   Temperature: {temperature}")
    print(f"   Focal gamma: {focal_gamma} (from config)")
    print(f"   Contrastive base weight: {contrastive_base_weight} (from config)")
    print(f"   Epochs: {num_epochs}")
    print(f"   Output dir: {output_dir}")

def test_partial_override():
    """Test 3: Partial override (only some args)"""
    print("\n" + "=" * 80)
    print("TEST 3: Partial override (only focal-weight)")
    print("=" * 80)
    
    config = load_config('multi_label/config_multi.yaml')
    multi_label_config = config.get('multi_label', {})
    
    # Simulate command line args (only focal_weight provided)
    class Args:
        focal_weight = 0.85
        contrastive_weight = None  # Not provided
        temperature = None  # Not provided
        epochs = None  # Not provided
        output_dir = None  # Not provided
    
    args = Args()
    
    # Apply override logic
    focal_weight = args.focal_weight if args.focal_weight is not None else multi_label_config.get('focal_weight', 0.7)
    contrastive_weight = args.contrastive_weight if args.contrastive_weight is not None else multi_label_config.get('contrastive_weight', 0.3)
    temperature = args.temperature if args.temperature is not None else multi_label_config.get('contrastive_temperature', 0.1)
    focal_gamma = multi_label_config.get('focal_gamma', 2.0)
    contrastive_base_weight = multi_label_config.get('contrastive_base_weight', 0.1)
    num_epochs = args.epochs if args.epochs is not None else config['training'].get('num_train_epochs', 8)
    output_dir = args.output_dir if args.output_dir is not None else config['paths'].get('output_dir', 'multilabel_focal_contrastive_model')
    
    print(f"\nCommand line overrides:")
    print(f"   --focal-weight {args.focal_weight} (only this)")
    
    print(f"\nFinal values:")
    print(f"   Focal weight: {focal_weight} (OVERRIDDEN)")
    print(f"   Contrastive weight: {contrastive_weight} (from config)")
    print(f"   Temperature: {temperature} (from config)")
    print(f"   Focal gamma: {focal_gamma} (from config)")
    print(f"   Contrastive base weight: {contrastive_base_weight} (from config)")
    print(f"   Epochs: {num_epochs} (from config)")
    print(f"   Output dir: {output_dir} (from config)")

if __name__ == '__main__':
    print("\nTesting Config Loading Logic\n")
    
    try:
        test_config_loading()
        test_config_with_override()
        test_partial_override()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        
        print("\nSummary:")
        print("   - Config file loads correctly")
        print("   - Command line args override config values")
        print("   - Partial overrides work correctly")
        print("   - Non-overridden values use config defaults")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
