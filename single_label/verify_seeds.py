"""
Quick Script to Verify Seed Configuration
==========================================
This script verifies that all seeds are correctly configured
and accessible from the config file.
"""

import yaml
import sys
import os
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def verify_seeds(config_path):
    """Verify seed configuration"""
    
    print("\n" + "="*70)
    print("SEED CONFIGURATION VERIFICATION")
    print("="*70)
    
    # Load config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"\nConfig loaded: {config_path}")
    except Exception as e:
        print(f"\nFailed to load config: {e}")
        return False
    
    # Check reproducibility section exists
    if 'reproducibility' not in config:
        print("\nERROR: 'reproducibility' section not found in config")
        print("  Add reproducibility section to your config file")
        return False
    
    print("\nReproducibility section found")
    
    # Required seeds (master_seed removed - not used in code)
    required_seeds = [
        'data_split_seed',
        'oversampling_seed',
        'shuffle_seed',
        'training_seed',
        'dataloader_seed'
    ]
    
    # Optional seeds (for reference only)
    optional_seeds = ['master_seed']
    
    # Verify all seeds
    repro_config = config['reproducibility']
    all_ok = True
    
    print("\n" + "-"*70)
    print("Seed Values:")
    print("-"*70)
    
    for seed_name in required_seeds:
        if seed_name in repro_config:
            seed_value = repro_config[seed_name]
            print(f"  OK {seed_name:25} = {seed_value}")
        else:
            print(f"  X  {seed_name:25} = MISSING")
            all_ok = False
    
    # Check optional seeds
    for seed_name in optional_seeds:
        if seed_name in repro_config:
            seed_value = repro_config[seed_name]
            print(f"  OPT {seed_name:25} = {seed_value} (optional)")
    
    # Check paths section
    print("\n" + "-"*70)
    print("Data Paths:")
    print("-"*70)
    
    if 'paths' in config:
        paths = config['paths']
        for key in ['train_file', 'validation_file', 'test_file']:
            if key in paths:
                print(f"  OK {key:20} = {paths[key]}")
            else:
                print(f"  X  {key:20} = MISSING")
                all_ok = False
    else:
        print("  X  'paths' section not found")
        all_ok = False
    
    # Summary
    print("\n" + "="*70)
    if all_ok:
        print("VERIFICATION PASSED")
        print("="*70)
        print("\nAll seeds are configured correctly.")
        print("You can now run the pipeline:")
        print("\n  python run_full_pipeline.py --config", config_path)
        return True
    else:
        print("VERIFICATION FAILED")
        print("="*70)
        print("\nSome seeds or paths are missing.")
        print("Please check your config file.")
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'config_single.yaml'
    
    if not os.path.exists(config_path):
        print(f"\nConfig file not found: {config_path}")
        print("\nUsage: python verify_seeds.py [config_file]")
        sys.exit(1)
    
    success = verify_seeds(config_path)
    sys.exit(0 if success else 1)
