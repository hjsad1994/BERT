"""
Full Pipeline Script for Reproducible ABSA Training
====================================================
This script runs the complete pipeline with reproducible seeds:
1. Data preparation (split with seed)
2. Aspect-wise oversampling (with seed)
3. Model training (with seed)

Usage:
    python run_full_pipeline.py --config config_single.yaml
    
    # Or for multi-label:
    python run_full_pipeline.py --config ../multi_label/config_multi.yaml
"""

import argparse
import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*70)
    print(f"STEP: {description}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, shell=False)
    
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n[SUCCESS] {description} completed successfully")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run full ABSA pipeline with reproducible seeds'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Skip data preparation step (use existing train/val/test split)'
    )
    parser.add_argument(
        '--skip-oversample',
        action='store_true',
        help='Skip oversampling step (use original train.csv)'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip training step (only prepare data)'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("REPRODUCIBLE ABSA PIPELINE")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Skip prepare: {args.skip_prepare}")
    print(f"Skip oversample: {args.skip_oversample}")
    print(f"Skip train: {args.skip_train}")
    
    # Step 1: Data Preparation
    if not args.skip_prepare:
        run_command(
            ['python', 'prepare_data.py', '--config', args.config],
            "Data Preparation (train/val/test split with seed)"
        )
    else:
        print("\n[SKIPPED] Data preparation")
    
    # Step 2: Aspect-wise Oversampling
    if not args.skip_oversample:
        run_command(
            ['python', 'aspect_wise_oversampling.py', '--config', args.config],
            "Aspect-wise Oversampling (with seed)"
        )
    else:
        print("\n[SKIPPED] Oversampling")
    
    # Step 3: Training
    if not args.skip_train:
        run_command(
            ['python', 'train.py', '--config', args.config],
            "Model Training (with seed)"
        )
    else:
        print("\n[SKIPPED] Training")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nAll steps completed with reproducible seeds from config.")
    print("You can now compare results across different models/configurations.")


if __name__ == '__main__':
    main()
