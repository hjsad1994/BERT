"""
Run all tests for BiLSTM Aspect Detection
"""

import sys
import subprocess


def run_command(command, description):
    """Run a command and print results"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {command}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode == 0:
        print(f"\n✓ {description} - PASSED")
    else:
        print(f"\n✗ {description} - FAILED")
        return False
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("BiLSTM Aspect Detection - Test Suite")
    print("="*80)
    
    tests = [
        ("python model_bilstm_ad.py", "Test 1: Model Architecture"),
        ("python dataset_bilstm_ad.py", "Test 2: Dataset Loader"),
    ]
    
    all_passed = True
    
    for command, description in tests:
        if not run_command(command, description):
            all_passed = False
            break
    
    print(f"\n{'='*80}")
    if all_passed:
        print("✓ All tests PASSED!")
        print("\nYou can now train the model:")
        print("  python train_bilstm_ad.py --config config_bilstm_ad.yaml")
    else:
        print("✗ Some tests FAILED!")
        print("Please fix the errors before training.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
