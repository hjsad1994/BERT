"""Check when models were trained"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import torch
from datetime import datetime
import glob

print("="*80)
print("CHECKING MODEL FILES AND TIMESTAMPS")
print("="*80)

model_dir = 'multi_label/models/multilabel_focal_contrastive'

# List all .pt files
pt_files = glob.glob(os.path.join(model_dir, '*.pt'))

if not pt_files:
    print(f"\n❌ No .pt files found in {model_dir}")
else:
    print(f"\nFound {len(pt_files)} checkpoint files:")
    print(f"\n{'File':<30} {'Modified Time':<25} {'Size (MB)':<12} {'Has Pooler'}")
    print("-"*85)
    
    for pt_file in sorted(pt_files):
        filename = os.path.basename(pt_file)
        
        # Get file info
        stat = os.stat(pt_file)
        size_mb = stat.st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        
        # Load checkpoint to check pooler
        try:
            checkpoint = torch.load(pt_file, map_location='cpu', weights_only=False)
            has_pooler = 'bert.pooler.dense.weight' in checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', '?')
            
            pooler_status = "✅ YES (OLD)" if has_pooler else "❌ NO (NEW)"
            
            print(f"{filename:<30} {mod_time:%Y-%m-%d %H:%M:%S}   {size_mb:>8.1f} MB   {pooler_status}")
            
            # Show more details for best_model.pt
            if filename == 'best_model.pt':
                metrics = checkpoint.get('metrics', {})
                print(f"  → Epoch {epoch}, Val F1: {metrics.get('overall_f1', 0)*100:.2f}%")
                
        except Exception as e:
            print(f"{filename:<30} {mod_time:%Y-%m-%d %H:%M:%S}   {size_mb:>8.1f} MB   ERROR: {e}")

# Check predictions file timestamp
print("\n" + "="*80)
print("CHECKING PREDICTIONS FILE")
print("="*80)

pred_file = 'multi_label/results/test_predictions_multi.csv'
if os.path.exists(pred_file):
    stat = os.stat(pred_file)
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    print(f"\nPredictions file: {pred_file}")
    print(f"  Last modified: {mod_time:%Y-%m-%d %H:%M:%S}")
    print(f"  Size: {stat.st_size / 1024:.1f} KB")
else:
    print(f"\n❌ Predictions file not found: {pred_file}")

# Check error analysis timestamp
error_file = 'multi_label/error_analysis_results/aspect_error_analysis.csv'
if os.path.exists(error_file):
    stat = os.stat(error_file)
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    print(f"\nError analysis file: {error_file}")
    print(f"  Last modified: {mod_time:%Y-%m-%d %H:%M:%S}")
else:
    print(f"\n❌ Error analysis file not found")

# Current time
print(f"\n" + "="*80)
print(f"Current time: {datetime.now():%Y-%m-%d %H:%M:%S}")
print("="*80)

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if pt_files:
    best_model = os.path.join(model_dir, 'best_model.pt')
    if os.path.exists(best_model):
        checkpoint = torch.load(best_model, map_location='cpu', weights_only=False)
        has_pooler = 'bert.pooler.dense.weight' in checkpoint['model_state_dict']
        
        stat = os.stat(best_model)
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        age_hours = (datetime.now() - mod_time).total_seconds() / 3600
        
        print(f"\nCurrent best_model.pt:")
        print(f"  Last modified: {mod_time:%Y-%m-%d %H:%M:%S} ({age_hours:.1f} hours ago)")
        print(f"  Has pooler: {'YES' if has_pooler else 'NO'}")
        print(f"  Type: {'OLD MODEL (trained WITHOUT masking)' if has_pooler else 'NEW MODEL (trained WITH masking)'}")
        
        if has_pooler:
            print(f"\n❌ THIS IS OLD MODEL!")
            print(f"   • Trained before masking implementation")
            print(f"   • Has Neutral bias (87.7% training data)")
            print(f"   • This explains low error analysis accuracy!")
            print(f"\n✅ SOLUTION: Train new model with masking")
        else:
            print(f"\n✅ THIS IS NEW MODEL!")
            print(f"   • Trained with masking")
            print(f"   • Should have less Neutral bias")
            print(f"   • If errors still high, need investigation")

print("\n" + "="*80)
