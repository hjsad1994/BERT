# Quick Start: Multi-Seed Training

## Objective
Run multi-label ABSA training 6 times with different seeds (100-105) and get statistical metrics (mean ± std) for paper reporting.

## One-Command Quick Start

```bash
# Run all 6 training runs (takes 12-24 hours)
python multi_label/run_multiple_seeds.py

# After completion, generate visualizations
python multi_label/visualize_multi_seed_results.py
```

That's it! ✨

## What You Get

### Automatic Outputs

After running, you'll find everything in: `multi_label/results/multi_seed_experiments/`

**For Your Paper (Ready to Copy):**
1. `experiment_report.txt` - Complete report with mean±std for all metrics
2. `aggregated_statistics.csv` - Spreadsheet with all statistics
3. 4 publication-ready plots (PNG, 300 DPI)

**Example from report:**
```
Overall Metrics (Mean ± Std):
  ACCURACY : 86.45% ± 0.52%
  F1       : 85.32% ± 0.61%
  PRECISION: 85.78% ± 0.48%
  RECALL   : 85.89% ± 0.55%
```

### Individual Run Data

Each seed's complete results saved in:
- `seed_100/`, `seed_101/`, ..., `seed_105/`
- Contains: model checkpoints, logs, predictions, training history

## Time Estimate

- **Per seed**: 2-4 hours (12 epochs on RTX 3070)
- **Total**: 12-24 hours for all 6 seeds
- **Recommendation**: Start before bed or over weekend

## What Changed in Config

✅ **Done automatically - no manual changes needed!**

The script:
- Removed `dataloader_seed` (as requested)
- Uses only `training_seed` for model training
- Keeps data preprocessing seeds fixed (200)
- Backs up and restores config automatically

## Monitoring Progress

Check real-time progress:
```bash
# View overall experiment log
tail -f multi_label/results/multi_seed_experiments/multi_seed_log_*.txt

# View current seed's training log
tail -f multi_label/results/multi_seed_experiments/seed_10X/training_log_*.txt
```

## Quick Test (Optional)

Test the workflow quickly before full run:

```bash
# 1. Edit config for 2 epochs (instead of 12)
# In config_multi.yaml: num_train_epochs: 2

# 2. Run test
python multi_label/run_multiple_seeds.py

# 3. Check outputs are created
ls multi_label/results/multi_seed_experiments/

# 4. Restore epochs to 12 for real run
# In config_multi.yaml: num_train_epochs: 12
```

## Requirements

- **GPU**: CUDA-capable (RTX 3070 or better)
- **Disk Space**: ~15GB free
- **RAM**: 16GB+ recommended
- **Python**: 3.8+
- **Packages**: Already installed (from requirements.txt)

## Troubleshooting

### If a seed fails:
```bash
# Check the log
cat multi_label/results/multi_seed_experiments/seed_10X/training_log_*.txt

# Re-run that seed manually
python multi_label/train_multilabel.py \
  --config multi_label/config_multi.yaml \
  --output-dir multi_label/results/multi_seed_experiments/seed_10X
```

### Out of memory:
Edit `config_multi.yaml`:
```yaml
training:
  per_device_train_batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 8  # Increase from 4
```

## For Your Paper

Copy directly from `experiment_report.txt`:

```latex
We trained our multi-label model with 6 different random seeds 
(100-105). The model achieved an overall F1 score of 85.32% ± 0.61% 
(mean ± std), demonstrating consistent and robust performance.
```

## Full Documentation

For detailed information, see:
- `README_MULTI_SEED.md` - Complete usage guide
- `CHANGES_MULTI_SEED.md` - All changes made
- Training script: `train_multilabel.py`
- Multi-seed script: `run_multiple_seeds.py`

## Support

Files created:
- ✅ `run_multiple_seeds.py` - Main experiment script
- ✅ `visualize_multi_seed_results.py` - Visualization script
- ✅ `config_multi.yaml` - Updated (removed dataloader_seed)
- ✅ `README_MULTI_SEED.md` - Full documentation
- ✅ `CHANGES_MULTI_SEED.md` - Change log
- ✅ `QUICKSTART_MULTI_SEED.md` - This file

All ready to go! 🚀
