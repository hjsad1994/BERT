# Multi-Seed Training Experiments

This directory contains scripts for running multi-label ABSA training with multiple random seeds and aggregating the results for scientific reporting.

## Overview

Training neural networks with different random seeds helps assess model stability and provides statistically meaningful results. This approach runs the same training configuration multiple times with different seeds and calculates mean ± standard deviation for all metrics.

## Files

- **`run_multiple_seeds.py`**: Main script to run training with multiple seeds (100-105)
- **`visualize_multi_seed_results.py`**: Visualization script for aggregated results
- **`config_multi.yaml`**: Configuration file (only `training_seed` is used, `dataloader_seed` removed)

## Usage

### 1. Run Multi-Seed Training

This will run training 6 times with seeds: 100, 101, 102, 103, 104, 105

```bash
python multi_label/run_multiple_seeds.py
```

**What it does:**
- Runs training for each seed sequentially
- Saves each run to `multi_label/results/multi_seed_experiments/seed_XXX/`
- Collects all training logs, test results, and metrics
- Aggregates results across all runs
- Calculates mean and standard deviation for all metrics

**Output directory:** `multi_label/results/multi_seed_experiments/`

### 2. Visualize Results

After training completes, visualize the aggregated results:

```bash
python multi_label/visualize_multi_seed_results.py
```

Or specify custom results directory:

```bash
python multi_label/visualize_multi_seed_results.py --results-dir multi_label/results/multi_seed_experiments
```

## Output Files

### Aggregated Results Directory Structure

```
multi_label/results/multi_seed_experiments/
├── seed_100/                           # Results for seed 100
│   ├── best_model.pt
│   ├── test_results.json
│   ├── training_history.csv
│   ├── training_log_YYYYMMDD_HHMMSS.txt
│   └── ...
├── seed_101/                           # Results for seed 101
│   └── ...
├── ...
├── seed_105/                           # Results for seed 105
│   └── ...
├── individual_runs_results.csv         # All metrics for each seed
├── aggregated_statistics.csv           # Mean ± Std for all metrics
├── experiment_report.txt               # Detailed text report (for paper)
├── multi_seed_log_YYYYMMDD_HHMMSS.txt # Complete experiment log
├── overall_metrics_comparison.png      # Overall metrics visualization
├── per_aspect_f1_comparison.png        # Per-aspect F1 scores
├── metrics_heatmap.png                 # Heatmap of all metrics
└── variance_analysis.png               # Coefficient of variation analysis
```

### Key Output Files for Reporting

1. **`experiment_report.txt`**: Comprehensive report with:
   - Overall metrics (mean ± std) for accuracy, F1, precision, recall
   - Per-aspect metrics (mean ± std) for all 11 aspects
   - Individual run summaries

2. **`aggregated_statistics.csv`**: Spreadsheet-ready statistics:
   - Mean and std for every metric
   - Easy to import into papers/presentations

3. **`individual_runs_results.csv`**: Raw data:
   - Complete metrics for each seed
   - Useful for additional analysis

## Configuration

### Seeds

To change the seeds used for experiments, edit `run_multiple_seeds.py`:

```python
# Line 14
SEEDS = [100, 101, 102, 103, 104, 105]  # Modify as needed
```

### Training Configuration

All training parameters are in `config_multi.yaml`. The multi-seed script only modifies the `training_seed` field:

```yaml
reproducibility:
  training_seed: 100  # Will be updated automatically for each run
  # Note: dataloader_seed removed - only training_seed is used
```

**Important:** The script automatically:
1. Backs up the original config
2. Updates `training_seed` for each run
3. Restores the original config after completion

## Example Report Output

```
================================================================================
OVERALL METRICS (Mean ± Std)
================================================================================

ACCURACY       : 86.45% ± 0.52%
F1             : 85.32% ± 0.61%
PRECISION      : 85.78% ± 0.48%
RECALL         : 85.89% ± 0.55%

================================================================================
PER-ASPECT METRICS (Mean ± Std)
================================================================================

Aspect          Accuracy             F1 Score             Precision            Recall              
-----------------------------------------------------------------------------------------------
Battery         87.23±0.45%  86.45±0.52%  86.78±0.48%  86.12±0.58%
Camera          88.56±0.38%  87.89±0.42%  88.12±0.39%  87.67±0.45%
Performance     85.67±0.62%  84.89±0.68%  85.23±0.61%  84.56±0.71%
...
```

## Time Estimates

- **Single run**: ~2-4 hours (depends on epochs, batch size, GPU)
- **6 runs**: ~12-24 hours total
- **Recommendation**: Run overnight or over weekend

## Tips

1. **Monitor Progress**: Check `multi_seed_log_*.txt` for real-time progress
2. **Resume Failed Runs**: If a run fails, you can manually run it:
   ```bash
   python multi_label/train_multilabel.py --config multi_label/config_multi.yaml --output-dir multi_label/results/multi_seed_experiments/seed_XXX
   ```
3. **Disk Space**: Each run takes ~500MB-2GB. Ensure sufficient space.
4. **GPU Memory**: Monitor with `nvidia-smi` to avoid OOM errors

## Statistical Reporting

When reporting in papers, use this format:

```
Our multi-label model achieved an overall F1 score of 85.32% ± 0.61% 
(mean ± std over 6 runs with seeds 100-105), demonstrating consistent 
performance across different random initializations.
```

## Troubleshooting

### Issue: Training fails for a seed
- Check the specific seed's log file: `seed_XXX/training_log_*.txt`
- Run that seed manually to debug

### Issue: Out of memory
- Reduce batch size in `config_multi.yaml`
- Enable gradient checkpointing

### Issue: Results not aggregating
- Ensure all runs completed successfully
- Check that `test_results.json` exists in each `seed_XXX/` directory

## Notes

- **Reproducibility**: Only `training_seed` affects model training. Data preprocessing uses fixed seeds (200) for consistency.
- **Best Model**: Each run saves its best model based on validation F1 score
- **Fairness**: All seeds use the same train/val/test split (fixed at preprocessing)
