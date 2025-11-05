# Changes Made for Multi-Seed Training

## Summary

Implemented a complete multi-seed training workflow for the multi-label ABSA model to enable statistically robust reporting. The system now:

1. Runs training multiple times with different seeds (100-105)
2. Saves all logs and results for each run
3. Calculates mean and standard deviation for all metrics
4. Generates comprehensive reports and visualizations

## Files Modified

### 1. `config_multi.yaml`

**Changed:**
```yaml
# OLD (Lines 137-139)
training_seed: 103
dataloader_seed: 103
# best is 100 100

# NEW (Lines 137-138)
training_seed: 100
# Note: dataloader_seed removed - only training_seed is used for reproducibility
```

**Reason:** 
- Removed `dataloader_seed` as requested - only `training_seed` controls reproducibility
- Reset default seed to 100 (first in the sequence)

## Files Created

### 1. `run_multiple_seeds.py` (Main Script)

**Purpose:** Orchestrates multi-seed training experiments

**Key Features:**
- Runs training with seeds: 100, 101, 102, 103, 104, 105
- Automatically updates config for each seed
- Backs up and restores original config
- Collects results from each run
- Aggregates metrics (mean ± std) across all runs
- Generates comprehensive reports

**Outputs:**
- `multi_label/results/multi_seed_experiments/seed_XXX/` - Individual run results
- `individual_runs_results.csv` - All metrics for each seed
- `aggregated_statistics.csv` - Mean ± Std for all metrics
- `experiment_report.txt` - Detailed report for paper writing
- Complete training logs for each run

### 2. `visualize_multi_seed_results.py` (Visualization Script)

**Purpose:** Creates publication-ready visualizations

**Generates 4 plots:**
1. **Overall metrics comparison** - Shows accuracy, F1, precision, recall with error bars
2. **Per-aspect F1 comparison** - Bar chart with mean ± std for all 11 aspects
3. **Metrics heatmap** - Complete view of all metrics across all seeds
4. **Variance analysis** - Identifies which metrics are most/least stable

### 3. `README_MULTI_SEED.md` (Documentation)

**Purpose:** Complete usage guide

**Sections:**
- Overview and motivation
- Usage instructions
- Output file descriptions
- Configuration options
- Time estimates
- Statistical reporting guidelines
- Troubleshooting

### 4. `CHANGES_MULTI_SEED.md` (This file)

**Purpose:** Documents all changes made

## Usage Example

### Step 1: Run Multi-Seed Training

```bash
# This will take ~12-24 hours for 6 runs
python multi_label/run_multiple_seeds.py
```

### Step 2: Generate Visualizations

```bash
python multi_label/visualize_multi_seed_results.py
```

### Step 3: Get Results for Paper

Open `multi_label/results/multi_seed_experiments/experiment_report.txt`

Example content:
```
================================================================================
OVERALL METRICS (Mean ± Std)
================================================================================

ACCURACY       : 86.45% ± 0.52%
F1             : 85.32% ± 0.61%
PRECISION      : 85.78% ± 0.48%
RECALL         : 85.89% ± 0.55%
```

## Key Design Decisions

### 1. Seeds Selection: 100-105

**Why these seeds?**
- Sequential and easy to remember
- Starting from 100 (your previous best seed)
- 6 runs provides good statistical power without excessive computation

### 2. Only `training_seed` Used

**As requested:**
- Removed `dataloader_seed` from config
- Data preprocessing seeds (split, oversampling, shuffle) remain fixed at 200
- Ensures all runs use same train/val/test split
- Only model initialization and training vary across runs

### 3. Comprehensive Logging

**Each run saves:**
- Training logs (timestamped)
- Training history (CSV and JSON)
- Test predictions (detailed CSV)
- Test results (JSON with all metrics)
- Best model checkpoint

### 4. Robust Error Handling

**Script features:**
- Backs up config before modifications
- Restores config even if runs fail
- Continues to next seed if one fails
- Saves intermediate results after each run

## Output Structure

```
multi_label/results/multi_seed_experiments/
│
├── seed_100/                    # Run 1
│   ├── best_model.pt
│   ├── checkpoint_epoch_*.pt
│   ├── test_results.json
│   ├── test_predictions_detailed.csv
│   ├── test_evaluation_summary.txt
│   ├── training_history.csv
│   ├── training_history.json
│   └── training_log_*.txt
│
├── seed_101/ ... seed_105/      # Runs 2-6 (same structure)
│
├── individual_runs_results.csv  # Raw data: all metrics for each seed
├── aggregated_statistics.csv    # Statistics: mean ± std for all metrics
├── experiment_report.txt         # Human-readable report
├── multi_seed_log_*.txt         # Complete experiment log
│
└── Visualizations:
    ├── overall_metrics_comparison.png
    ├── per_aspect_f1_comparison.png
    ├── metrics_heatmap.png
    └── variance_analysis.png
```

## Metrics Calculated

For **each metric** (accuracy, F1, precision, recall), the system calculates:

1. **Overall metrics** (averaged across all aspects)
2. **Per-aspect metrics** (for each of 11 aspects)
3. **Mean** across all 6 seeds
4. **Standard deviation** across all 6 seeds
5. **Coefficient of variation** (CV) for stability analysis

## Example Paper Reporting

```latex
\subsection{Multi-Label ABSA Results}

We trained our multi-label model with 6 different random seeds (100-105) 
to assess stability. The model achieved an overall F1 score of 
85.32\% $\pm$ 0.61\% (mean $\pm$ std), demonstrating consistent performance 
across different initializations. Per-aspect F1 scores ranged from 
78.45\% $\pm$ 1.23\% (Others) to 91.67\% $\pm$ 0.38\% (Camera), 
with coefficient of variation below 2\% for all aspects, indicating 
robust predictions.

\begin{table}[h]
\centering
\caption{Multi-label ABSA Results (Mean $\pm$ Std over 6 seeds)}
\begin{tabular}{lcccc}
\hline
Metric & Accuracy & F1 Score & Precision & Recall \\
\hline
Overall & 86.45±0.52 & 85.32±0.61 & 85.78±0.48 & 85.89±0.55 \\
\hline
\end{tabular}
\end{table}
```

## Testing

To test the workflow with fewer epochs:

1. Edit `config_multi.yaml`:
   ```yaml
   training:
     num_train_epochs: 2  # For quick testing
   ```

2. Run:
   ```bash
   python multi_label/run_multiple_seeds.py
   ```

3. Verify outputs are created correctly

4. Restore epochs:
   ```yaml
   training:
     num_train_epochs: 12  # Original value
   ```

## Notes

- **Time**: Each run takes 2-4 hours (12 epochs), so 6 runs = 12-24 hours total
- **Storage**: Each run uses ~500MB-2GB, total ~3-12GB
- **GPU**: Requires CUDA-capable GPU (tested on RTX 3070)
- **Reproducibility**: Data splits are identical across all runs (seed=200)
- **Statistical Power**: 6 runs is standard in ML research for reporting mean±std

## Verification Checklist

Before running full experiment:

- [ ] Config has `training_seed: 100` and no `dataloader_seed`
- [ ] GPU memory sufficient (check with `nvidia-smi`)
- [ ] Disk space available (~15GB recommended)
- [ ] Backup important data (script creates backups automatically)
- [ ] Test with 1-2 epochs first (optional)

## Support

If issues arise:
1. Check logs in `multi_label/results/multi_seed_experiments/multi_seed_log_*.txt`
2. Review individual seed logs in `seed_XXX/training_log_*.txt`
3. Verify config backup exists: `config_multi.yaml.backup`
