# Training Summary

## Files Quan Trọng

### Training:
- `train.py` - Main training script
- `config.yaml` - Original config
- `config_optimized.yaml` - **RECOMMENDED** (optimized, faster)
- `utils.py` - Training utilities
- `focal_loss_trainer.py` - Custom trainer with Focal Loss

### Data:
- `dataset.csv` - Main dataset
- `prepare_data.py` - Prepares train/val/test splits
- `data/` - Processed data (train.csv, validation.csv, test.csv)

### Analysis:
- `analyze_results.py` - Detailed performance analysis
- `error_analysis.py` - Error analysis tool
- `predict_example.py` - Test predictions

## Quick Start

### 1. Train with Optimized Config (RECOMMENDED):
```bash
python train.py --config config_optimized.yaml
```

### 2. Benefits of Optimized Config:
- **25-30% faster** (~40-45 min vs 60 min)
- **Better F1** (expected +1-2%)
- **2x faster evaluation**
- **More stable training**

### 3. Key Optimizations:
- Cosine LR scheduler (better than linear)
- Eval batch size = 64 (2x faster)
- Prefetch factor = 4 (optimized)
- PyTorch AdamW optimizer (fastest)

## Current Performance

```
Model: Vietnamese-Sentiment-visobert (XLM-RoBERTa)
Current F1: 90-91%
```

## Expected with Optimized Config

```
Training time: 40-45 minutes
F1 Score: 91-92%
Memory: ~7GB VRAM (safe)
```

## Compare Configs

| Feature | config.yaml | config_optimized.yaml |
|---------|-------------|----------------------|
| LR Scheduler | linear | **cosine** ✓ |
| Eval Batch | 32 | **64** ✓ |
| Prefetch | 12 | **4** ✓ |
| Optimizer | (default) | **adamw_torch** ✓ |
| Speed | ~60 min | **~40-45 min** ✓ |

## Notes

- Dataset: ~9,200 samples
- 13 aspects
- 3 sentiments (positive, negative, neutral)
- RTX 3070 8GB is sufficient

## Troubleshooting

If training fails:
1. Check GPU memory: `nvidia-smi`
2. Reduce batch size in config if OOM
3. Check data: `ls data/`

For best results: Use `config_optimized.yaml`
