# Quick Start: Config-Based Training

## ✅ What Changed?

**Before:**
```bash
# Hardcoded defaults, must specify everything
python train_multilabel_focal_contrastive.py \
    --focal-weight 0.7 \
    --contrastive-weight 0.3 \
    --temperature 0.1 \
    --epochs 8
```

**After:**
```bash
# All configs in YAML, run directly
python train_multilabel_focal_contrastive.py
```

---

## 🎯 Current Config Values

From `multi_label/config_multi.yaml`:

```yaml
multi_label:
  focal_weight: 0.8
  contrastive_weight: 0.2
  focal_gamma: 2.0
  contrastive_temperature: 0.1
  contrastive_base_weight: 0.1

training:
  num_train_epochs: 15
  
paths:
  output_dir: "multi_label/models/multilabel_focal_contrastive"
```

---

## 🚀 How to Use

### Option 1: Use Config (Recommended)

```bash
# Run with all settings from config
python multi_label\train_multilabel_focal_contrastive.py
```

### Option 2: Override Specific Values

```bash
# Change only what you need
python multi_label\train_multilabel_focal_contrastive.py \
    --focal-weight 0.9 \
    --epochs 20
```

### Option 3: Edit Config for Experiments

Edit `config_multi.yaml`:
```yaml
multi_label:
  focal_weight: 0.85  # Change this
  contrastive_weight: 0.15  # And this
```

Then run:
```bash
python multi_label\train_multilabel_focal_contrastive.py
```

---

## 📋 All Available Args

```bash
--config CONFIG              # Config file path (default: config_multi.yaml)
--epochs EPOCHS              # Override num_train_epochs
--output-dir OUTPUT_DIR      # Override output directory
--focal-weight WEIGHT        # Override focal_weight
--contrastive-weight WEIGHT  # Override contrastive_weight
--temperature TEMP           # Override contrastive_temperature
```

---

## ✅ Test It

```bash
# 1. Test config loading
python multi_label\test_config_loading.py

# 2. See help
python multi_label\train_multilabel_focal_contrastive.py --help

# 3. Run training (uses config)
python multi_label\train_multilabel_focal_contrastive.py
```

---

## 📝 Benefits

| Feature | Config-Based | Hardcoded |
|---------|-------------|-----------|
| Easy to reproduce | ✅ | ❌ |
| Version control | ✅ | ❌ |
| Experiment tracking | ✅ | ❌ |
| Quick experiments | ✅ (override) | ✅ |
| Team collaboration | ✅ | ❌ |

---

**See full docs:** `CONFIG_USAGE.md`
