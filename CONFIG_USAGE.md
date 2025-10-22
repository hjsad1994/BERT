# Config Management - Contrastive Loss Settings

## Summary

**DONE!** ✅ All contrastive loss configs now load from `config_multi.yaml` by default, with command line override support.

---

## 📋 How It Works

### 1. Default: Load from Config File

```bash
# Run with all settings from config_multi.yaml
python multi_label\train_multilabel_focal_contrastive.py
```

**Current config values:**
- `focal_weight`: 0.8 (80% classification)
- `contrastive_weight`: 0.2 (20% representations)
- `focal_gamma`: 2.0
- `contrastive_temperature`: 0.1
- `contrastive_base_weight`: 0.1
- `num_train_epochs`: 15
- `output_dir`: multi_label/models/multilabel_focal_contrastive

---

### 2. Override with Command Line

```bash
# Override specific values
python multi_label\train_multilabel_focal_contrastive.py \
    --focal-weight 0.9 \
    --contrastive-weight 0.1 \
    --temperature 0.05
```

**Result:**
- `focal_weight`: 0.9 (OVERRIDDEN)
- `contrastive_weight`: 0.1 (OVERRIDDEN)
- `temperature`: 0.05 (OVERRIDDEN)
- `focal_gamma`: 2.0 (from config)
- Other values: from config

---

### 3. Partial Override

```bash
# Only override what you need
python multi_label\train_multilabel_focal_contrastive.py --epochs 20
```

**Result:**
- `num_epochs`: 20 (OVERRIDDEN)
- All other values: from config

---

## 🎯 Common Use Cases

### **Default Training (Recommended)**

```bash
# Uses all settings from config_multi.yaml
python multi_label\train_multilabel_focal_contrastive.py
```

### **Quick Experiment**

```bash
# Try different focal/contrastive balance
python multi_label\train_multilabel_focal_contrastive.py \
    --focal-weight 0.85 \
    --contrastive-weight 0.15
```

### **Longer Training**

```bash
# More epochs without changing config
python multi_label\train_multilabel_focal_contrastive.py --epochs 20
```

### **Custom Output Directory**

```bash
# Save to different location
python multi_label\train_multilabel_focal_contrastive.py \
    --output-dir experiments/focal_0.9
```

---

## 📝 Editing Config File

Edit `multi_label/config_multi.yaml`:

```yaml
# Multi-label specific settings
multi_label:
  # Loss function combination
  focal_weight: 0.8         # 80% focus on classification
  contrastive_weight: 0.2   # 20% focus on representations
  
  # Focal Loss settings
  focal_gamma: 2.0          # Focusing parameter
  
  # Contrastive Learning settings
  contrastive_temperature: 0.1    # Temperature for softmax
  contrastive_base_weight: 0.1    # Base weight for soft similarity

training:
  num_train_epochs: 15      # Number of epochs
  
paths:
  output_dir: "multi_label/models/multilabel_focal_contrastive"
```

**After editing:**
- All future runs use new values
- Can still override with command line
- Easy to version control experiments

---

## 🔧 Available Command Line Args

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `multi_label/config_multi.yaml` | Config file path |
| `--epochs` | int | from config (15) | Number of epochs |
| `--output-dir` | str | from config | Output directory |
| `--focal-weight` | float | from config (0.8) | Focal loss weight |
| `--contrastive-weight` | float | from config (0.2) | Contrastive loss weight |
| `--temperature` | float | from config (0.1) | Contrastive temperature |

**Note:** `focal_gamma` and `contrastive_base_weight` can only be set in config (not commonly changed).

---

## ✅ Advantages

### Config-Based (New Approach)
- ✅ Easy to track experiments
- ✅ Version control friendly
- ✅ No need to remember long commands
- ✅ Reproducible by default
- ✅ Still flexible with overrides

### Hardcoded (Old Approach)
- ❌ Hard to track what was used
- ❌ Must remember all parameters
- ❌ Difficult to reproduce
- ❌ No experiment history

---

## 🧪 Testing

Verify config loading works:

```bash
python multi_label\test_config_loading.py
```

**Expected output:**
```
ALL TESTS PASSED!

Summary:
   - Config file loads correctly
   - Command line args override config values
   - Partial overrides work correctly
   - Non-overridden values use config defaults
```

---

## 📊 Example Workflows

### **Experiment 1: Aggressive Focal**

Edit config:
```yaml
multi_label:
  focal_weight: 0.9
  contrastive_weight: 0.1
```

Run:
```bash
python multi_label\train_multilabel_focal_contrastive.py
```

### **Experiment 2: Balanced**

Edit config:
```yaml
multi_label:
  focal_weight: 0.5
  contrastive_weight: 0.5
```

Run:
```bash
python multi_label\train_multilabel_focal_contrastive.py
```

### **Experiment 3: Quick Test**

Don't edit config, just override:
```bash
python multi_label\train_multilabel_focal_contrastive.py \
    --epochs 3 \
    --output-dir test_run
```

---

## 🎓 Best Practices

1. **Default experiments:** Edit config file
2. **Quick tests:** Use command line overrides
3. **Production runs:** Use config file (reproducible)
4. **Version control:** Commit config changes with code
5. **Documentation:** Update config comments when changing

---

## 🚀 Quick Reference

### Run with config defaults:
```bash
python multi_label\train_multilabel_focal_contrastive.py
```

### Run with overrides:
```bash
python multi_label\train_multilabel_focal_contrastive.py --focal-weight 0.85 --epochs 20
```

### Check current config:
```bash
python multi_label\test_config_loading.py
```

---

## Changes Made

✅ Updated `train_multilabel_focal_contrastive.py`:
- Reads all contrastive configs from YAML by default
- Command line args override config values
- Better default config path: `multi_label/config_multi.yaml`

✅ Config file `multi_label/config_multi.yaml`:
- Already has all required settings
- Well-documented with comments
- Ready to use

✅ Test script `test_config_loading.py`:
- Verifies config loading logic
- Tests override behavior
- Validates partial overrides
