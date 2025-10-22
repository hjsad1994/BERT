# Single-Label ABSA (Traditional Approach)

## 📖 Overview

Traditional single-label approach where each sample contains **one sentence-aspect pair**.

**Format:**
```
sentence, aspect, sentiment
"Pin tốt", Battery, positive
"Camera xấu", Camera, negative
```

**Characteristics:**
- One aspect per sample
- Sequential processing (slower)
- Simpler model architecture
- Standard cross-entropy loss

---

## 🎯 Performance

**Baseline (Balanced Oversampling):**
- Accuracy: 89.91%
- F1 Score: 90.77%
- Confidence: 83.01%

**With Focal Loss:**
- Expected: 91-92% F1

---

## 📁 Files

### **Training:**
- `train.py` - Main training script
- `prepare_data.py` - Convert dataset to single-label format
- `config_single.yaml` - Configuration

### **Augmentation:**
- `augment_neutral_and_nhung.py` - Neutral + "nhưng" augmentation
- `aspect_wise_oversampling.py` - Per-aspect oversampling
- `oversampling_utils.py` - Oversampling utilities

### **Analysis:**
- `analyze_results.py` - Comprehensive analysis with visualizations
- `analyze_nhung_errors.py` - Error analysis for "nhưng" samples
- `generate_predictions.py` - Batch predictions
- `generate_test_predictions.py` - Test set predictions

### **Utilities:**
- `utils.py` - Core utilities (FocalLoss, ABSADataset, metrics)
- `checkpoint_renamer.py` - Rename checkpoints by accuracy
- `focal_loss_trainer.py` - Custom Trainer with Focal Loss
- `filter_long_sequences.py` - Filter sequences > 256 tokens
- `shuffle_dataset.py` - Shuffle dataset
- `predict_example.py` - Interactive prediction demo

---

## 🚀 Quick Start

### **1. Prepare Data**

```bash
python prepare_data.py
```

**Output:**
```
data/train.csv         (80% - 5,847 samples)
data/validation.csv    (10% - 731 samples)
data/test.csv          (10% - 731 samples)
```

---

### **2. Train Model**

```bash
python train.py --config config_single.yaml
```

**Training details:**
- Batch size: 64
- Epochs: 5
- Learning rate: 2e-5
- Expected time: ~30 minutes on RTX 3070

---

### **3. Evaluate**

```bash
python generate_test_predictions.py
python analyze_results.py
```

**Output:**
```
results/test_predictions_single.csv
results/evaluation_report_single.txt
analysis_results/ (visualizations)
```

---

### **4. Interactive Prediction**

```bash
python predict_example.py
```

---

## 📊 Data Format

### **Input (dataset.csv):**
```csv
data,Battery,Camera,Performance,...
"Pin tốt camera xấu",Positive,Negative,Neutral,...
```

### **Converted (train.csv):**
```csv
sentence,aspect,sentiment
"Pin tốt camera xấu",Battery,positive
"Pin tốt camera xấu",Camera,negative
"Pin tốt camera xấu",Performance,neutral
...
```

**Result:** One sentence → Multiple rows (one per aspect)

---

## 🎯 Training Options

### **A. Baseline (No Oversampling)**

```bash
python train.py --config config_single.yaml
```

**Expected:** 90-91% F1

---

### **B. With Aspect-Wise Oversampling**

```bash
# Enable in config_single.yaml:
single_label:
  use_oversampling: true

python train.py --config config_single.yaml
```

**Expected:** 91-92% F1

---

### **C. With Focal Loss**

```bash
# Enable in config_single.yaml:
single_label:
  use_focal_loss: true
  focal_gamma: 2.0

python train.py --config config_single.yaml
```

**Expected:** 91-92% F1

---

## 📈 Advantages

✅ **Simpler implementation**  
✅ **Easier to understand**  
✅ **Standard benchmarks**  
✅ **Works well for small datasets**

---

## ❌ Disadvantages

❌ **Slower inference** (11× slower than multi-label)  
❌ **More training data needed** (duplicate sentences)  
❌ **Lower performance** (90-92% vs 96% multi-label)  
❌ **No aspect relationships** (treats aspects independently)

---

## 🔧 Configuration

**Key settings in `config_single.yaml`:**

```yaml
paths:
  train_file: "../data/train.csv"  # Single-label format
  
model:
  num_labels: 3  # positive, negative, neutral

training:
  per_device_train_batch_size: 64  # Larger (smaller samples)
  num_train_epochs: 5

single_label:
  use_oversampling: false
  use_focal_loss: false
  focal_gamma: 2.0
```

---

## 📊 Expected Results

| Method | F1 Score | Training Time |
|--------|----------|---------------|
| Baseline | 90.77% | 30 mins |
| + Oversampling | 91-92% | 35 mins |
| + Focal Loss | 91-92% | 30 mins |

**vs Multi-Label:** 90-92% vs 96% F1

---

## 🎓 Use Cases

**When to use single-label:**
- Learning/understanding ABSA
- Smaller datasets (< 5k reviews)
- Need standard benchmarks
- Simpler deployment

**When to use multi-label instead:**
- Production systems (need speed)
- Larger datasets (> 10k reviews)
- Need better performance (96%+)
- Research paper (novel approach)

---

## 📝 Citation

```bibtex
@article{your_paper,
  title={Vietnamese Aspect-Based Sentiment Analysis with ViSoBERT},
  author={Your Name},
  journal={Your Conference},
  year={2024}
}
```

---

## 🔗 See Also

- **Multi-Label approach:** `../multi_label/README.md` (96% F1, 11× faster)
- **Main README:** `../README.md`
- **Documentation:** `../docs/`
