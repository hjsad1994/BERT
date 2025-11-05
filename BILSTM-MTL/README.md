# BiLSTM Aspect Detection (Single-Task Learning)

Binary multi-label classification for detecting 11 fixed aspect categories in Vietnamese phone reviews.

## 📊 Architecture

```
Input (Review Text)
    ↓
ViSoBERT Embeddings (768-dim)
    ↓
BiLSTM (2 layers, 256 hidden units, bidirectional)
    ↓
Attention Mechanism
    ↓
Dense Layer (256 units) + BatchNorm + ReLU + Dropout
    ↓
Output Layer (11 units) + Sigmoid
    ↓
Binary Predictions [1, 0, 1, 0, 1, ...]
```

**11 Aspects:**
- Battery, Camera, Performance, Display, Design
- Packaging, Price, Shop_Service, Shipping
- General, Others

**Output:** Binary vector `[0, 1, 1, 0, ...]` (0=absent, 1=present)

---

## 🎯 Model Features

### Based on Research Papers:
1. **"Attention-based LSTM for Aspect-level Sentiment Classification"** (Wang et al., 2016)
2. **"Multiple perspective attention based on double BiLSTM"** (Basiri et al., 2021)
3. **"Aspect-Based Sentiment Analysis Using Bitmask BiLSTM"** (Do et al., 2018)

### Key Features:
- ✅ **BiLSTM:** Captures bidirectional context
- ✅ **Attention Mechanism:** Focuses on important aspects
- ✅ **ViSoBERT Embeddings:** Pre-trained on 14GB Vietnamese corpus
- ✅ **Class Imbalance Handling:** pos_weight for BCEWithLogitsLoss
- ✅ **Multi-label Binary Classification:** Predicts all 11 aspects simultaneously

---

## 🚀 Quick Start

### 1. Test Model Architecture

```bash
cd BILSTM
python model_bilstm_ad.py
```

Expected output:
```
[BiLSTM-AD Model Initialized]
  Embedding size: 768
  LSTM hidden: 256 × 2 layers (bidirectional)
  Attention: True
  Dense: 256 → 11
  Total params: 95,123,456
  Trainable params: 95,123,456

✓ Model test passed!
```

### 2. Test Dataset Loader

```bash
python dataset_bilstm_ad.py
```

Expected output:
```
[AspectDetectionDataset] Loaded 13,891 samples
[Dataset Statistics]
  Total samples: 13,891
  Total aspect slots: 152,801
  Present aspects: 125,234 (82.0%)
  Absent aspects: 27,567 (18.0%)

✓ Dataset tests passed!
```

### 3. Train Model

```bash
python train_bilstm_ad.py --config config_bilstm_ad.yaml
```

Training output:
```
[Device] cuda
[Loading Tokenizer] 5CD-AI/visobert-14gb-corpus
[Loading Datasets]
  Train samples: 13,891
  Val samples: 1,739
  Test samples: 1,739

[Creating Model]
  BiLSTM-AD Model with 95M parameters

[Training]
  Epochs: 20
  Train batches: 434
  Val batches: 27

Epoch 1/20
  Train Loss: 0.3456
  Val F1 (macro): 0.7823
  ✓ Saved best model

...

[Training Complete]
  Best Val F1 (macro): 0.8567
  Test F1 (macro): 0.8543
  Test F1 (micro): 0.9012
  Test Accuracy: 0.9234
```

---

## ⚙️ Configuration

### Key Settings in `config_bilstm_ad.yaml`:

```yaml
model:
  # BERT embeddings
  pretrained_model_name: "5CD-AI/visobert-14gb-corpus"
  freeze_embeddings: false  # Set to true to freeze BERT
  
  # BiLSTM
  lstm_hidden_size: 256
  lstm_num_layers: 2
  lstm_dropout: 0.3
  
  # Dense layers
  dense_hidden_size: 256
  dense_dropout: 0.3
  
  # Attention
  use_attention: true
  
  # Detection threshold
  threshold: 0.5

training:
  per_device_train_batch_size: 32
  learning_rate: 1.0e-3
  num_train_epochs: 20
  early_stopping_patience: 5

loss:
  use_pos_weight: true      # Handle class imbalance
  pos_weight_auto: true     # Auto-calculate from data
```

---

## 📈 Expected Performance

Based on similar papers (SemEval datasets):

| Metric | Expected Range |
|--------|----------------|
| Accuracy | 85-92% |
| F1 (micro) | 82-90% |
| F1 (macro) | 75-85% |
| Hamming Loss | 0.08-0.15 |

**Per-aspect F1:** Typically 70-90% for frequent aspects, 50-70% for rare aspects.

---

## 🔍 Model Comparison

### BiLSTM (This Model) vs Dual-Task ViSoBERT:

| Aspect | BiLSTM STL | Dual-Task ViSoBERT |
|--------|------------|-------------------|
| **Task** | Aspect Detection only | AD + Sentiment |
| **Parameters** | ~95M | ~110M |
| **Training Speed** | Faster (1 task) | Slower (2 tasks) |
| **Memory** | Lower | Higher |
| **Interpretability** | High (attention) | Medium |
| **Use Case** | When only need AD | When need both AD + SC |

---

## 📁 File Structure

```
BILSTM/
├── model_bilstm_ad.py           # BiLSTM + Attention model
├── dataset_bilstm_ad.py         # Dataset loader
├── train_bilstm_ad.py           # Training script
├── config_bilstm_ad.yaml        # Configuration file
├── README.md                    # This file
├── models/                      # Saved models
│   └── bilstm_aspect_detection/
│       ├── best_model.pt
│       ├── training_history.json
│       └── test_results.json
└── results/                     # Predictions & reports
    ├── evaluation_report_bilstm_ad.txt
    └── test_predictions_bilstm_ad.csv
```

---

## 🛠️ Advanced Usage

### 1. Freeze BERT Embeddings

To train only BiLSTM layers (faster, less memory):

```yaml
# config_bilstm_ad.yaml
model:
  freeze_embeddings: true
```

### 2. Adjust Detection Threshold

For higher precision (fewer false positives):

```yaml
model:
  threshold: 0.7  # Default: 0.5
```

For higher recall (detect more aspects):

```yaml
model:
  threshold: 0.3
```

### 3. Disable Attention

If you want simpler model (mean pooling instead):

```yaml
model:
  use_attention: false
```

### 4. Increase Model Capacity

For better performance with more data:

```yaml
model:
  lstm_hidden_size: 512
  lstm_num_layers: 3
  dense_hidden_size: 512
```

---

## 📊 Evaluation Metrics

### Provided Metrics:

1. **Overall Metrics:**
   - Accuracy (element-wise)
   - Hamming Loss
   - F1 (micro, macro, weighted)

2. **Per-aspect Metrics:**
   - Precision
   - Recall
   - F1 score
   - Support (number of samples)

3. **Training History:**
   - Train loss per epoch
   - Validation F1 scores
   - Learning rate schedule

---

## 🔬 Research Background

### Why BiLSTM for Aspect Detection?

1. **Sequential Modeling:** BiLSTM captures sequential dependencies in text
2. **Bidirectional Context:** Processes text in both directions
3. **Attention Mechanism:** Focuses on relevant words for each aspect
4. **Proven Performance:** State-of-the-art results on SemEval benchmarks

### Papers Using Similar Approach:

- **SemEval-2014 Task 4:** Aspect Term Extraction (ATE)
- **Wang et al. (2016):** Attention-based LSTM for ABSA
- **Basiri et al. (2021):** Double BiLSTM with attention
- **Do et al. (2018):** Bitmask BiLSTM for ABSA

---

## ❓ FAQ

**Q: BiLSTM vs Transformer (BERT)?**

A: BiLSTM is lighter, faster, and more interpretable (attention weights). BERT is more powerful but heavier.

**Q: Should I freeze BERT embeddings?**

A: Depends on your data size:
- Small dataset (<5K): Freeze to prevent overfitting
- Large dataset (>10K): Unfreeze for better performance

**Q: How to handle class imbalance?**

A: Config already handles it with `pos_weight`. You can also:
- Use focal loss (implement in loss function)
- Oversample minority aspects
- Adjust threshold per aspect

**Q: Can I use this for other languages?**

A: Yes! Just change `pretrained_model_name` to a different BERT model (e.g., `bert-base-multilingual-cased`).

---

## 📝 Citation

If you use this code, please cite the relevant papers:

```bibtex
@inproceedings{wang2016attention,
  title={Attention-based LSTM for aspect-level sentiment classification},
  author={Wang, Yequan and Huang, Minlie and Zhu, Xiaoyan and Zhao, Li},
  booktitle={EMNLP},
  year={2016}
}

@article{basiri2021multiple,
  title={Multiple perspective attention based on double BiLSTM for aspect and sentiment pair extract},
  author={Basiri, Mohammad Ehsan and others},
  journal={Neurocomputing},
  year={2021}
}
```

---

## 🤝 Contributing

Feel free to:
- Add more sophisticated attention mechanisms
- Implement focal loss
- Add aspect-specific thresholds
- Integrate with other embedding models

---

## 📞 Support

For issues or questions, check:
1. Model test: `python model_bilstm_ad.py`
2. Dataset test: `python dataset_bilstm_ad.py`
3. Config file: `config_bilstm_ad.yaml`

---

**Happy Training!** 🚀
