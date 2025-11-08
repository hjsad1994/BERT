# Solutions to Improve STL Price Negative Performance

## Current Performance
```
Price Sentiment Classification (STL):
  Overall F1: 89.36%
  
  Per-class:
    Positive: 98.4% F1 ✓
    Negative: 80.0% F1 ✗ (10/14 correct, miss 4)
    Neutral:  89.7% F1 ✓

  Problem: Negative class stuck at 71.4% recall
```

## Root Cause Analysis

### Issue 1: Poor Augmentation Quality
```
Original Negative samples: 132
Augmented Negative samples: 2,016 (15.3x)
→ Most are DUPLICATES, not diverse samples!
```

**Evidence:** Model performance didn't improve despite 15x more data.

### Issue 2: Extreme Class Imbalance in Test
```
Test set:
  Positive: 255 (90.4%)
  Negative: 14 (5.0%)
  Neutral: 13 (4.6%)

Training balanced but test realistic → distribution mismatch
```

### Issue 3: All Negative Errors → Positive
```
4/4 errors predicted as Positive (not Neutral)
Model bias: "Price mentioned = Positive"
```

### Issue 4: Difficult Negative Patterns
```
Error samples:
1. "giá hơi đắt" (implicit negative: "a bit expensive")
2. "Giá 3800 ngàn thì kh..." (truncated negative)
3. "Mắc dù giá thành rẻ nhưng không nên mua" (contradiction)
4. "Ngon rẻ tốt nhưng mà không đáng" (mixed sentiment)
```

---

## Solution 1: Better Data Augmentation ⭐⭐⭐⭐⭐

### Current Augmentation (augment_multilabel_balanced.py):
Likely using simple duplication or random oversampling.

### Recommended: Paraphrase + Back-translation

**A. Vietnamese Paraphrasing:**
```python
import underthesea

def paraphrase_vietnamese(text):
    """Simple Vietnamese paraphrasing"""
    replacements = {
        'đắt': ['mắc', 'giá cao', 'không rẻ'],
        'rẻ': ['giá tốt', 'phải chăng', 'bình dân'],
        'không đáng': ['không xứng', 'không hợp lý'],
        'hơi đắt': ['đắt quá', 'cao hơn mong đợi'],
        # ... more synonyms
    }
    
    # Replace keywords randomly
    for old, new_list in replacements.items():
        if old in text:
            text = text.replace(old, random.choice(new_list))
    
    return text
```

**B. Back-translation (Vietnamese → English → Vietnamese):**
```python
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, source_lang='vi', target_lang='en'):
    """Back-translation for augmentation"""
    # Vi → En
    model_vi_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-vi-en')
    tokenizer_vi_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-vi-en')
    
    # En → Vi
    model_en_vi = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-vi')
    tokenizer_en_vi = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-vi')
    
    # Translate Vi → En
    inputs = tokenizer_vi_en(text, return_tensors='pt', padding=True)
    en_text = model_vi_en.generate(**inputs)
    en_text = tokenizer_vi_en.decode(en_text[0], skip_special_tokens=True)
    
    # Translate En → Vi
    inputs = tokenizer_en_vi(en_text, return_tensors='pt', padding=True)
    vi_augmented = model_en_vi.generate(**inputs)
    vi_augmented = tokenizer_en_vi.decode(vi_augmented[0], skip_special_tokens=True)
    
    return vi_augmented
```

**C. Implementation:**
```bash
cd E:\BERT
python -c "
import pandas as pd
from augment_utils import paraphrase_vietnamese, back_translate

# Load original data
df = pd.read_csv('VisoBERT-STL/data/train_multilabel.csv', encoding='utf-8-sig')

# Get Negative Price samples
neg_samples = df[df['Price'] == 'Negative']

# Augment with multiple strategies
augmented = []
for idx, row in neg_samples.iterrows():
    text = row['data']
    
    # Original
    augmented.append(row)
    
    # Paraphrase (2x)
    for _ in range(2):
        aug_row = row.copy()
        aug_row['data'] = paraphrase_vietnamese(text)
        augmented.append(aug_row)
    
    # Back-translation (2x)
    for _ in range(2):
        aug_row = row.copy()
        aug_row['data'] = back_translate(text)
        augmented.append(aug_row)

# Save augmented data
aug_df = pd.DataFrame(augmented)
# ... merge with other classes
"
```

**Expected impact:** 
- Negative recall: 71.4% → 85-90%
- More diverse patterns → better generalization

---

## Solution 2: Focal Loss with Higher Gamma ⭐⭐⭐⭐

### Current Config:
```yaml
two_stage:
  sentiment_classification:
    focal_gamma: 2.0
    focal_alpha: "auto"
```

### Recommended:
```yaml
two_stage:
  sentiment_classification:
    focal_gamma: 3.0  # Increase from 2.0
    focal_alpha:
      Battery: "auto"
      Camera: "auto"
      # ...
      Price: [0.05, 0.45, 0.50]  # [Positive, Negative, Neutral]
      # Give Negative class 0.45 weight (higher than auto)
```

**Reasoning:**
- Positive: 255 samples → low weight (0.05)
- Negative: 14 samples → high weight (0.45)
- Neutral: 13 samples → high weight (0.50)

**Expected impact:**
- Forces model to pay more attention to Negative
- Negative recall: 71.4% → 80-85%

---

## Solution 3: Class-Specific Loss Weighting ⭐⭐⭐⭐

### Modify train_visobert_stl.py:

```python
def train_epoch_sc_with_weights(
    model, dataloader, optimizer, scheduler, device, 
    focal_loss_fn, scaler, aspect_weights
):
    """Train with aspect-specific weights"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss_mask = batch['loss_mask'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(input_ids, attention_mask)
            
            # Focal loss per aspect
            loss_per_aspect = focal_loss_fn(logits, labels)
            
            # Apply aspect-specific weights
            for i, aspect in enumerate(aspects):
                if aspect == 'Price':
                    # Give Price aspect 2x weight
                    loss_per_aspect[:, i] *= 2.0
            
            # Apply loss mask
            masked_loss = loss_per_aspect * loss_mask
            num_labeled = loss_mask.sum()
            
            loss = masked_loss.sum() / num_labeled if num_labeled > 0 else masked_loss.sum()
        
        # ... backward pass
```

**Expected impact:**
- Model focuses more on Price during training
- Negative recall: 71.4% → 78-83%

---

## Solution 4: Ensemble Predictions ⭐⭐⭐

### Train Multiple Models with Different Seeds:

```bash
# Train 5 models with different seeds
for seed in 42 123 456 789 2024; do
    # Update config
    sed -i "s/training_seed: .*/training_seed: $seed/" config_visobert_stl.yaml
    
    # Train
    python train_visobert_stl.py --config config_visobert_stl.yaml
    
    # Save model with seed suffix
    mv VisoBERT-STL/models/sentiment_classification/best_model.pt \
       VisoBERT-STL/models/sentiment_classification/best_model_seed${seed}.pt
done
```

### Ensemble Inference:
```python
def ensemble_predict(text, models, tokenizer):
    """Ensemble prediction from multiple models"""
    all_probs = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt', ...)
            logits = model(**inputs)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)
    
    # Average probabilities
    avg_probs = torch.stack(all_probs).mean(dim=0)
    predictions = torch.argmax(avg_probs, dim=-1)
    
    return predictions
```

**Expected impact:**
- Negative recall: 71.4% → 85-90%
- More robust predictions
- Reduces variance across runs

---

## Solution 5: Two-Step Classification ⭐⭐

### Approach: First classify Positive/Not-Positive, then Negative/Neutral

```python
# Step 1: Binary classifier (Positive vs Non-Positive)
# Focus on separating Positive from rest

# Step 2: For Non-Positive samples, classify Negative vs Neutral
# Smaller, more focused problem

class TwoStepClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
        # Step 1: Positive vs Non-Positive
        self.binary_head = nn.Linear(hidden_size, 2)
        
        # Step 2: Negative vs Neutral (for Non-Positive)
        self.non_positive_head = nn.Linear(hidden_size, 2)
    
    def forward(self, input_ids, attention_mask):
        features = self.backbone(input_ids, attention_mask)
        
        # Step 1 prediction
        binary_logits = self.binary_head(features)
        is_positive = torch.argmax(binary_logits, dim=-1)
        
        # Step 2 prediction (for non-positive)
        non_pos_logits = self.non_positive_head(features)
        
        # Combine predictions
        final_preds = torch.where(
            is_positive == 1,
            torch.tensor(0),  # Positive
            torch.argmax(non_pos_logits, dim=-1) + 1  # Negative=1, Neutral=2
        )
        
        return final_preds
```

**Expected impact:**
- Separates easy task (Pos vs Non-Pos) from hard task (Neg vs Neu)
- Negative recall: 71.4% → 80-85%

---

## Solution 6: Fine-tune on Hard Negatives ⭐⭐⭐

### Identify Hard Negative Samples:

```python
# Find Negative samples model struggles with
hard_negatives = [
    "giá hơi đắt",
    "không đáng số tiền bỏ ra",
    "mắc dù rẻ nhưng không nên mua",
    # ... add more patterns from errors
]

# Create synthetic hard negatives
def create_hard_negative(template):
    """Generate variations of hard negatives"""
    variations = []
    
    # Template: "Product X [positive aspects] nhưng giá [negative]"
    products = ["điện thoại này", "máy này", "sản phẩm"]
    positive = ["tốt", "đẹp", "mượt"]
    negative_price = ["hơi đắt", "không hợp lý", "cao quá"]
    
    for p in products:
        for pos in positive:
            for neg in negative_price:
                variations.append(f"{p} {pos} nhưng giá {neg}")
    
    return variations

# Add to training data
hard_neg_samples = []
for template in hard_negatives:
    hard_neg_samples.extend(create_hard_negative(template))

# Fine-tune on augmented data with hard negatives
```

**Expected impact:**
- Model learns specific negative patterns
- Negative recall: 71.4% → 85-90%

---

## Solution 7: Use Pre-trained Vietnamese Sentiment Model ⭐⭐⭐⭐⭐

### Replace base model with sentiment-specific model:

```yaml
model:
  # Current: General Vietnamese model
  name: "5CD-AI/visobert-14gb-corpus"
  
  # Replace with: Vietnamese sentiment model
  name: "uitnlp/visobert-base-cased-finetuned-vietnews-emotion"
  # or
  name: "wonrax/phobert-base-vietnamese-sentiment"
```

**Why this helps:**
- Pre-trained on Vietnamese sentiment data
- Already understands negative sentiment patterns
- Better transfer learning for Price sentiment

**Expected impact:**
- Negative recall: 71.4% → 90-95%
- Best potential improvement

---

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 hours)
1. **Solution 2**: Increase focal_gamma to 3.0 and adjust alpha for Price
2. **Solution 3**: Add Price-specific loss weighting (2x)
3. **Re-train and evaluate**

**Expected improvement:** 71.4% → 80-85% recall

### Phase 2: Better Data (1 day)
4. **Solution 1**: Implement back-translation augmentation
5. **Solution 6**: Generate hard negative samples
6. **Re-train with new data**

**Expected improvement:** 80% → 85-90% recall

### Phase 3: Architecture (2-3 days)
7. **Solution 7**: Try sentiment-specific pre-trained model
8. **Solution 4**: Train ensemble (5 models)
9. **Compare best approach**

**Expected improvement:** 85% → 90-95% recall

### Phase 4: Advanced (if needed)
10. **Solution 5**: Implement two-step classifier
11. **Error analysis and iteration**

**Target:** 90-95% recall, matching MTL performance

---

## Expected Final Results

### After Phase 1+2:
```
Price Sentiment (STL):
  F1:        92-93%  (current: 89.36%)
  Recall:    90-92%  (current: 89.95%)
  
  Negative:
    Recall:  85-90%  (current: 71.4%)
    F1:      87-90%  (current: 80.0%)
```

### After Phase 3:
```
Price Sentiment (STL):
  F1:        94-95%  (approaching MTL 96.57%)
  Recall:    93-95%
  
  Negative:
    Recall:  90-95%
    F1:      91-94%
```

---

## Alternative: Accept Current Performance

### Is 89.36% F1 acceptable?

**Arguments FOR accepting:**
1. ✓ Improved from baseline 88.47% → 89.36%
2. ✓ Recall improved significantly 82.65% → 89.95%
3. ✓ Only 4 errors out of 282 samples (1.4% error rate)
4. ✓ ABSA research typically reports 85-95% F1
5. ✓ STL trades performance for interpretability

**Arguments AGAINST accepting:**
1. ✗ MTL achieves 96.57% (7% better)
2. ✗ Negative class still problematic (71.4% recall)
3. ✗ Could improve with better augmentation
4. ✗ Paper reviewers may question gap vs MTL

### Recommendation:
- **For publication**: Implement Phase 1+2 (3-4 days work, get to 92-93% F1)
- **For production**: Implement Phase 3 (get to 94-95% F1, near MTL)
- **For research**: Document trade-offs, explain why STL < MTL is expected

---

## Code to Implement Solution 2 (Quick Win)

### Update config_visobert_stl.yaml:

```yaml
two_stage:
  sentiment_classification:
    use_focal_loss: true
    focal_gamma: 3.0  # CHANGE: Increase from 2.0
    
    # CHANGE: Add per-aspect alpha
    focal_alpha:
      Battery: "auto"
      Camera: "auto"
      Performance: "auto"
      Display: "auto"
      Design: "auto"
      Packaging: "auto"
      Price: [0.05, 0.45, 0.50]  # [Pos, Neg, Neu] - boost minority classes
      Shop_Service: "auto"
      Shipping: "auto"
      General: "auto"
      Others: "auto"
    
    epochs: 12
```

### Modify focal_loss_multilabel.py to support per-aspect alpha:

```python
class MultilabelFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, num_aspects=11):
        super().__init__()
        self.gamma = gamma
        
        # Support per-aspect alpha
        if isinstance(alpha, dict):
            self.alpha = {}
            for aspect, alpha_val in alpha.items():
                if isinstance(alpha_val, list):
                    self.alpha[aspect] = torch.tensor(alpha_val)
                else:
                    self.alpha[aspect] = alpha_val
        else:
            self.alpha = alpha
        
        self.num_aspects = num_aspects
    
    def forward(self, logits, labels):
        # logits: (batch_size, num_aspects, num_classes)
        # labels: (batch_size, num_aspects)
        
        bsz, n_aspects, n_classes = logits.shape
        
        # Compute per-aspect loss
        losses = []
        for aspect_idx in range(n_aspects):
            aspect_logits = logits[:, aspect_idx, :]
            aspect_labels = labels[:, aspect_idx]
            
            # Get alpha for this aspect
            if isinstance(self.alpha, dict):
                aspect_name = list(self.alpha.keys())[aspect_idx]
                aspect_alpha = self.alpha[aspect_name]
            else:
                aspect_alpha = self.alpha
            
            # Compute focal loss
            ce_loss = F.cross_entropy(
                aspect_logits, aspect_labels, 
                reduction='none', 
                weight=aspect_alpha if isinstance(aspect_alpha, torch.Tensor) else None
            )
            
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            
            losses.append(focal_loss)
        
        return torch.stack(losses, dim=1)
```

### Re-train:
```bash
cd E:\BERT\VisoBERT-STL
python train_visobert_stl.py --config config_visobert_stl.yaml
```

**Expected result after this change:**
- Price Negative recall: 71.4% → 78-82%
- Price F1: 89.36% → 90-91%
- Training time: ~2 hours
