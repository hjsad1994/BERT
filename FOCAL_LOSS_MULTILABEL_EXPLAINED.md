# 🔥 Focal Loss trong Multi-Label ABSA

## 🎯 Vấn Đề: Multi-Label vs Single-Label

### Single-Label (đơn giản):
```python
Input:  ("Pin tốt, camera kém", "Battery")
Output: [positive, negative, neutral]  # Shape: [batch, 3]
Alpha:  [w1, w2, w3]                   # Shape: [3]

# Focal Loss: 1 alpha vector cho 3 classes
```

### Multi-Label (phức tạp):
```python
Input:  "Pin tốt, camera kém"
Output: 11 aspects × 3 sentiments       # Shape: [batch, 11, 3]
Alpha:  ???                             # 2 options!

# Option 1: GLOBAL ALPHA (same for all aspects)
alpha = [w1, w2, w3]                   # Shape: [3]
# Apply same weights to ALL aspects

# Option 2: PER-ASPECT ALPHA (different per aspect)
alpha = [[w1, w2, w3],                 # Battery
         [w1, w2, w3],                 # Camera
         ...                           # 11 aspects
         [w1, w2, w3]]                 # Others
# Shape: [11, 3]
```

---

## 📊 Hai Approaches

### Approach 1: GLOBAL ALPHA ⭐ Recommended

**Concept**: Dùng CÙNG alpha weights cho TẤT CẢ aspects

```python
# Config
focal_alpha: "auto"  # hoặc [1.0, 1.2, 2.0]

# Implementation
alpha = [0.8, 1.0, 2.5]  # Global: [positive, negative, neutral]

# Apply to all aspects
for aspect_idx in range(11):
    aspect_logits = output[:, aspect_idx, :]  # [batch, 3]
    aspect_labels = labels[:, aspect_idx]      # [batch]
    
    # Focal Loss với SAME alpha cho mọi aspect
    loss += focal_loss(aspect_logits, aspect_labels, alpha=alpha)
```

**Pros**:
- ✅ Simple và consistent
- ✅ Easy to configure (chỉ 3 values)
- ✅ Works well in practice

**Cons**:
- ⚠️ Không adapt per aspect (Battery và Camera dùng chung weights)

---

### Approach 2: PER-ASPECT ALPHA (Advanced)

**Concept**: Mỗi aspect có RIÊNG alpha weights

```python
# Config
focal_alpha: 
  Battery: [0.8, 1.0, 2.0]
  Camera: [1.0, 1.5, 3.0]     # Camera neutral quan trọng hơn
  Performance: [0.9, 1.0, 1.5]
  ...

# Implementation
alpha = {
    0: [0.8, 1.0, 2.0],  # Battery
    1: [1.0, 1.5, 3.0],  # Camera
    ...
}

# Apply per-aspect alpha
for aspect_idx in range(11):
    aspect_logits = output[:, aspect_idx, :]
    aspect_labels = labels[:, aspect_idx]
    
    # Focal Loss với DIFFERENT alpha per aspect
    aspect_alpha = alpha[aspect_idx]
    loss += focal_loss(aspect_logits, aspect_labels, alpha=aspect_alpha)
```

**Pros**:
- ✅ Flexible: adapt per aspect
- ✅ Handle aspect-specific imbalance

**Cons**:
- ⚠️ Complex config (11 × 3 = 33 values!)
- ⚠️ Hard to tune
- ⚠️ May overfit

---

## 💡 Recommendation: Use GLOBAL ALPHA

**Lý do**:
1. **Simplicity**: Chỉ cần config 3 values (positive, negative, neutral)
2. **Consistency**: Same criteria across all aspects
3. **Effectiveness**: Works well in practice (proven in papers)
4. **Interpretability**: Easy to understand và debug

**When to use PER-ASPECT ALPHA**:
- ❌ Khi có aspects với dramatically different distributions
- ❌ Khi domain knowledge chỉ ra aspect X cần special treatment
- ❌ Khi đã thử global và không đủ tốt

**Reality**: 95% cases, GLOBAL ALPHA is enough!

---

## 🏗️ Implementation for Multi-Label

### Option 1: GLOBAL ALPHA (Recommended)

```python
class MultilabelFocalLoss(nn.Module):
    def __init__(self, alpha=[1.0, 1.0, 1.0], gamma=2.0, num_aspects=11):
        super().__init__()
        self.alpha = torch.tensor(alpha)  # [3]
        self.gamma = gamma
        self.num_aspects = num_aspects
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [batch, num_aspects, num_sentiments]
            labels: [batch, num_aspects]
        
        Returns:
            loss: scalar
        """
        batch_size = logits.size(0)
        total_loss = 0
        
        # Loop over aspects
        for aspect_idx in range(self.num_aspects):
            aspect_logits = logits[:, aspect_idx, :]  # [batch, 3]
            aspect_labels = labels[:, aspect_idx]      # [batch]
            
            # Softmax
            probs = F.softmax(aspect_logits, dim=1)  # [batch, 3]
            
            # Get prob of correct class
            class_probs = probs[range(batch_size), aspect_labels]  # [batch]
            
            # Focal weight: (1 - p)^gamma
            focal_weight = (1 - class_probs) ** self.gamma
            
            # Class weight (alpha)
            alpha_weight = self.alpha[aspect_labels]  # [batch]
            
            # Cross entropy
            ce_loss = F.cross_entropy(aspect_logits, aspect_labels, reduction='none')
            
            # Focal loss
            loss = alpha_weight * focal_weight * ce_loss
            total_loss += loss.mean()
        
        # Average over aspects
        return total_loss / self.num_aspects
```

**Config**:
```yaml
multi_label:
  use_focal_loss: true
  focal_gamma: 2.0
  focal_alpha: "auto"  # Global alpha for all aspects
```

---

### Option 2: PER-ASPECT ALPHA (Advanced)

```python
class PerAspectFocalLoss(nn.Module):
    def __init__(self, alpha_dict, gamma=2.0):
        super().__init__()
        # alpha_dict: {aspect_idx: [w1, w2, w3]}
        self.alpha_dict = {k: torch.tensor(v) for k, v in alpha_dict.items()}
        self.gamma = gamma
    
    def forward(self, logits, labels):
        batch_size = logits.size(0)
        num_aspects = logits.size(1)
        total_loss = 0
        
        for aspect_idx in range(num_aspects):
            aspect_logits = logits[:, aspect_idx, :]
            aspect_labels = labels[:, aspect_idx]
            
            # Get aspect-specific alpha
            aspect_alpha = self.alpha_dict[aspect_idx]
            
            # Compute focal loss with aspect-specific alpha
            probs = F.softmax(aspect_logits, dim=1)
            class_probs = probs[range(batch_size), aspect_labels]
            focal_weight = (1 - class_probs) ** self.gamma
            alpha_weight = aspect_alpha[aspect_labels]
            ce_loss = F.cross_entropy(aspect_logits, aspect_labels, reduction='none')
            
            loss = alpha_weight * focal_weight * ce_loss
            total_loss += loss.mean()
        
        return total_loss / num_aspects
```

**Config**:
```yaml
multi_label:
  use_focal_loss: true
  focal_gamma: 2.0
  focal_alpha:
    Battery: [0.8, 1.0, 2.0]
    Camera: [1.0, 1.5, 3.0]
    Performance: [0.9, 1.0, 1.5]
    # ... 11 aspects
```

---

## 📊 Comparison

| Aspect | Global Alpha | Per-Aspect Alpha |
|--------|--------------|------------------|
| **Config size** | 3 values | 33 values (11×3) |
| **Complexity** | Simple | Complex |
| **Tuning** | Easy | Hard |
| **Flexibility** | Low | High |
| **Use cases** | 95% cases | Special cases |
| **Interpretability** | High | Low |

---

## 🎯 Auto-Calculate Alpha in Multi-Label

### Global Auto (Recommended):

```python
def calculate_global_alpha(train_df, aspect_cols):
    """
    Calculate global alpha from ALL aspects combined
    
    Returns: [alpha_pos, alpha_neg, alpha_neu]
    """
    # Collect all sentiments from all aspects
    all_sentiments = []
    for aspect in aspect_cols:
        sentiments = train_df[aspect].dropna()
        all_sentiments.extend(sentiments.tolist())
    
    # Count
    from collections import Counter
    counts = Counter(all_sentiments)
    total = sum(counts.values())
    
    # Inverse frequency
    alpha = []
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        count = counts.get(sentiment, 1)
        alpha.append(total / (3 * count))
    
    print(f"Global alpha (auto): {alpha}")
    return alpha
```

**Example output**:
```
Sentiment counts across all aspects:
  Positive: 35,000
  Negative: 20,000
  Neutral:  8,000

Global alpha:
  Positive: 0.60
  Negative: 1.05
  Neutral:  2.63
```

---

### Per-Aspect Auto (Advanced):

```python
def calculate_per_aspect_alpha(train_df, aspect_cols):
    """
    Calculate alpha per aspect
    
    Returns: {aspect_idx: [alpha_pos, alpha_neg, alpha_neu]}
    """
    alpha_dict = {}
    
    for aspect_idx, aspect in enumerate(aspect_cols):
        sentiments = train_df[aspect].dropna()
        
        from collections import Counter
        counts = Counter(sentiments)
        total = len(sentiments)
        
        alpha = []
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            count = counts.get(sentiment, 1)
            alpha.append(total / (3 * count))
        
        alpha_dict[aspect_idx] = alpha
        print(f"{aspect}: {alpha}")
    
    return alpha_dict
```

**Example output**:
```
Battery:     [0.58, 1.02, 2.89]  ← Battery-specific
Camera:      [0.72, 1.15, 2.34]  ← Camera-specific
Performance: [0.65, 0.98, 3.12]  ← Performance-specific
...
```

---

## 🎓 Real Example

### Dataset:
```
Reviews: 7,000
Total aspect-sentiment pairs: 7,000 × 11 = 77,000 (potential)
Actual mentioned: ~35,000 (many aspects not mentioned)

Overall distribution:
  Positive: 20,000 (57%)
  Negative: 10,000 (29%)
  Neutral:  5,000 (14%)
```

### Global Alpha (Simple):
```python
focal_alpha: "auto"

# Computed:
alpha = [0.58, 1.05, 2.50]
# Applied to ALL aspects
```

### Per-Aspect Alpha (Complex):
```python
focal_alpha:
  Battery:
    Positive: 18% (1,260 / 7,000)
    Negative: 8%  (560 / 7,000)
    Neutral: 3%   (210 / 7,000)
    → Alpha: [1.52, 3.12, 8.33]  # Neutral heavily weighted
  
  Camera:
    Positive: 22% (1,540 / 7,000)
    Negative: 15% (1,050 / 7,000)
    Neutral: 5%   (350 / 7,000)
    → Alpha: [1.25, 1.98, 5.00]
  
  ... (9 more aspects)
```

**Question**: Phức tạp quá, có cần thiết không?

**Answer**: Thường KHÔNG cần! Global alpha đủ tốt.

---

## 💡 Practical Recommendation

### Start Simple:

```yaml
# config_multi.yaml
multi_label:
  use_focal_loss: true
  focal_gamma: 2.0
  focal_alpha: "auto"  # Global alpha
```

**Code**:
```python
# Calculate global alpha
all_sentiments = []
for aspect in aspect_cols:
    all_sentiments.extend(train_df[aspect].dropna().tolist())

counts = Counter(all_sentiments)
alpha = [counts_to_alpha(counts)]  # [0.58, 1.05, 2.50]

# Use for ALL aspects
focal_loss = MultilabelFocalLoss(alpha=alpha, gamma=2.0)
```

---

### If Needed, Upgrade:

```yaml
multi_label:
  focal_alpha:
    mode: "per_aspect"
    Battery: [0.8, 1.0, 2.0]
    Camera: [1.0, 1.5, 3.0]
    # ... (only if global doesn't work)
```

---

## 🧪 Experiment Guide

### Experiment 1: Baseline (No Focal Loss)
```yaml
use_focal_loss: false
```

### Experiment 2: Focal Loss (Global Alpha - Auto)
```yaml
use_focal_loss: true
focal_gamma: 2.0
focal_alpha: "auto"
```

### Experiment 3: Focal Loss (Global Alpha - Custom)
```yaml
use_focal_loss: true
focal_gamma: 2.0
focal_alpha: [0.8, 1.0, 2.5]  # Boost neutral
```

### Experiment 4: Focal Loss (Per-Aspect Alpha)
```yaml
use_focal_loss: true
focal_gamma: 2.0
focal_alpha:
  mode: "per_aspect"
  Battery: [auto]
  Camera: [auto]
  # ... compute per aspect
```

**Expected**: Experiments 1 → 2 → 3 should improve. Experiment 4 may overfit.

---

## ✅ Summary

### For Multi-Label ABSA:

1. **Use GLOBAL ALPHA** (recommended)
   - Config: `focal_alpha: "auto"`
   - Simple: 3 values for all aspects
   - Works well: proven effective

2. **Use PER-ASPECT ALPHA** (advanced)
   - Config: 11 × 3 = 33 values
   - Complex: hard to tune
   - Rarely needed: only for special cases

3. **Implementation**:
   ```python
   # Global alpha
   for aspect in range(11):
       loss += focal_loss(logits[:, aspect, :], 
                         labels[:, aspect],
                         alpha=global_alpha)  # Same for all
   ```

4. **Config**:
   ```yaml
   multi_label:
     focal_alpha: "auto"  # Recommended
     # focal_alpha: [w1, w2, w3]  # Custom global
     # focal_alpha: {Battery: [...], Camera: [...]}  # Per-aspect
   ```

---

**Recommendation**: Start với GLOBAL ALPHA ("auto"). 99% cases đủ rồi!
