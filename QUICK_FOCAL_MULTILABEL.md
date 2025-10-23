# ⚡ Quick: Focal Alpha trong Multi-Label

## 🎯 Câu hỏi: focal_alpha handle như thế nào trong multi-label?

---

## 📊 Hai Approaches

### Approach 1: GLOBAL ALPHA ⭐ (Recommended)

**Concept**: CÙNG alpha weights cho TẤT CẢ aspects

```yaml
multi_label:
  focal_alpha: "auto"  # hoặc [1.0, 1.2, 2.5]
```

```python
# Implementation
alpha = [0.8, 1.0, 2.5]  # Global cho 11 aspects

for aspect in range(11):
    loss += focal_loss(logits[:, aspect, :], 
                      labels[:, aspect],
                      alpha=alpha)  # Same alpha!
```

**Pros**: ✅ Simple, ✅ Easy config (3 values), ✅ Works well

---

### Approach 2: PER-ASPECT ALPHA (Advanced)

**Concept**: MỖI aspect có RIÊNG alpha weights

```yaml
multi_label:
  focal_alpha:
    Battery: [0.8, 1.0, 2.0]
    Camera: [1.0, 1.5, 3.0]
    Performance: [0.9, 1.0, 1.5]
    # ... 11 aspects
```

```python
# Implementation
alpha_dict = {
    0: [0.8, 1.0, 2.0],  # Battery
    1: [1.0, 1.5, 3.0],  # Camera
    ...
}

for aspect in range(11):
    loss += focal_loss(logits[:, aspect, :],
                      labels[:, aspect],
                      alpha=alpha_dict[aspect])  # Different!
```

**Pros**: ✅ Flexible  
**Cons**: ❌ Complex (33 values!), ❌ Hard to tune

---

## 💡 Recommendation

```
START: Global Alpha ("auto")
  ↓
95% cases: Đủ rồi, stop here!
  ↓
Still not good?
  ↓
TRY: Global Alpha (custom [w1, w2, w3])
  ↓
Still not good?
  ↓
LAST RESORT: Per-Aspect Alpha
```

---

## 📐 Comparison

| Feature | Global | Per-Aspect |
|---------|--------|------------|
| Config | 3 values | 33 values |
| Complexity | Simple | Complex |
| Use cases | 95% | 5% |
| Tuning | Easy | Hard |

---

## 🎯 Example

### Your Multi-Label Output:
```python
Input:  "Pin tốt, camera kém, giá cao"
Output: [batch, 11 aspects, 3 sentiments]

# Global Alpha (Recommended)
alpha = [0.8, 1.0, 2.5]  # Apply to ALL 11 aspects

# Per-Aspect Alpha (Rarely needed)
alpha = {
    0: [0.8, 1.0, 2.0],  # Battery-specific
    1: [1.0, 1.5, 3.0],  # Camera-specific
    ...
}
```

---

## ✅ Config Recommendation

```yaml
# Default: Global alpha (auto)
multi_label:
  use_focal_loss: true
  focal_gamma: 2.0
  focal_alpha: "auto"  # ← RECOMMENDED

# Advanced: Global alpha (custom)
multi_label:
  focal_alpha: [1.0, 1.2, 2.5]  # Boost neutral

# Rarely: Per-aspect alpha
multi_label:
  focal_alpha:
    Battery: [0.8, 1.0, 2.0]
    Camera: [1.0, 1.5, 3.0]
    # ... 11 aspects (33 values total!)
```

---

## 🔥 Implementation

```python
class MultilabelFocalLoss(nn.Module):
    def __init__(self, alpha=[1.0, 1.0, 1.0], gamma=2.0):
        super().__init__()
        self.alpha = torch.tensor(alpha)  # [3] - global
        self.gamma = gamma
    
    def forward(self, logits, labels):
        # logits: [batch, 11, 3]
        # labels: [batch, 11]
        
        total_loss = 0
        for aspect in range(11):
            aspect_logits = logits[:, aspect, :]  # [batch, 3]
            aspect_labels = labels[:, aspect]      # [batch]
            
            # Focal loss với SAME alpha cho mọi aspect
            probs = F.softmax(aspect_logits, dim=1)
            class_probs = probs[range(batch), aspect_labels]
            
            focal_weight = (1 - class_probs) ** self.gamma
            alpha_weight = self.alpha[aspect_labels]
            ce_loss = F.cross_entropy(aspect_logits, aspect_labels, 
                                     reduction='none')
            
            loss = alpha_weight * focal_weight * ce_loss
            total_loss += loss.mean()
        
        return total_loss / 11  # Average over aspects
```

---

## ⚖️ Auto-Calculate

### Global (Simple):
```python
# Collect ALL sentiments from ALL aspects
all_sentiments = []
for aspect in aspect_cols:
    all_sentiments.extend(df[aspect].dropna())

# Count and compute
counts = Counter(all_sentiments)
alpha = [total/(3*count) for count in counts.values()]
# → [0.8, 1.0, 2.5]
```

### Per-Aspect (Complex):
```python
# Compute per aspect
alpha_dict = {}
for aspect in aspect_cols:
    sentiments = df[aspect].dropna()
    counts = Counter(sentiments)
    alpha_dict[aspect] = [total/(3*count) for count in counts.values()]

# → Battery: [0.8, 1.0, 2.0]
#    Camera: [1.0, 1.5, 3.0]
#    ...
```

---

## 🎓 Key Insight

```
Multi-Label = Single-Label repeated 11 times

For each aspect:
  - Apply softmax  [batch, 3]
  - Compute focal loss with alpha
  - Use SAME alpha for all aspects (global)
  
OR
  - Use DIFFERENT alpha per aspect (per-aspect)
  
95% cases: GLOBAL is enough!
```

---

## 📚 Files

- Full explanation: `FOCAL_LOSS_MULTILABEL_EXPLAINED.md`
- Quick reference: `QUICK_FOCAL_MULTILABEL.md` (this file)

---

**Recommendation**: Dùng **GLOBAL ALPHA ("auto")** cho multi-label, giống single-label!
