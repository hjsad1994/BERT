# KẾ HOẠCH CLEAN DỮ LIỆU CHO BÀI TOÁN ABSA

## PHẦN 1: CÁC BƯỚC CLEAN CƠ BẢN

### 1.1. Xử lý câu quá ngắn/dài
```python
# Loại bỏ câu quá ngắn (<10 ký tự)
df = df[df['sentence'].str.len() >= 10]

# Xử lý câu quá dài (>500 ký tự)
# Option 1: Cắt bớt
df.loc[df['sentence'].str.len() > 500, 'sentence'] = df['sentence'].str[:500]

# Option 2: Loại bỏ (nếu không ảnh hưởng nhiều)
df = df[df['sentence'].str.len() <= 500]
```

### 1.2. Loại bỏ dữ liệu trùng lặp
```python
# Loại bỏ trùng lặp hoàn toàn
df = df.drop_duplicates()

# Hoặc loại trùng theo sentence + aspect
df = df.drop_duplicates(subset=['sentence', 'aspect'], keep='first')
```

### 1.3. Làm sạch text
```python
import re

def clean_text(text):
    # Loại bỏ emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Loại bỏ ký tự đặc biệt không cần thiết
    text = re.sub(r'[^\w\s,.\-àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', ' ', text)
    
    # Chuẩn hóa khoảng trắng
    text = ' '.join(text.split())
    
    # Chuyển về lowercase (tùy chọn, phụ thuộc vào model)
    # text = text.lower()
    
    return text

df['sentence'] = df['sentence'].apply(clean_text)
```

### 1.4. Kiểm tra và sửa label
```python
# Kiểm tra aspect hợp lệ
valid_aspects = ['Shipping', 'General', 'Battery', 'Shop_Service', 
                 'Performance', 'Design', 'Price', 'Packaging', 
                 'Camera', 'Display', 'Software', 'Audio', 'Others']
df = df[df['aspect'].isin(valid_aspects)]

# Kiểm tra sentiment hợp lệ
valid_sentiments = ['positive', 'negative', 'neutral']
df = df[df['sentiment'].isin(valid_sentiments)]

# Xử lý aspect "Others" - xem xét gán lại aspect cụ thể hơn
# hoặc loại bỏ nếu không có giá trị
df = df[df['aspect'] != 'Others']  # Tùy chọn
```

## PHẦN 2: XỬ LÝ IMBALANCE (QUAN TRỌNG)

### 2.1. Xử lý Aspect Imbalance

#### Option 1: Under-sampling (Giảm mẫu nhiều)
```python
from sklearn.utils import resample

# Xác định số mẫu tối đa cho mỗi aspect
max_samples_per_aspect = 800  # hoặc median của distribution

balanced_dfs = []
for aspect in df['aspect'].unique():
    aspect_df = df[df['aspect'] == aspect]
    
    if len(aspect_df) > max_samples_per_aspect:
        # Down-sample
        aspect_df = resample(aspect_df, 
                           n_samples=max_samples_per_aspect,
                           random_state=42)
    
    balanced_dfs.append(aspect_df)

df_balanced = pd.concat(balanced_dfs).reset_index(drop=True)
```

#### Option 2: Over-sampling (Tăng mẫu ít) - KHUYẾN NGHỊ
```python
# Xác định số mẫu tối thiểu
min_samples_per_aspect = 500  # hoặc mean của distribution

balanced_dfs = []
for aspect in df['aspect'].unique():
    aspect_df = df[df['aspect'] == aspect]
    
    if len(aspect_df) < min_samples_per_aspect:
        # Up-sample với replacement
        aspect_df = resample(aspect_df, 
                           n_samples=min_samples_per_aspect,
                           replace=True,
                           random_state=42)
    
    balanced_dfs.append(aspect_df)

df_balanced = pd.concat(balanced_dfs).reset_index(drop=True)
```

#### Option 3: SMOTE-like cho text (Nâng cao)
```python
# Sử dụng back-translation hoặc paraphrasing
# để tạo synthetic data cho các aspect ít mẫu

from googletrans import Translator  # hoặc dùng model paraphrase

def augment_text(text, n_augments=2):
    """Back-translation: VI -> EN -> VI"""
    translator = Translator()
    augmented = []
    
    # Translate to English
    en_text = translator.translate(text, src='vi', dest='en').text
    # Translate back to Vietnamese
    vi_text = translator.translate(en_text, src='en', dest='vi').text
    augmented.append(vi_text)
    
    return augmented

# Áp dụng cho các aspect ít mẫu
```

### 2.2. Xử lý Sentiment Imbalance

#### Option 1: Weighted Loss trong training
```python
from sklearn.utils.class_weight import compute_class_weight

# Tính class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['sentiment']),
    y=df['sentiment']
)

# Sử dụng trong trainer (xem oversampling_utils.py)
```

#### Option 2: Aspect-wise Over-sampling (Đã có trong code)
```python
# Code này đã có trong oversampling_utils.py
# Oversample minority sentiment cho từng aspect
def aspect_wise_oversample(df, target_samples_per_class=None):
    balanced_dfs = []
    
    for aspect in df['aspect'].unique():
        aspect_df = df[df['aspect'] == aspect]
        
        # Oversample trong từng aspect
        for sentiment in ['positive', 'negative', 'neutral']:
            sent_df = aspect_df[aspect_df['sentiment'] == sentiment]
            
            if len(sent_df) < target_samples_per_class:
                sent_df = resample(sent_df, 
                                 n_samples=target_samples_per_class,
                                 replace=True,
                                 random_state=42)
            
            balanced_dfs.append(sent_df)
    
    return pd.concat(balanced_dfs).reset_index(drop=True)
```

## PHẦN 3: DATA AUGMENTATION (Tùy chọn)

### 3.1. Synonym Replacement
```python
# Thay thế từ đồng nghĩa (cần Vietnamese WordNet hoặc dictionary)
import random

synonyms_dict = {
    'tốt': ['tuyệt', 'ổn', 'ngon', 'xuất sắc'],
    'xấu': ['tệ', 'kém', 'dở'],
    # ... more synonyms
}

def synonym_replacement(text, n=2):
    words = text.split()
    for _ in range(n):
        # Random replace
        for i, word in enumerate(words):
            if word in synonyms_dict:
                words[i] = random.choice(synonyms_dict[word])
    return ' '.join(words)
```

### 3.2. Random Insertion
```python
def random_insertion(text, n=1):
    words = text.split()
    for _ in range(n):
        # Insert random word from sentence
        random_word = random.choice(words)
        random_idx = random.randint(0, len(words))
        words.insert(random_idx, random_word)
    return ' '.join(words)
```

### 3.3. Random Swap
```python
def random_swap(text, n=1):
    words = text.split()
    for _ in range(n):
        if len(words) >= 2:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)
```

## PHẦN 4: KIỂM TRA CHẤT LƯỢNG SAU CLEAN

### 4.1. Kiểm tra phân bố
```python
print("=== PHÂN BỐ SAU CLEAN ===")
print("\nAspect distribution:")
print(df_cleaned['aspect'].value_counts())

print("\nSentiment distribution:")
print(df_cleaned['sentiment'].value_counts())

print("\nAspect-Sentiment crosstab:")
print(pd.crosstab(df_cleaned['aspect'], df_cleaned['sentiment']))
```

### 4.2. Kiểm tra độ dài
```python
df_cleaned['length'] = df_cleaned['sentence'].str.len()
print("\n=== ĐỘ DÀI CÂU ===")
print(f"Mean: {df_cleaned['length'].mean():.2f}")
print(f"Min: {df_cleaned['length'].min()}")
print(f"Max: {df_cleaned['length'].max()}")
print(f"Median: {df_cleaned['length'].median()}")
```

### 4.3. Sample check
```python
print("\n=== SAMPLE KIỂM TRA ===")
for aspect in df_cleaned['aspect'].unique()[:3]:
    print(f"\nAspect: {aspect}")
    samples = df_cleaned[df_cleaned['aspect'] == aspect].head(2)
    for _, row in samples.iterrows():
        print(f"  - {row['sentence'][:80]}... | {row['sentiment']}")
```

## PHẦN 5: LƯU DỮ LIỆU ĐÃ CLEAN

```python
# Lưu dữ liệu đã clean
df_cleaned.to_csv('data/train_cleaned.csv', index=False, encoding='utf-8')
df_test_cleaned.to_csv('data/test_cleaned.csv', index=False, encoding='utf-8')
df_val_cleaned.to_csv('data/validation_cleaned.csv', index=False, encoding='utf-8')

print("✅ Đã lưu dữ liệu cleaned!")
```

## KHUYẾN NGHỊ

### 1. Ưu tiên cao:
- ✅ Loại bỏ câu quá ngắn (<10 ký tự)
- ✅ Làm sạch emoji và ký tự đặc biệt
- ✅ Xử lý imbalance bằng aspect-wise oversampling
- ✅ Loại bỏ duplicates

### 2. Ưu tiên trung bình:
- Xử lý câu quá dài (truncate hoặc remove)
- Data augmentation cho các aspect ít mẫu
- Review và fix label sai (đặc biệt aspect "Others")

### 3. Tùy chọn:
- Lowercase transformation
- Stemming/Lemmatization (ít cần thiết với BERT)
- Advanced augmentation (back-translation)

### 4. Lưu ý khi training:
- Sử dụng class weights trong loss function
- Monitor metrics theo từng aspect riêng biệt
- Sử dụng stratified split khi chia data
- Consider focal loss cho imbalanced classes
