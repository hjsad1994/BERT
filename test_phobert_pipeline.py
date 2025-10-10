"""
Script test PhoBERT pipeline với word segmentation
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from transformers import AutoTokenizer

print("="*70)
print("TEST PHOBERT TOKENIZER VỚI WORD SEGMENTATION")
print("="*70)

# Test sentences (word-segmented)
test_sentences = [
    "sản_phẩm tuyệt_vời , đúng với mô_tả",
    "pin yếu , camera sau chụp ảnh ngã vàng",
    "giao hàng nhanh_chóng , shipper thân_thiện"
]

# Load PhoBERT tokenizer
print("\n1. Loading PhoBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
print("✓ Tokenizer loaded")

print("\n2. Testing tokenization với word-segmented text:")
for i, sentence in enumerate(test_sentences, 1):
    print(f"\n   Sentence {i}: {sentence}")
    
    # Tokenize
    tokens = tokenizer.tokenize(sentence)
    print(f"   Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"   Tokens: {tokens}")
    
    # Encode
    encoded = tokenizer(sentence, padding=False, truncation=True, max_length=256)
    print(f"   Input IDs length: {len(encoded['input_ids'])}")

print("\n3. Testing pair tokenization (sentence + aspect):")
sentence = "sản_phẩm tuyệt_vời , pin tốt"
aspect = "Battery"
print(f"   Sentence: {sentence}")
print(f"   Aspect: {aspect}")

encoded = tokenizer(sentence, aspect, padding=False, truncation=True, max_length=256)
tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
print(f"   Tokens: {tokens}")
print(f"   Length: {len(encoded['input_ids'])}")

print("\n4. Kiểm tra dataset đã prepare:")
try:
    df = pd.read_csv('data/train.csv', encoding='utf-8-sig', nrows=3)
    print("   ✓ File data/train.csv tồn tại")
    print(f"   ✓ Columns: {list(df.columns)}")
    
    # Check if underscores preserved
    sample_sentence = df['sentence'].iloc[0]
    underscore_count = sample_sentence.count('_')
    
    print(f"\n   Sample sentence: {sample_sentence[:80]}...")
    print(f"   Underscores in sample: {underscore_count}")
    
    if underscore_count > 0:
        print("   ✅ Word segmentation ĐÃ được giữ nguyên (PhoBERT-ready)")
    else:
        print("   ⚠️  Không có underscores (có thể cần chạy lại prepare_data.py)")
        
except FileNotFoundError:
    print("   ⚠️  Chưa có data/train.csv, cần chạy: python prepare_data.py")

print("\n" + "="*70)
print("✅ TEST HOÀN TẤT")
print("="*70)
