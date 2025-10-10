import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

# Load dataset
df = pd.read_csv('dataset.csv', encoding='utf-8-sig', nrows=10)

print("="*70)
print("KIỂM TRA WORD SEGMENTATION TRONG DATASET")
print("="*70)

print("\n5 câu mẫu:")
for i, sentence in enumerate(df['data'].head(5), 1):
    print(f"{i}. {sentence}")

# Count underscores
underscore_count = df['data'].astype(str).str.count('_').sum()
total_sentences = len(df)

print(f"\n📊 Thống kê:")
print(f"   Tổng số câu kiểm tra: {total_sentences}")
print(f"   Số underscores: {underscore_count}")

if underscore_count > 0:
    print(f"\n✅ Dataset ĐÃ được word-segment (có underscores)")
    print(f"   → PhoBERT: GIỮ underscores")
    print(f"   → ViSoBERT: CẦN remove underscores")
else:
    print(f"\n⚠️  Dataset CHƯA được word-segment (không có underscores)")
    print(f"   → PhoBERT: CẦN thêm word segmentation")
    print(f"   → ViSoBERT: Dùng trực tiếp")
