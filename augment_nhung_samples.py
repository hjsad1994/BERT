"""
Data Augmentation cho các samples có từ chuyển ý "nhưng"
Oversampling và tạo thêm variants để model học tốt hơn
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import os
import random

def augment_nhung_samples(train_file='data/train.csv', output_file='data/train_augmented_nhung.csv', oversample_factor=2):
    """
    Augment training data với oversampling cho samples có 'nhưng'
    
    Args:
        train_file: File training data gốc
        output_file: File output sau augmentation
        oversample_factor: Số lần nhân bản samples có 'nhưng' (mặc định: 3x)
    """
    print("="*80)
    print("🔄 DATA AUGMENTATION CHO SAMPLES CÓ 'NHƯNG'")
    print("="*80)
    
    os.chdir('D:/BERT')
    
    # Load training data
    if not os.path.exists(train_file):
        print(f"❌ File không tồn tại: {train_file}")
        return
    
    df = pd.read_csv(train_file, encoding='utf-8-sig')
    
    print(f"\n📊 Training data gốc: {len(df)} samples")
    
    # Find samples with "nhưng"
    nhung_samples = df[df['sentence'].str.contains('nhưng', case=False, na=False)]
    non_nhung_samples = df[~df['sentence'].str.contains('nhưng', case=False, na=False)]
    
    print(f"📊 Samples có 'nhưng': {len(nhung_samples)} ({len(nhung_samples)/len(df)*100:.1f}%)")
    print(f"📊 Samples không có 'nhưng': {len(non_nhung_samples)} ({len(non_nhung_samples)/len(df)*100:.1f}%)")
    
    # Oversample nhung samples
    print(f"\n🔄 Đang oversample samples có 'nhưng' (x{oversample_factor})...")
    
    oversampled_nhung = pd.concat([nhung_samples] * oversample_factor, ignore_index=True)
    
    # Combine
    augmented_df = pd.concat([non_nhung_samples, oversampled_nhung], ignore_index=True)
    
    # Shuffle
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✓ Augmented data: {len(augmented_df)} samples")
    print(f"✓ Tăng thêm: {len(augmented_df) - len(df)} samples (+{(len(augmented_df) - len(df))/len(df)*100:.1f}%)")
    
    # Save
    augmented_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Saved augmented data to: {output_file}")
    
    # Statistics
    print(f"\n{'='*80}")
    print("📊 THỐNG KÊ AUGMENTED DATA:")
    print(f"{'='*80}")
    
    new_nhung_samples = augmented_df[augmented_df['sentence'].str.contains('nhưng', case=False, na=False)]
    print(f"   • Tổng samples: {len(augmented_df)}")
    print(f"   • Samples có 'nhưng': {len(new_nhung_samples)} ({len(new_nhung_samples)/len(augmented_df)*100:.1f}%)")
    print(f"   • Samples không có 'nhưng': {len(augmented_df) - len(new_nhung_samples)} ({(len(augmented_df) - len(new_nhung_samples))/len(augmented_df)*100:.1f}%)")
    
    # Sentiment distribution
    print(f"\n📊 PHÂN BỐ SENTIMENT:")
    sentiment_dist = augmented_df['sentiment'].value_counts()
    for sent, count in sentiment_dist.items():
        print(f"   • {sent:<10}: {count:>5} samples ({count/len(augmented_df)*100:.1f}%)")
    
    # Aspect distribution
    print(f"\n📊 TOP 10 ASPECTS:")
    aspect_dist = augmented_df['aspect'].value_counts().head(10)
    for asp, count in aspect_dist.items():
        print(f"   • {asp:<15}: {count:>5} samples ({count/len(augmented_df)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("🎯 NEXT STEPS:")
    print(f"{'='*80}")
    print("\n1. Update config.yaml để sử dụng augmented data:")
    print("   train_file: data/train_augmented_nhung.csv")
    print("\n2. Retrain model:")
    print("   python train.py")
    print("\n3. So sánh performance:")
    print("   - Trước: ~79.57% accuracy trên câu có 'nhưng'")
    print("   - Mục tiêu: Tăng lên ~85-88% (gần với overall 91.34%)")
    print(f"\n{'='*80}\n")
    
    return augmented_df


def create_advanced_augmentation(train_file='data/train.csv', output_file='data/train_augmented_nhung_advanced.csv'):
    """
    Advanced augmentation với các kỹ thuật phức tạp hơn
    """
    print("="*80)
    print("🔄 ADVANCED DATA AUGMENTATION CHO SAMPLES CÓ 'NHƯNG'")
    print("="*80)
    
    os.chdir('D:/BERT')
    
    # Load training data
    if not os.path.exists(train_file):
        print(f"❌ File không tồn tại: {train_file}")
        return
    
    df = pd.read_csv(train_file, encoding='utf-8-sig')
    
    print(f"\n📊 Training data gốc: {len(df)} samples")
    
    # Find samples with adversative conjunctions
    adversative_words = ['nhưng', 'tuy nhiên', 'mặc dù', 'song', 'nhưng mà', 'tuy vậy']
    
    pattern = '|'.join(adversative_words)
    adv_samples = df[df['sentence'].str.contains(pattern, case=False, na=False)]
    non_adv_samples = df[~df['sentence'].str.contains(pattern, case=False, na=False)]
    
    print(f"📊 Samples có từ chuyển ý: {len(adv_samples)} ({len(adv_samples)/len(df)*100:.1f}%)")
    
    # Strategy 1: Oversample 2x
    oversampled = pd.concat([adv_samples] * 2, ignore_index=True)
    
    # Strategy 2: Add [ADV] marker (optional - comment out if not using)
    # for idx, row in oversampled.iterrows():
    #     for word in adversative_words:
    #         if word in row['sentence'].lower():
    #             oversampled.at[idx, 'sentence'] = row['sentence'].replace(word, f"[ADV] {word}")
    #             break
    
    # Combine
    augmented_df = pd.concat([non_adv_samples, oversampled], ignore_index=True)
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✓ Advanced augmented data: {len(augmented_df)} samples")
    print(f"✓ Tăng thêm: {len(augmented_df) - len(df)} samples (+{(len(augmented_df) - len(df))/len(df)*100:.1f}%)")
    
    # Save
    augmented_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Saved advanced augmented data to: {output_file}")
    
    return augmented_df


if __name__ == '__main__':
    # Basic augmentation (recommended)
    print("\n🎯 OPTION 1: BASIC AUGMENTATION (Khuyến nghị)")
    print("-"*80)
    augment_nhung_samples(oversample_factor=2)
    
    print("\n\n")
    
    # Advanced augmentation (experimental)
    print("🎯 OPTION 2: ADVANCED AUGMENTATION (Thử nghiệm)")
    print("-"*80)
    create_advanced_augmentation()
    
    print("\n\n")
    print("="*80)
    print("✅ HOÀN TẤT DATA AUGMENTATION!")
    print("="*80)
    print("\n💡 Chọn một trong hai file để train:")
    print("   1. data/train_augmented_nhung.csv (basic - khuyến nghị)")
    print("   2. data/train_augmented_nhung_advanced.csv (advanced - thử nghiệm)")
    print("\nUpdate config.yaml rồi chạy:")
    print("   python train.py")
    print("\n" + "="*80 + "\n")
