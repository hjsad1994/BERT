"""
Script Chuẩn Bị Dữ Liệu ABSA
============================
Chuyển đổi dataset từ format multi-label (nhiều aspect trên một dòng)
sang format single-label (một mẫu cho mỗi cặp sentence-aspect)

Input: dataset.csv (format gốc với nhiều cột aspect)
Output: 
    - data/train.csv
    - data/validation.csv  
    - data/test.csv
    
Format output: sentence, aspect, sentiment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from collections import Counter


class ABSADataPreparator:
    """Class để chuẩn bị dữ liệu ABSA"""
    
    # Danh sách các khía cạnh hợp lệ từ dataset gốc
    VALID_ASPECTS = [
        'Battery', 'Camera', 'Performance', 'Display', 'Design',
        'Software', 'Packaging', 'Price', 'Audio', 'Warranty', 'Shop_Service',
        'Shipping', 'General', 'Others'
    ]
    
    # Mapping sentiment từ format gốc sang chuẩn hóa
    SENTIMENT_MAPPING = {
        'Positive': 'positive',
        'Negative': 'negative',
        'Neutral': 'neutral',
        'positive': 'positive',
        'negative': 'negative',
        'neutral': 'neutral'
    }
    
    def __init__(self, input_file, output_dir='data', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """
        Khởi tạo ABSADataPreparator
        
        Args:
            input_file: Đường dẫn file CSV input (multi-label format)
            output_dir: Thư mục lưu output files
            train_ratio: Tỷ lệ tập train
            val_ratio: Tỷ lệ tập validation
            test_ratio: Tỷ lệ tập test
            random_seed: Random seed cho reproducibility
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Tỷ lệ phải tổng bằng 1.0, nhận được {train_ratio + val_ratio + test_ratio}")
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        """Tải dataset từ file CSV"""
        print(f"\n{'='*70}")
        print(f"📁 Đang tải dataset từ: {self.input_file}")
        print(f"{'='*70}")
        
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Không tìm thấy file: {self.input_file}")
        
        # Đọc CSV với encoding UTF-8-sig để xử lý BOM
        self.df = pd.read_csv(self.input_file, encoding='utf-8-sig')
        
        print(f"✓ Kích thước dataset: {self.df.shape}")
        print(f"✓ Các cột: {', '.join(self.df.columns)}")
        print(f"✓ Tổng số dòng: {len(self.df)}")
        
        # Kiểm tra cột 'data' (chứa câu văn)
        if 'data' not in self.df.columns:
            raise ValueError("Dataset phải có cột 'data' chứa câu văn")
        
        # Xử lý VNCoreNLP segmentation: remove underscores cho BERT tokenizer
        underscore_count = self.df['data'].astype(str).str.count('_').sum()
        if underscore_count > 0:
            print(f"\n🔧 Phát hiện {underscore_count:,} underscores (VNCoreNLP segmentation)")
            print(f"🔧 Đang chuyển đổi để tương thích với BERT tokenizer: 'Chăm_sóc' → 'Chăm sóc'")
            self.df['data'] = self.df['data'].astype(str).str.replace('_', ' ', regex=False)
            print(f"✓ Đã xử lý VNCoreNLP segmentation (BERT-friendly format)")
        
        # Kiểm tra các cột aspect
        found_aspects = [col for col in self.VALID_ASPECTS if col in self.df.columns]
        print(f"✓ Tìm thấy {len(found_aspects)} khía cạnh: {', '.join(found_aspects)}")
        
        if len(found_aspects) == 0:
            raise ValueError("Không tìm thấy cột aspect nào trong dataset")
        
        self.aspect_columns = found_aspects
        
        return self
    
    def clean_data(self):
        """Làm sạch dữ liệu"""
        print(f"\n{'='*70}")
        print("🧹 Đang làm sạch dữ liệu...")
        print(f"{'='*70}")
        
        initial_size = len(self.df)
        
        # Loại bỏ các dòng có câu văn rỗng
        self.df = self.df[self.df['data'].notna() & (self.df['data'].str.strip() != '')]
        
        # Làm sạch khoảng trắng trong câu văn
        self.df['data'] = self.df['data'].str.strip()
        
        final_size = len(self.df)
        removed = initial_size - final_size
        
        print(f"✓ Dòng ban đầu: {initial_size}")
        print(f"✓ Dòng sau khi làm sạch: {final_size}")
        if removed > 0:
            print(f"✓ Đã loại bỏ: {removed} dòng có câu văn rỗng")
        
        return self
    
    def convert_to_single_label(self):
        """Chuyển đổi từ multi-label sang single-label format"""
        print(f"\n{'='*70}")
        print("🔄 Đang chuyển đổi sang format ABSA single-label...")
        print(f"{'='*70}")
        
        absa_samples = []
        skipped_count = 0
        
        # Lặp qua từng dòng trong dataset
        for idx, row in self.df.iterrows():
            sentence = row['data']
            
            # Lặp qua từng aspect column
            for aspect in self.aspect_columns:
                sentiment_value = row[aspect]
                
                # Bỏ qua nếu aspect không được đề cập (NaN hoặc empty)
                if pd.isna(sentiment_value) or str(sentiment_value).strip() == '':
                    continue
                
                sentiment_str = str(sentiment_value).strip()
                
                # Chuẩn hóa sentiment
                if sentiment_str in self.SENTIMENT_MAPPING:
                    normalized_sentiment = self.SENTIMENT_MAPPING[sentiment_str]
                else:
                    # Bỏ qua các giá trị sentiment không hợp lệ
                    skipped_count += 1
                    continue
                
                # Thêm mẫu ABSA
                absa_samples.append({
                    'sentence': sentence,
                    'aspect': aspect,
                    'sentiment': normalized_sentiment
                })
        
        # Tạo DataFrame mới
        self.absa_df = pd.DataFrame(absa_samples)
        
        print(f"✓ Số dòng gốc: {len(self.df)}")
        print(f"✓ Số mẫu ABSA được tạo: {len(self.absa_df)}")
        print(f"✓ Trung bình aspects/câu: {len(self.absa_df)/len(self.df):.2f}")
        if skipped_count > 0:
            print(f"✓ Bỏ qua {skipped_count} giá trị sentiment không hợp lệ")
        
        return self
    
    def analyze_distribution(self):
        """Phân tích phân bố dữ liệu"""
        print(f"\n{'='*70}")
        print("📊 Phân tích phân bố dữ liệu...")
        print(f"{'='*70}")
        
        # Phân bố theo sentiment
        print("\n1. Phân bố theo Sentiment:")
        sentiment_counts = self.absa_df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = count / len(self.absa_df) * 100
            print(f"   {sentiment:>10}: {count:>6} mẫu ({percentage:>5.2f}%)")
        
        # Phân bố theo aspect
        print("\n2. Phân bố theo Aspect:")
        aspect_counts = self.absa_df['aspect'].value_counts()
        for aspect, count in aspect_counts.items():
            percentage = count / len(self.absa_df) * 100
            print(f"   {aspect:>15}: {count:>6} mẫu ({percentage:>5.2f}%)")
        
        # Phân bố kết hợp
        print("\n3. Phân bố Sentiment theo từng Aspect:")
        for aspect in self.aspect_columns:
            aspect_data = self.absa_df[self.absa_df['aspect'] == aspect]
            if len(aspect_data) > 0:
                print(f"\n   {aspect}:")
                sentiment_dist = aspect_data['sentiment'].value_counts()
                for sentiment, count in sentiment_dist.items():
                    percentage = count / len(aspect_data) * 100
                    print(f"      {sentiment:>10}: {count:>5} ({percentage:>5.1f}%)")
        
        return self
    
    def stratified_split(self):
        """Chia dữ liệu với stratified sampling để đảm bảo phân bố cân bằng"""
        print(f"\n{'='*70}")
        print("✂️  Đang chia dữ liệu với stratified sampling...")
        print(f"{'='*70}")
        
        # Set random seed
        np.random.seed(self.random_seed)
        
        # Tạo stratify label bằng cách kết hợp aspect và sentiment
        self.absa_df['stratify_label'] = self.absa_df['aspect'] + '_' + self.absa_df['sentiment']
        
        # Split 1: train+val vs test
        train_val_df, test_df = train_test_split(
            self.absa_df,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=self.absa_df['stratify_label'],
            shuffle=True
        )
        
        # Split 2: train vs val
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=self.random_seed,
            stratify=train_val_df['stratify_label'],
            shuffle=True
        )
        
        # Xóa cột stratify_label không cần thiết
        train_df = train_df[['sentence', 'aspect', 'sentiment']].reset_index(drop=True)
        val_df = val_df[['sentence', 'aspect', 'sentiment']].reset_index(drop=True)
        test_df = test_df[['sentence', 'aspect', 'sentiment']].reset_index(drop=True)
        
        # Lưu trữ splits
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        # In thống kê
        total = len(self.absa_df)
        print(f"\n✓ Thống kê phân chia:")
        print(f"   Train:      {len(self.train_df):>6} mẫu ({len(self.train_df)/total*100:>5.1f}%)")
        print(f"   Validation: {len(self.val_df):>6} mẫu ({len(self.val_df)/total*100:>5.1f}%)")
        print(f"   Test:       {len(self.test_df):>6} mẫu ({len(self.test_df)/total*100:>5.1f}%)")
        print(f"   Tổng:       {total:>6} mẫu")
        
        return self
    
    def validate_splits(self):
        """Kiểm tra tính hợp lệ của các splits"""
        print(f"\n{'='*70}")
        print("✅ Đang kiểm tra tính hợp lệ của các splits...")
        print(f"{'='*70}")
        
        # Kiểm tra sentiment distribution trong từng split
        print("\nPhân bố Sentiment trong các splits:")
        for split_name, split_df in [('Train', self.train_df), ('Val', self.val_df), ('Test', self.test_df)]:
            print(f"\n{split_name}:")
            sentiment_dist = split_df['sentiment'].value_counts(normalize=True) * 100
            for sentiment, percentage in sentiment_dist.items():
                print(f"   {sentiment:>10}: {percentage:>5.1f}%")
        
        # Kiểm tra aspect distribution trong từng split
        print("\nPhân bố Aspect trong các splits:")
        for split_name, split_df in [('Train', self.train_df), ('Val', self.val_df), ('Test', self.test_df)]:
            print(f"\n{split_name}:")
            aspect_dist = split_df['aspect'].value_counts(normalize=True) * 100
            for aspect, percentage in aspect_dist.items():
                print(f"   {aspect:>15}: {percentage:>5.1f}%")
        
        return self
    
    def save_splits(self):
        """Lưu các splits thành CSV files"""
        print(f"\n{'='*70}")
        print("💾 Đang lưu các splits...")
        print(f"{'='*70}")
        
        # Define output paths
        train_path = os.path.join(self.output_dir, 'train.csv')
        val_path = os.path.join(self.output_dir, 'validation.csv')
        test_path = os.path.join(self.output_dir, 'test.csv')
        
        # Lưu thành CSV với UTF-8 encoding
        self.train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
        self.val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
        self.test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
        
        # In thông tin file
        print(f"\n✓ Files đã được lưu thành công:")
        print(f"\n   📄 Train:      {train_path}")
        print(f"      Kích thước: {os.path.getsize(train_path) / 1024:.2f} KB")
        print(f"      Số mẫu:     {len(self.train_df)}")
        
        print(f"\n   📄 Validation: {val_path}")
        print(f"      Kích thước: {os.path.getsize(val_path) / 1024:.2f} KB")
        print(f"      Số mẫu:     {len(self.val_df)}")
        
        print(f"\n   📄 Test:       {test_path}")
        print(f"      Kích thước: {os.path.getsize(test_path) / 1024:.2f} KB")
        print(f"      Số mẫu:     {len(self.test_df)}")
        
        return self
    
    def save_metadata(self):
        """Lưu metadata về quá trình chuẩn bị dữ liệu"""
        import json
        from datetime import datetime
        
        metadata = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_file': self.input_file,
            'output_directory': self.output_dir,
            'random_seed': self.random_seed,
            'split_ratios': {
                'train': self.train_ratio,
                'validation': self.val_ratio,
                'test': self.test_ratio
            },
            'split_sizes': {
                'train': len(self.train_df),
                'validation': len(self.val_df),
                'test': len(self.test_df),
                'total': len(self.absa_df)
            },
            'aspects': self.aspect_columns,
            'sentiments': ['positive', 'negative', 'neutral'],
            'format': {
                'columns': ['sentence', 'aspect', 'sentiment'],
                'description': 'Single-label ABSA format: mỗi dòng chứa một cặp (sentence, aspect) và sentiment tương ứng'
            }
        }
        
        metadata_path = os.path.join(self.output_dir, 'data_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Metadata đã được lưu: {metadata_path}")
        
        return self
    
    def run(self):
        """Thực thi toàn bộ pipeline chuẩn bị dữ liệu"""
        try:
            self.load_data()
            self.clean_data()
            self.convert_to_single_label()
            self.analyze_distribution()
            self.stratified_split()
            self.validate_splits()
            self.save_splits()
            self.save_metadata()
            
            print(f"\n{'='*70}")
            print("🎉 [THÀNH CÔNG] Chuẩn bị dữ liệu hoàn tất!")
            print(f"{'='*70}")
            print(f"\n✓ Output files đã được lưu tại: {os.path.abspath(self.output_dir)}/")
            print(f"\n✓ Bạn có thể bắt đầu huấn luyện bằng lệnh:")
            print(f"   python train.py --config config.yaml")
            
            return self
            
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"❌ [LỖI] Đã xảy ra lỗi trong quá trình chuẩn bị dữ liệu!")
            print(f"{'='*70}")
            print(f"Chi tiết lỗi: {str(e)}")
            raise


def main():
    """Hàm main"""
    import sys
    import io
    
    # Thiết lập UTF-8 cho console output trên Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # Cấu hình
    INPUT_FILE = 'dataset.csv'
    OUTPUT_DIR = 'data'
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    
    print("\n" + "="*70)
    print("🚀 ABSA DATA PREPARATION PIPELINE")
    print("="*70)
    print(f"\nCấu hình:")
    print(f"  Input file:     {INPUT_FILE}")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  Train ratio:    {TRAIN_RATIO:.0%}")
    print(f"  Val ratio:      {VAL_RATIO:.0%}")
    print(f"  Test ratio:     {TEST_RATIO:.0%}")
    print(f"  Random seed:    {RANDOM_SEED}")
    
    # Tạo preparator và chạy
    preparator = ABSADataPreparator(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED
    )
    
    preparator.run()


if __name__ == '__main__':
    main()
