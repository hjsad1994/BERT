"""
Error Analysis - Phân Tích Lỗi Chi Tiết
======================================
Phân tích các dự đoán sai của model để tìm patterns và cải thiện

Features:
    1. Tìm các predictions sai
    2. Phân tích theo aspect
    3. Phân tích theo sentiment
    4. Phân tích confusion patterns
    5. Tìm hard cases
    6. Đề xuất cải thiện
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class ErrorAnalyzer:
    """Class để phân tích lỗi của model"""
    
    # Label mappings
    ID2LABEL = {0: 'positive', 1: 'negative', 2: 'neutral'}
    LABEL2ID = {'positive': 0, 'negative': 1, 'neutral': 2}
    VALID_LABELS = ['positive', 'negative', 'neutral']
    
    def __init__(self, test_file='multi_label/data/test_multilabel.csv', 
                 predictions_file='multi_label/models/multilabel_focal_contrastive/test_predictions_detailed.csv'):
        """
        Khởi tạo ErrorAnalyzer
        
        Args:
            test_file: File CSV chứa ground truth
            predictions_file: File CSV chứa predictions của model
        """
        self.test_file = test_file
        self.predictions_file = predictions_file
        self.output_dir = 'multi_label/error_analysis_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data storage
        self.df = None
        self.test_df = None
        self.pred_df = None
    
    def normalize_value(self, value):
        """Normalize a value to lowercase string sentiment label"""
        if isinstance(value, str):
            value = value.strip().lower()
            return value if value else None
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)) and not pd.isna(value):
            return self.ID2LABEL.get(int(value), None)
        value = str(value).strip().lower()
        return value if value else None
    
    def _load_numeric_format_predictions(self, pred_raw, test_wide):
        """Load predictions from numeric format (test_predictions_detailed.csv)"""
        min_len = min(len(pred_raw), len(test_wide))
        pred_raw = pred_raw.iloc[:min_len]
        test_wide = test_wide.iloc[:min_len]
        
        pred_wide = pd.DataFrame()
        pred_wide['data'] = test_wide['data'].values
        
        true_wide = pd.DataFrame()
        true_wide['data'] = test_wide['data'].values
        
        # Extract aspects from column names
        aspects = sorted(set(
            col.replace('_pred', '') 
            for col in pred_raw.columns 
            if col.endswith('_pred')
        ))
        
        print(f"✓ Found {len(aspects)} aspects: {', '.join(aspects)}")
        
        # Convert predictions and true labels
        for aspect in aspects:
            pred_col = f"{aspect}_pred"
            
            # Predictions: convert numeric to string
            if pred_col in pred_raw.columns:
                pred_wide[aspect] = pred_raw[pred_col].map(self.ID2LABEL).fillna('neutral')
            else:
                pred_wide[aspect] = 'neutral'
            
            # True labels: ALWAYS use test data as ground truth
            # This distinguishes real Neutral from unlabeled placeholder (NaN)
            if aspect in test_wide.columns:
                true_wide[aspect] = test_wide[aspect].apply(
                    lambda x: self.normalize_value(x) if pd.notna(x) and str(x).strip() != '' else None
                )
            else:
                true_wide[aspect] = None
        
        return pred_wide, true_wide, aspects
    
    def _load_string_format_predictions(self, pred_raw, test_wide):
        """Load predictions from string format"""
        pred_wide = pred_raw
        
        true_file = self.predictions_file.replace('.csv', '_true.csv')
        if os.path.exists(true_file):
            true_wide = pd.read_csv(true_file, encoding='utf-8-sig')
            print(f"✓ Loaded true labels: {len(true_wide)} sentences (wide format)")
        else:
            true_wide = test_wide.copy()
            print(f"⚠️  True labels file not found, using test data")
        
        aspects = [col for col in pred_wide.columns if col != 'data']
        print(f"✓ Found {len(aspects)} aspects: {', '.join(aspects)}")
        
        return pred_wide, true_wide, aspects
    
    def load_data(self):
        """Load test data và predictions, convert to long format"""
        print(f"\n{'='*70}")
        print("📁 Đang tải dữ liệu...")
        print(f"{'='*70}")
        
        # Load ground truth
        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Không tìm thấy file: {self.test_file}")
        
        test_wide = pd.read_csv(self.test_file, encoding='utf-8-sig')
        print(f"✓ Loaded test data: {len(test_wide)} sentences (wide format)")
        
        # Load predictions
        if not os.path.exists(self.predictions_file):
            raise FileNotFoundError(f"Không tìm thấy file: {self.predictions_file}")
        
        pred_raw = pd.read_csv(self.predictions_file, encoding='utf-8-sig')
        print(f"✓ Loaded predictions file: {len(pred_raw)} rows")
        
        # Detect format and load accordingly
        if 'sample_id' in pred_raw.columns and '_pred' in str(pred_raw.columns):
            print("✓ Detected numeric format predictions file")
            pred_wide, true_wide, aspects = self._load_numeric_format_predictions(pred_raw, test_wide)
        else:
            print("✓ Detected string format predictions file")
            pred_wide, true_wide, aspects = self._load_string_format_predictions(pred_raw, test_wide)
        
        print(f"✓ Loaded predictions: {len(pred_wide)} sentences (wide format)")
        
        # Convert to long format: only include labeled aspects
        long_data = []
        labeled_count = 0
        skipped_unlabeled = 0
        
        for idx in range(len(pred_wide)):
            text = pred_wide.iloc[idx]['data']
            
            for aspect in aspects:
                pred_val = pred_wide.iloc[idx][aspect] if aspect in pred_wide.columns else None
                true_val = true_wide.iloc[idx][aspect] if aspect in true_wide.columns else None
                
                true_sentiment = self.normalize_value(true_val)
                pred_sentiment = self.normalize_value(pred_val)
                
                # Skip unlabeled aspects
                if true_sentiment is None:
                    skipped_unlabeled += 1
                    continue
                
                # Default prediction if None
                if pred_sentiment is None:
                    pred_sentiment = 'neutral'
                
                long_data.append({
                    'data': text,
                    'aspect': aspect,
                    'sentiment': true_sentiment,
                    'predicted_sentiment': pred_sentiment,
                    'predicted_label_id': self.LABEL2ID.get(pred_sentiment, 2),
                    'correct': true_sentiment == pred_sentiment
                })
                labeled_count += 1
        
        self.df = pd.DataFrame(long_data)
        self.test_df = test_wide
        self.pred_df = pred_wide
        
        # Print summary
        accuracy = self.df['correct'].mean()
        total_aspects = len(test_wide) * len(aspects)
        sentiment_counts = self.df['sentiment'].value_counts().to_dict()
        
        print(f"✓ Converted to long format: {len(self.df)} predictions (labeled aspects)")
        print(f"   • Total aspects in dataset: {total_aspects:,}")
        print(f"   • Labeled aspects: {labeled_count:,} ({labeled_count/total_aspects*100:.1f}%)")
        print(f"     - Positive: {sentiment_counts.get('positive', 0):,}")
        print(f"     - Negative: {sentiment_counts.get('negative', 0):,}")
        print(f"     - Neutral: {sentiment_counts.get('neutral', 0):,}")
        print(f"   • Skipped unlabeled: {skipped_unlabeled:,} ({skipped_unlabeled/total_aspects*100:.1f}%)")
        print(f"✓ Overall accuracy (on labeled aspects): {accuracy:.2%}")
        print(f"✓ Total errors: {(~self.df['correct']).sum()} / {len(self.df)}")
        print(f"⚠️  NOTE: Includes all labeled aspects (positive/negative/neutral), excludes only unlabeled")
        
        return self
    
    def analyze_by_aspect(self):
        """Phân tích lỗi theo từng aspect"""
        print(f"\n{'='*70}")
        print("📊 PHÂN TÍCH THEO ASPECT (LABELED ASPECTS: POSITIVE/NEGATIVE/NEUTRAL)")
        print(f"{'='*70}")
        
        aspect_stats = []
        for aspect in sorted(self.df['aspect'].unique()):
            aspect_df = self.df[self.df['aspect'] == aspect]
            total = len(aspect_df)
            correct = aspect_df['correct'].sum()
            
            aspect_stats.append({
                'aspect': aspect,
                'total': total,
                'correct': correct,
                'errors': total - correct,
                'accuracy': correct / total if total > 0 else 0,
                'error_rate': 1 - (correct / total if total > 0 else 0)
            })
        
        aspect_stats_df = pd.DataFrame(aspect_stats).sort_values('error_rate', ascending=False)
        
        # Print table
        print(f"\n{'Aspect':<15} {'Total':<8} {'Correct':<8} {'Errors':<8} {'Accuracy':<10} {'Error Rate'}")
        print(f"{'-'*70}")
        for _, row in aspect_stats_df.iterrows():
            print(f"{row['aspect']:<15} {row['total']:<8} {row['correct']:<8} "
                  f"{row['errors']:<8} {row['accuracy']:<10.2%} {row['error_rate']:.2%}")
        
        aspect_stats_df.to_csv(f"{self.output_dir}/aspect_error_analysis.csv", index=False, encoding='utf-8-sig')
        print(f"\n✓ Saved: {self.output_dir}/aspect_error_analysis.csv")
        
        # Top 5 weakest
        print(f"\n🔴 TOP 5 ASPECTS YẾU NHẤT (Error Rate cao nhất):")
        for i, (_, row) in enumerate(aspect_stats_df.head(5).iterrows(), 1):
            print(f"   {i}. {row['aspect']:<15} Error Rate: {row['error_rate']:.2%} ({row['errors']}/{row['total']} errors)")
        
        return aspect_stats_df
    
    def analyze_by_sentiment(self):
        """Phân tích lỗi theo sentiment class"""
        print(f"\n{'='*70}")
        print("📊 PHÂN TÍCH THEO SENTIMENT")
        print(f"{'='*70}")
        
        # Filter to only valid labels
        df_filtered = self.df[
            self.df['sentiment'].isin(self.VALID_LABELS) & 
            self.df['predicted_sentiment'].isin(self.VALID_LABELS)
        ].copy()
        
        print(f"✓ Filtered data: {len(df_filtered)} samples (only positive/negative/neutral)")
        if len(df_filtered) < len(self.df):
            print(f"  ⚠️  Excluded {len(self.df) - len(df_filtered)} samples with invalid labels")
        
        # Calculate stats per sentiment
        sentiment_stats = []
        for sentiment in self.VALID_LABELS:
            sent_df = df_filtered[df_filtered['sentiment'] == sentiment]
            total = len(sent_df)
            correct = sent_df['correct'].sum()
            
            sentiment_stats.append({
                'sentiment': sentiment,
                'total': total,
                'correct': correct,
                'errors': total - correct,
                'accuracy': correct / total if total > 0 else 0,
                'error_rate': 1 - (correct / total if total > 0 else 0)
            })
        
        sentiment_stats_df = pd.DataFrame(sentiment_stats).sort_values('error_rate', ascending=False)
        
        # Print table
        print(f"\n{'Sentiment':<12} {'Total':<8} {'Correct':<8} {'Errors':<8} {'Accuracy':<10} {'Error Rate'}")
        print(f"{'-'*70}")
        for _, row in sentiment_stats_df.iterrows():
            print(f"{row['sentiment']:<12} {row['total']:<8} {row['correct']:<8} "
                  f"{row['errors']:<8} {row['accuracy']:<10.2%} {row['error_rate']:.2%}")
        
        # Confusion matrix
        print(f"\n📊 CONFUSION MATRIX (CHỈ 3 NHÃN: Positive/Negative/Neutral):")
        cm = confusion_matrix(df_filtered['sentiment'], df_filtered['predicted_sentiment'], 
                             labels=self.VALID_LABELS)
        
        print(f"\n{'':>12} {'Predicted →':>12}")
        print(f"{'True ↓':<12} {'Positive':<12} {'Negative':<12} {'Neutral':<12}")
        print(f"{'-'*60}")
        for i, true_label in enumerate(self.VALID_LABELS):
            row_str = f"{true_label:<12}"
            for j in range(3):
                row_str += f" {cm[i][j]:<12}"
            print(row_str)
        
        sentiment_stats_df.to_csv(f"{self.output_dir}/sentiment_error_analysis.csv", index=False, encoding='utf-8-sig')
        print(f"\n✓ Saved: {self.output_dir}/sentiment_error_analysis.csv")
        
        return sentiment_stats_df, cm
    
    def analyze_confusion_patterns(self):
        """Phân tích các confusion patterns chi tiết"""
        print(f"\n{'='*70}")
        print("🔍 PHÂN TÍCH CONFUSION PATTERNS")
        print(f"{'='*70}")
        
        errors_df = self.df[~self.df['correct']].copy()
        
        # Count confusion pairs
        confusion_counts = errors_df.groupby(['sentiment', 'predicted_sentiment']).size().reset_index(name='count')
        confusion_counts = confusion_counts.sort_values('count', ascending=False)
        
        print(f"\n🔴 CONFUSION PAIRS (Nhầm gì thành gì):")
        print(f"\n{'True':<12} {'→':<5} {'Predicted':<12} {'Count':<8} {'% of Errors'}")
        print(f"{'-'*60}")
        
        total_errors = len(errors_df)
        for _, row in confusion_counts.iterrows():
            pct = row['count'] / total_errors * 100
            print(f"{row['sentiment']:<12} {'→':<5} {row['predicted_sentiment']:<12} "
                  f"{row['count']:<8} {pct:.1f}%")
        
        # Analyze by aspect
        print(f"\n🔍 CONFUSION PATTERNS BY ASPECT:")
        aspect_confusion = defaultdict(lambda: defaultdict(int))
        
        for _, row in errors_df.iterrows():
            aspect = row['aspect']
            confusion_pair = f"{row['sentiment']} → {row['predicted_sentiment']}"
            aspect_confusion[aspect][confusion_pair] += 1
        
        for aspect in sorted(aspect_confusion.keys()):
            patterns = aspect_confusion[aspect]
            if len(patterns) > 0:
                print(f"\n   {aspect}:")
                for pattern, count in sorted(patterns.items(), key=lambda x: -x[1])[:3]:
                    print(f"      • {pattern}: {count} cases")
        
        confusion_counts.to_csv(f"{self.output_dir}/confusion_patterns.csv", index=False, encoding='utf-8-sig')
        print(f"\n✓ Saved: {self.output_dir}/confusion_patterns.csv")
        
        return confusion_counts
    
    def find_hard_cases(self, top_n=50):
        """Tìm các cases khó nhất"""
        print(f"\n{'='*70}")
        print(f"🔴 TÌM TOP {top_n} HARD CASES")
        print(f"{'='*70}")
        
        errors_df = self.df[~self.df['correct']].copy()
        
        # Most confused aspects
        aspect_errors = errors_df.groupby('aspect').size().reset_index(name='error_count')
        aspect_errors = aspect_errors.sort_values('error_count', ascending=False)
        
        print(f"\n📊 Aspects có NHIỀU LỖI NHẤT:")
        for i, (_, row) in enumerate(aspect_errors.head(5).iterrows(), 1):
            print(f"   {i}. {row['aspect']:<15} {row['error_count']} errors")
        
        # Sentences with multiple errors
        sentence_error_counts = errors_df.groupby('data').size().reset_index(name='error_count')
        sentence_error_counts = sentence_error_counts.sort_values('error_count', ascending=False)
        
        print(f"\n📝 Câu có NHIỀU LỖI NHẤT (confused across multiple aspects):")
        for i, (_, row) in enumerate(sentence_error_counts.head(10).iterrows(), 1):
            sentence = row['data']
            count = row['error_count']
            sent_errors = errors_df[errors_df['data'] == sentence]
            aspects = ', '.join(sent_errors['aspect'].unique())
            
            print(f"\n   {i}. [{count} errors] {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
            print(f"      Aspects: {aspects}")
            for _, err in sent_errors.iterrows():
                print(f"      • {err['aspect']}: {err['sentiment']} → {err['predicted_sentiment']}")
        
        # Save hard cases
        print(f"\n💾 Saving detailed error examples...")
        hard_cases = []
        for aspect in errors_df['aspect'].unique():
            aspect_errors_df = errors_df[errors_df['aspect'] == aspect]
            for _, row in aspect_errors_df.head(5).iterrows():
                hard_cases.append({
                    'sentence': row['data'],
                    'aspect': row['aspect'],
                    'true_sentiment': row['sentiment'],
                    'predicted_sentiment': row['predicted_sentiment'],
                    'confusion_type': f"{row['sentiment']} → {row['predicted_sentiment']}"
                })
        
        hard_cases_df = pd.DataFrame(hard_cases)
        hard_cases_df.to_csv(f"{self.output_dir}/hard_cases.csv", index=False, encoding='utf-8-sig')
        print(f"✓ Saved: {self.output_dir}/hard_cases.csv")
        
        return hard_cases_df
    
    def save_all_errors(self):
        """Lưu TẤT CẢ các errors để phân tích chi tiết"""
        print(f"\n{'='*70}")
        print("💾 LƯU TẤT CẢ ERRORS CHO PHÂN TÍCH CHI TIẾT")
        print(f"{'='*70}")
        
        errors_df = self.df[~self.df['correct']].copy()
        print(f"\n📊 Tổng số errors: {len(errors_df)}")
        
        # Add columns
        errors_df['confusion_type'] = errors_df.apply(
            lambda row: f"{row['sentiment']} → {row['predicted_sentiment']}", axis=1
        )
        
        sentence_error_counts = errors_df.groupby('data').size().reset_index(name='error_count_for_sentence')
        errors_df = errors_df.merge(sentence_error_counts, on='data', how='left')
        
        errors_df_sorted = errors_df.sort_values(['aspect', 'confusion_type', 'data'])
        
        # Select columns to save
        columns_to_save = ['data', 'aspect', 'sentiment', 'predicted_sentiment', 'confusion_type', 
                          'error_count_for_sentence']
        
        all_errors_path = f"{self.output_dir}/all_errors_detailed.csv"
        errors_df_sorted[columns_to_save].to_csv(all_errors_path, index=False, encoding='utf-8-sig')
        
        print(f"✓ Saved ALL {len(errors_df)} errors to: {all_errors_path}")
        
        # Print statistics
        print(f"\n📊 THỐNG KÊ ERRORS:")
        print(f"   • Tổng số errors: {len(errors_df)}")
        print(f"   • Số câu unique có errors: {errors_df['data'].nunique()}")
        print(f"   • Số aspects bị ảnh hưởng: {errors_df['aspect'].nunique()}")
        
        print(f"\n📊 TOP 5 CONFUSION TYPES:")
        confusion_stats = errors_df.groupby('confusion_type').size().reset_index(name='count')
        confusion_stats = confusion_stats.sort_values('count', ascending=False)
        for i, (_, row) in enumerate(confusion_stats.head(5).iterrows(), 1):
            pct = row['count'] / len(errors_df) * 100
            print(f"   {i}. {row['confusion_type']:<25} {row['count']:>4} errors ({pct:.1f}%)")
        
        print(f"\n📊 ERRORS BY ASPECT:")
        aspect_error_counts = errors_df.groupby('aspect').size().reset_index(name='count')
        aspect_error_counts = aspect_error_counts.sort_values('count', ascending=False)
        for _, row in aspect_error_counts.iterrows():
            pct = row['count'] / len(errors_df) * 100
            print(f"   • {row['aspect']:<15} {row['count']:>4} errors ({pct:.1f}%)")
        
        # Save summary
        summary_df = errors_df.groupby(['aspect', 'confusion_type']).agg({'data': 'count'}).reset_index()
        summary_df.columns = ['aspect', 'confusion_type', 'error_count']
        summary_df = summary_df.sort_values(['aspect', 'error_count'], ascending=[True, False])
        summary_df.to_csv(f"{self.output_dir}/errors_summary_by_aspect.csv", index=False, encoding='utf-8-sig')
        print(f"\n✓ Saved error summary to: {self.output_dir}/errors_summary_by_aspect.csv")
        
        return errors_df_sorted
    
    def generate_improvement_suggestions(self, aspect_stats_df, sentiment_stats_df):
        """Tạo đề xuất cải thiện dựa trên phân tích"""
        print(f"\n{'='*70}")
        print("💡 ĐỀ XUẤT CÁCH CẢI THIỆN")
        print(f"{'='*70}")
        
        suggestions = []
        
        # 1. Weak aspects
        weak_aspects = aspect_stats_df[aspect_stats_df['error_rate'] > 0.15].head(5)
        if len(weak_aspects) > 0:
            suggestions.append("\n🎯 ASPECTS YẾU (Error Rate > 15%):")
            for _, row in weak_aspects.iterrows():
                suggestions.append(f"\n   📍 {row['aspect']} (Error Rate: {row['error_rate']:.2%})")
                suggestions.append(f"      • Thu thập thêm {int(row['errors'] * 2)} samples cho aspect này")
                suggestions.append(f"      • Kiểm tra quality của labels")
                suggestions.append(f"      • Cân nhắc thêm keywords/features đặc trưng")
        
        # 2. Weak sentiment classes
        weak_sentiments = sentiment_stats_df[sentiment_stats_df['error_rate'] > 0.15]
        if len(weak_sentiments) > 0:
            suggestions.append("\n\n🎯 SENTIMENT CLASSES YẾU:")
            for _, row in weak_sentiments.iterrows():
                suggestions.append(f"\n   📍 {row['sentiment'].upper()} (Error Rate: {row['error_rate']:.2%})")
                if row['sentiment'] == 'neutral':
                    suggestions.append(f"      ✓ Đã apply: Focal Loss + Oversampling")
                    suggestions.append(f"      • Có thể tăng oversampling ratio thêm (hiện tại: 40%)")
                    suggestions.append(f"      • Cân nhắc tăng Focal Loss gamma từ 2.0 → 3.0")
                    suggestions.append(f"      • Thêm data augmentation cho neutral class")
                else:
                    suggestions.append(f"      • Thu thập thêm samples chất lượng cao")
                    suggestions.append(f"      • Review lại labeling guidelines")
        
        # 3. Confusion patterns
        errors_df = self.df[~self.df['correct']]
        top_confusions = errors_df.groupby(['sentiment', 'predicted_sentiment']).size().nlargest(3)
        
        suggestions.append("\n\n🎯 CONFUSION PATTERNS PHỔ BIẾN:")
        for (true_sent, pred_sent), count in top_confusions.items():
            pct = count / len(errors_df) * 100
            suggestions.append(f"\n   📍 Nhầm {true_sent.upper()} thành {pred_sent.upper()} ({count} cases, {pct:.1f}%)")
            
            if true_sent == 'neutral' and pred_sent == 'positive':
                suggestions.append(f"      • Model có xu hướng positive bias")
                suggestions.append(f"      • Cân nhắc tăng alpha weight cho neutral trong Focal Loss")
            elif true_sent == 'neutral' and pred_sent == 'negative':
                suggestions.append(f"      • Model có xu hướng negative bias")
            elif true_sent in ['positive', 'negative'] and pred_sent in ['negative', 'positive']:
                suggestions.append(f"      • Confusion nghiêm trọng (ngược hoàn toàn)")
                suggestions.append(f"      • Kiểm tra sarcasm, irony, context")
        
        # 4. General improvements
        suggestions.append("\n\n🎯 GENERAL IMPROVEMENTS:")
        suggestions.append("\n   📍 Data Quality:")
        suggestions.append(f"      • Review lại labeling consistency")
        suggestions.append(f"      • Thêm inter-annotator agreement check")
        suggestions.append(f"\n   📍 Model Improvements:")
        suggestions.append(f"      • Fine-tune learning rate")
        suggestions.append(f"      • Thử different warmup ratios")
        suggestions.append(f"\n   📍 Training Strategy:")
        suggestions.append(f"      • Train thêm epochs nếu chưa converge")
        suggestions.append(f"      • Sử dụng early stopping với patience")
        
        suggestions_text = '\n'.join(suggestions)
        with open(f"{self.output_dir}/improvement_suggestions.txt", 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ĐỀ XUẤT CẢI THIỆN MODEL\n")
            f.write("="*70 + "\n")
            f.write(suggestions_text)
        
        print(suggestions_text)
        print(f"\n✓ Saved: {self.output_dir}/improvement_suggestions.txt")
        
        return suggestions_text
    
    def create_visualizations(self, aspect_stats_df, sentiment_stats_df, cm):
        """Tạo các visualizations"""
        print(f"\n{'='*70}")
        print("📊 Tạo visualizations...")
        print(f"{'='*70}")
        
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Aspect error rates
        fig, ax = plt.subplots(figsize=(12, 6))
        aspect_stats_sorted = aspect_stats_df.sort_values('error_rate')
        colors = ['green' if x < 0.1 else 'orange' if x < 0.2 else 'red' 
                 for x in aspect_stats_sorted['error_rate']]
        
        ax.barh(aspect_stats_sorted['aspect'], aspect_stats_sorted['error_rate'] * 100, color=colors)
        ax.set_xlabel('Error Rate (%)', fontsize=12)
        ax.set_ylabel('Aspect', fontsize=12)
        ax.set_title('Error Rate by Aspect', fontsize=14, fontweight='bold')
        ax.axvline(x=10, color='green', linestyle='--', alpha=0.5, label='Good (< 10%)')
        ax.axvline(x=20, color='orange', linestyle='--', alpha=0.5, label='Fair (< 20%)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/aspect_error_rates.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.output_dir}/aspect_error_rates.png")
        
        # 2. Confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Verify cm shape is 3x3
        if cm.shape != (3, 3):
            print(f"⚠️  Warning: Confusion matrix shape is {cm.shape}, expected (3, 3)")
            cm_padded = np.zeros((3, 3), dtype=int)
            min_rows, min_cols = min(3, cm.shape[0]), min(3, cm.shape[1])
            cm_padded[:min_rows, :min_cols] = cm[:min_rows, :min_cols]
            cm = cm_padded
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                   xticklabels=['Positive', 'Negative', 'Neutral'],
                   yticklabels=['Positive', 'Negative', 'Neutral'],
                   ax=ax)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix\n(Only Positive/Negative/Neutral)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.output_dir}/confusion_matrix.png")
        
        # 3. Sentiment error rates
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        ax.bar(sentiment_stats_df['sentiment'], sentiment_stats_df['error_rate'] * 100, color=colors)
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Error Rate (%)', fontsize=12)
        ax.set_title('Error Rate by Sentiment Class', fontsize=14, fontweight='bold')
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Target (< 10%)')
        ax.legend()
        
        for i, (_, row) in enumerate(sentiment_stats_df.iterrows()):
            ax.text(i, row['error_rate'] * 100 + 1, f"{row['error_rate']*100:.1f}%", 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/sentiment_error_rates.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.output_dir}/sentiment_error_rates.png")
    
    def generate_report(self):
        """Tạo comprehensive report"""
        print(f"\n{'='*70}")
        print("📝 Tạo Error Analysis Report...")
        print(f"{'='*70}")
        
        report_lines = [
            "="*70,
            "ERROR ANALYSIS REPORT",
            "ALL LABELED ASPECTS (Positive/Negative/Neutral)",
            "(Excludes only unlabeled aspects)",
            "="*70,
            ""
        ]
        
        total = len(self.df)
        correct = self.df['correct'].sum()
        accuracy = correct / total if total > 0 else 0
        sentiment_counts = self.df['sentiment'].value_counts().to_dict()
        
        report_lines.extend([
            "📊 OVERALL STATISTICS",
            "-"*70,
            f"Total labeled aspects:     {total:,}",
            f"  - Positive: {sentiment_counts.get('positive', 0):,}",
            f"  - Negative: {sentiment_counts.get('negative', 0):,}",
            f"  - Neutral: {sentiment_counts.get('neutral', 0):,}",
            f"Correct:           {correct:,} ({accuracy:.2%})",
            f"Errors:            {total - correct:,} ({1-accuracy:.2%})",
            "",
            "⚠️  NOTE: Includes all labeled aspects (positive/negative/neutral)",
            "   (Only unlabeled aspects are excluded)",
            ""
        ])
        
        # By aspect
        report_lines.append("📊 ERRORS BY ASPECT")
        report_lines.append("-"*70)
        aspect_stats = self.df.groupby('aspect').agg({
            'correct': ['sum', 'count', 'mean']
        }).round(4)
        
        for aspect in aspect_stats.index:
            correct_count = int(aspect_stats.loc[aspect, ('correct', 'sum')])
            total_count = int(aspect_stats.loc[aspect, ('correct', 'count')])
            acc = aspect_stats.loc[aspect, ('correct', 'mean')]
            report_lines.append(f"{aspect:<15} Accuracy: {acc:.2%}  Errors: {total_count - correct_count}/{total_count}")
        
        report_lines.append("")
        
        # By sentiment
        report_lines.append("📊 ERRORS BY SENTIMENT")
        report_lines.append("-"*70)
        for sentiment in self.VALID_LABELS:
            sent_df = self.df[self.df['sentiment'] == sentiment]
            if len(sent_df) > 0:
                acc = sent_df['correct'].mean()
                errors = (~sent_df['correct']).sum()
                report_lines.append(f"{sentiment:<12} Accuracy: {acc:.2%}  Errors: {errors}/{len(sent_df)}")
        
        report_text = '\n'.join(report_lines)
        with open(f"{self.output_dir}/error_analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ Saved: {self.output_dir}/error_analysis_report.txt")
        return report_text
    
    def run_full_analysis(self):
        """Chạy toàn bộ error analysis"""
        print(f"\n{'='*70}")
        print("🔍 BẮT ĐẦU ERROR ANALYSIS")
        print(f"{'='*70}")
        
        self.load_data()
        aspect_stats_df = self.analyze_by_aspect()
        sentiment_stats_df, cm = self.analyze_by_sentiment()
        confusion_counts = self.analyze_confusion_patterns()
        hard_cases_df = self.find_hard_cases(top_n=50)
        all_errors_df = self.save_all_errors()
        suggestions = self.generate_improvement_suggestions(aspect_stats_df, sentiment_stats_df)
        self.create_visualizations(aspect_stats_df, sentiment_stats_df, cm)
        report = self.generate_report()
        
        print(f"\n{'='*70}")
        print("✅ ERROR ANALYSIS HOÀN TẤT!")
        print(f"{'='*70}")
        print(f"\n📁 Kết quả được lưu tại: {self.output_dir}/")
        print(f"\n⚠️  LƯU Ý: Chỉ tính trên LABELED ASPECTS (positive/negative/neutral)")
        print(f"   • Bao gồm: positive, negative, và neutral labels")
        print(f"   • Bỏ qua: unlabeled aspects (NaN/empty)")
        print(f"\nFiles:")
        print(f"   • aspect_error_analysis.csv")
        print(f"   • sentiment_error_analysis.csv")
        print(f"   • confusion_patterns.csv")
        print(f"   • hard_cases.csv")
        print(f"   • all_errors_detailed.csv (TẤT CẢ {len(all_errors_df)} errors)")
        print(f"   • errors_summary_by_aspect.csv")
        print(f"   • improvement_suggestions.txt")
        print(f"   • error_analysis_report.txt")
        print(f"   • aspect_error_rates.png")
        print(f"   • confusion_matrix.png")
        print(f"   • sentiment_error_rates.png")
        
        return self


def main():
    """Main function"""
    analyzer = ErrorAnalyzer(
        test_file='multi_label/data/test_multilabel.csv',
        predictions_file='multi_label/models/multilabel_focal_contrastive/test_predictions_detailed.csv'
    )
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
