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
from collections import Counter, defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class ErrorAnalyzer:
    """Class để phân tích lỗi của model"""
    
    def __init__(self, test_file='multi_label/data/test_multilabel.csv', predictions_file='multi_label/results/test_predictions_multi.csv'):
        """
        Khởi tạo ErrorAnalyzer
        
        Args:
            test_file: File CSV chứa ground truth
            predictions_file: File CSV chứa predictions của model
        """
        self.test_file = test_file
        self.predictions_file = predictions_file
        
        # Label mappings
        self.id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
        self.label2id = {'positive': 0, 'negative': 1, 'neutral': 2}
        
        # Output directory
        self.output_dir = 'multi_label/error_analysis_results'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        """Load test data và predictions (multi-label wide format) và convert to long format"""
        print(f"\n{'='*70}")
        print("📁 Đang tải dữ liệu...")
        print(f"{'='*70}")
        
        # Load ground truth (wide format)
        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Không tìm thấy file: {self.test_file}")
        
        test_wide = pd.read_csv(self.test_file, encoding='utf-8-sig')
        print(f"✓ Loaded test data: {len(test_wide)} sentences (wide format)")
        
        # Load predictions (wide format)
        if not os.path.exists(self.predictions_file):
            raise FileNotFoundError(f"Không tìm thấy file: {self.predictions_file}")
        
        pred_wide = pd.read_csv(self.predictions_file, encoding='utf-8-sig')
        print(f"✓ Loaded predictions: {len(pred_wide)} sentences (wide format)")
        
        # Load true labels (wide format)
        true_file = self.predictions_file.replace('.csv', '_true.csv')
        if os.path.exists(true_file):
            true_wide = pd.read_csv(true_file, encoding='utf-8-sig')
            print(f"✓ Loaded true labels: {len(true_wide)} sentences (wide format)")
        else:
            # If true labels file doesn't exist, use test data
            true_wide = test_wide.copy()
            print(f"⚠️  True labels file not found, using test data")
        
        # Get aspect columns (all columns except 'data')
        aspect_cols = [col for col in pred_wide.columns if col != 'data']
        print(f"✓ Found {len(aspect_cols)} aspects: {', '.join(aspect_cols)}")
        
        # Convert wide format to long format
        # Wide: data, Battery, Camera, ... (one row per sentence)
        # Long: data, aspect, sentiment, predicted_sentiment (one row per aspect)
        long_data = []
        
        for idx in range(len(pred_wide)):
            text = pred_wide.iloc[idx]['data']

            for aspect in aspect_cols:
                pred_raw = pred_wide.iloc[idx][aspect]
                true_raw = true_wide.iloc[idx][aspect]

                def normalize(value):
                    if isinstance(value, str):
                        value = value.strip().lower()
                        return value if value else None
                    if pd.isna(value):
                        return None
                    value = str(value).strip().lower()
                    return value if value else None

                true_sentiment = normalize(true_raw)
                pred_sentiment = normalize(pred_raw)

                # Skip aspects without ground-truth label (missing in dataset)
                if true_sentiment is None:
                    continue

                if pred_sentiment is None:
                    pred_sentiment = 'neutral'

                long_data.append({
                    'data': text,
                    'aspect': aspect,
                    'sentiment': true_sentiment,
                    'predicted_sentiment': pred_sentiment,
                    'predicted_label_id': self.label2id.get(pred_sentiment, 2),
                    'correct': true_sentiment == pred_sentiment
                })
        
        self.df = pd.DataFrame(long_data)
        
        # Store for reference
        self.test_df = test_wide
        self.pred_df = pred_wide
        
        # Calculate accuracy
        accuracy = self.df['correct'].mean()
        print(f"✓ Converted to long format: {len(self.df)} predictions (sentences × aspects)")
        print(f"✓ Overall accuracy: {accuracy:.2%}")
        print(f"✓ Total errors: {(~self.df['correct']).sum()} / {len(self.df)}")
        
        return self
    
    def analyze_by_aspect(self):
        """Phân tích lỗi theo từng aspect"""
        print(f"\n{'='*70}")
        print("📊 PHÂN TÍCH THEO ASPECT")
        print(f"{'='*70}")
        
        # Calculate accuracy per aspect
        aspect_stats = []
        
        for aspect in sorted(self.df['aspect'].unique()):
            aspect_df = self.df[self.df['aspect'] == aspect]
            
            total = len(aspect_df)
            correct = aspect_df['correct'].sum()
            errors = total - correct
            accuracy = correct / total
            
            aspect_stats.append({
                'aspect': aspect,
                'total': total,
                'correct': correct,
                'errors': errors,
                'accuracy': accuracy,
                'error_rate': 1 - accuracy
            })
        
        aspect_stats_df = pd.DataFrame(aspect_stats)
        aspect_stats_df = aspect_stats_df.sort_values('error_rate', ascending=False)
        
        # Print table
        print(f"\n{'Aspect':<15} {'Total':<8} {'Correct':<8} {'Errors':<8} {'Accuracy':<10} {'Error Rate'}")
        print(f"{'-'*70}")
        
        for _, row in aspect_stats_df.iterrows():
            print(f"{row['aspect']:<15} {row['total']:<8} {row['correct']:<8} "
                  f"{row['errors']:<8} {row['accuracy']:<10.2%} {row['error_rate']:.2%}")
        
        # Save to file
        aspect_stats_df.to_csv(f"{self.output_dir}/aspect_error_analysis.csv", index=False, encoding='utf-8-sig')
        print(f"\n✓ Saved: {self.output_dir}/aspect_error_analysis.csv")
        
        # Find weakest aspects
        print(f"\n🔴 TOP 5 ASPECTS YẾU NHẤT (Error Rate cao nhất):")
        for i, row in aspect_stats_df.head(5).iterrows():
            print(f"   {i+1}. {row['aspect']:<15} Error Rate: {row['error_rate']:.2%} ({row['errors']}/{row['total']} errors)")
        
        return aspect_stats_df
    
    def analyze_by_sentiment(self):
        """Phân tích lỗi theo sentiment class"""
        print(f"\n{'='*70}")
        print("📊 PHÂN TÍCH THEO SENTIMENT")
        print(f"{'='*70}")
        
        sentiment_stats = []
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sent_df = self.df[self.df['sentiment'] == sentiment]
            
            total = len(sent_df)
            correct = sent_df['correct'].sum()
            errors = total - correct
            accuracy = correct / total if total > 0 else 0
            
            sentiment_stats.append({
                'sentiment': sentiment,
                'total': total,
                'correct': correct,
                'errors': errors,
                'accuracy': accuracy,
                'error_rate': 1 - accuracy
            })
        
        sentiment_stats_df = pd.DataFrame(sentiment_stats)
        sentiment_stats_df = sentiment_stats_df.sort_values('error_rate', ascending=False)
        
        # Print table
        print(f"\n{'Sentiment':<12} {'Total':<8} {'Correct':<8} {'Errors':<8} {'Accuracy':<10} {'Error Rate'}")
        print(f"{'-'*70}")
        
        for _, row in sentiment_stats_df.iterrows():
            print(f"{row['sentiment']:<12} {row['total']:<8} {row['correct']:<8} "
                  f"{row['errors']:<8} {row['accuracy']:<10.2%} {row['error_rate']:.2%}")
        
        # Confusion matrix
        print(f"\n📊 CONFUSION MATRIX:")
        
        cm = confusion_matrix(self.df['sentiment'], self.df['predicted_sentiment'], 
                             labels=['positive', 'negative', 'neutral'])
        
        print(f"\n{'':>12} {'Predicted →':>12}")
        print(f"{'True ↓':<12} {'Positive':<12} {'Negative':<12} {'Neutral':<12}")
        print(f"{'-'*60}")
        
        for i, true_label in enumerate(['positive', 'negative', 'neutral']):
            row_str = f"{true_label:<12}"
            for j in range(3):
                row_str += f" {cm[i][j]:<12}"
            print(row_str)
        
        # Save
        sentiment_stats_df.to_csv(f"{self.output_dir}/sentiment_error_analysis.csv", index=False, encoding='utf-8-sig')
        print(f"\n✓ Saved: {self.output_dir}/sentiment_error_analysis.csv")
        
        return sentiment_stats_df, cm
    
    def analyze_confusion_patterns(self):
        """Phân tích các confusion patterns chi tiết"""
        print(f"\n{'='*70}")
        print("🔍 PHÂN TÍCH CONFUSION PATTERNS")
        print(f"{'='*70}")
        
        # Get errors only
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
        
        # Analyze by aspect + sentiment
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
        
        # Save
        confusion_counts.to_csv(f"{self.output_dir}/confusion_patterns.csv", index=False, encoding='utf-8-sig')
        print(f"\n✓ Saved: {self.output_dir}/confusion_patterns.csv")
        
        return confusion_counts
    
    def find_hard_cases(self, top_n=50):
        """Tìm các cases khó nhất (sai nhiều lần hoặc có patterns đặc biệt)"""
        print(f"\n{'='*70}")
        print(f"🔴 TÌM TOP {top_n} HARD CASES")
        print(f"{'='*70}")
        
        errors_df = self.df[~self.df['correct']].copy()
        
        # Find most confused aspects
        aspect_errors = errors_df.groupby('aspect').size().reset_index(name='error_count')
        aspect_errors = aspect_errors.sort_values('error_count', ascending=False)
        
        print(f"\n📊 Aspects có NHIỀU LỖI NHẤT:")
        for i, row in aspect_errors.head(5).iterrows():
            print(f"   {i+1}. {row['aspect']:<15} {row['error_count']} errors")
        
        # Find sentences with multiple errors (appeared in errors with different aspects)
        sentence_error_counts = errors_df.groupby('data').size().reset_index(name='error_count')
        sentence_error_counts = sentence_error_counts.sort_values('error_count', ascending=False)
        
        print(f"\n📝 Câu có NHIỀU LỖI NHẤT (confused across multiple aspects):")
        for i, row in sentence_error_counts.head(10).iterrows():
            sentence = row['data']
            count = row['error_count']
            
            # Get details
            sent_errors = errors_df[errors_df['data'] == sentence]
            aspects = ', '.join(sent_errors['aspect'].unique())
            
            print(f"\n   {i+1}. [{count} errors] {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
            print(f"      Aspects: {aspects}")
            
            for _, err in sent_errors.iterrows():
                print(f"      • {err['aspect']}: {err['sentiment']} → {err['predicted_sentiment']}")
        
        # Save detailed error examples
        print(f"\n💾 Saving detailed error examples...")
        
        # Top N errors per aspect
        hard_cases = []
        
        for aspect in errors_df['aspect'].unique():
            aspect_errors_df = errors_df[errors_df['aspect'] == aspect]
            
            for _, row in aspect_errors_df.head(5).iterrows():  # Top 5 per aspect
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
        """Lưu TẤT CẢ các errors (không chỉ hard cases) để phân tích chi tiết"""
        print(f"\n{'='*70}")
        print("💾 LƯU TẤT CẢ ERRORS CHO PHÂN TÍCH CHI TIẾT")
        print(f"{'='*70}")
        
        # Get ALL errors
        errors_df = self.df[~self.df['correct']].copy()
        
        print(f"\n📊 Tổng số errors: {len(errors_df)}")
        
        # Add additional columns for analysis
        errors_df['confusion_type'] = errors_df.apply(
            lambda row: f"{row['sentiment']} → {row['predicted_sentiment']}",
            axis=1
        )
        
        # Add confidence if available
        if 'confidence' in self.pred_df.columns:
            errors_df['confidence'] = self.pred_df.loc[errors_df.index, 'confidence']
        
        # Count errors per sentence
        sentence_error_counts = errors_df.groupby('data').size().reset_index(name='error_count_for_sentence')
        errors_df = errors_df.merge(sentence_error_counts, on='data', how='left')
        
        # Sort by aspect and confusion type for better analysis
        errors_df_sorted = errors_df.sort_values(['aspect', 'confusion_type', 'data'])
        
        # Select columns to save
        columns_to_save = ['data', 'aspect', 'sentiment', 'predicted_sentiment', 'confusion_type']
        if 'confidence' in errors_df.columns:
            columns_to_save.append('confidence')
        columns_to_save.append('error_count_for_sentence')
        
        # Save to CSV
        all_errors_path = f"{self.output_dir}/all_errors_detailed.csv"
        errors_df_sorted[columns_to_save].to_csv(all_errors_path, index=False, encoding='utf-8-sig')
        
        print(f"✓ Saved ALL {len(errors_df)} errors to: {all_errors_path}")
        
        # Print summary statistics
        print(f"\n📊 THỐNG KÊ ERRORS:")
        print(f"   • Tổng số errors: {len(errors_df)}")
        print(f"   • Số câu unique có errors: {errors_df['data'].nunique()}")
        print(f"   • Số aspects bị ảnh hưởng: {errors_df['aspect'].nunique()}")
        
        print(f"\n📊 TOP 5 CONFUSION TYPES:")
        confusion_stats = errors_df.groupby('confusion_type').size().reset_index(name='count')
        confusion_stats = confusion_stats.sort_values('count', ascending=False)
        for i, row in confusion_stats.head(5).iterrows():
            pct = row['count'] / len(errors_df) * 100
            print(f"   {i+1}. {row['confusion_type']:<25} {row['count']:>4} errors ({pct:.1f}%)")
        
        print(f"\n📊 ERRORS BY ASPECT:")
        aspect_error_counts = errors_df.groupby('aspect').size().reset_index(name='count')
        aspect_error_counts = aspect_error_counts.sort_values('count', ascending=False)
        for _, row in aspect_error_counts.iterrows():
            pct = row['count'] / len(errors_df) * 100
            print(f"   • {row['aspect']:<15} {row['count']:>4} errors ({pct:.1f}%)")
        
        # Also save a summary version grouped by aspect and confusion type
        summary_path = f"{self.output_dir}/errors_summary_by_aspect.csv"
        
        # Build aggregation dict dynamically
        agg_dict = {'data': 'count'}
        if 'confidence' in errors_df.columns:
            agg_dict['confidence'] = 'mean'
        
        summary_df = errors_df.groupby(['aspect', 'confusion_type']).agg(agg_dict).reset_index()
        
        # Rename columns based on what was aggregated
        if 'confidence' in errors_df.columns:
            summary_df.columns = ['aspect', 'confusion_type', 'error_count', 'avg_confidence']
        else:
            summary_df.columns = ['aspect', 'confusion_type', 'error_count']
        summary_df = summary_df.sort_values(['aspect', 'error_count'], ascending=[True, False])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        print(f"\n✓ Saved error summary to: {summary_path}")
        
        return errors_df_sorted
    
    def generate_improvement_suggestions(self, aspect_stats_df, sentiment_stats_df):
        """Tạo đề xuất cải thiện dựa trên phân tích"""
        print(f"\n{'='*70}")
        print("💡 ĐỀ XUẤT CÁCH CẢI THIỆN")
        print(f"{'='*70}")
        
        suggestions = []
        
        # 1. Aspects yếu
        weak_aspects = aspect_stats_df[aspect_stats_df['error_rate'] > 0.15].head(5)
        
        if len(weak_aspects) > 0:
            suggestions.append("\n🎯 ASPECTS YẾU (Error Rate > 15%):")
            for _, row in weak_aspects.iterrows():
                suggestions.append(f"\n   📍 {row['aspect']} (Error Rate: {row['error_rate']:.2%})")
                suggestions.append(f"      • Thu thập thêm {int(row['errors'] * 2)} samples cho aspect này")
                suggestions.append(f"      • Kiểm tra quality của labels")
                suggestions.append(f"      • Cân nhắc thêm keywords/features đặc trưng")
        
        # 2. Sentiment classes yếu
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
                suggestions.append(f"      • Review các neutral samples có từ tích cực")
            
            elif true_sent == 'neutral' and pred_sent == 'negative':
                suggestions.append(f"      • Model có xu hướng negative bias")
                suggestions.append(f"      • Review các neutral samples có từ tiêu cực")
            
            elif true_sent == 'positive' and pred_sent == 'negative':
                suggestions.append(f"      • Confusion nghiêm trọng (ngược hoàn toàn)")
                suggestions.append(f"      • Kiểm tra data quality và labeling")
                suggestions.append(f"      • Có thể có sarcasm hoặc context phức tạp")
            
            elif true_sent == 'negative' and pred_sent == 'positive':
                suggestions.append(f"      • Confusion nghiêm trọng (ngược hoàn toàn)")
                suggestions.append(f"      • Kiểm tra sarcasm, irony, context")
                suggestions.append(f"      • Cân nhắc thêm features hoặc context window")
        
        # 4. General improvements
        suggestions.append("\n\n🎯 GENERAL IMPROVEMENTS:")
        suggestions.append("\n   📍 Data Quality:")
        suggestions.append(f"      • Review lại labeling consistency")
        suggestions.append(f"      • Thêm inter-annotator agreement check")
        suggestions.append(f"      • Xem xét data augmentation")
        
        suggestions.append("\n   📍 Model Improvements:")
        suggestions.append(f"      • Fine-tune learning rate (hiện tại: 2e-5)")
        suggestions.append(f"      • Thử different warmup ratios")
        suggestions.append(f"      • Cân nhắc ensemble multiple models")
        
        suggestions.append("\n   📍 Training Strategy:")
        suggestions.append(f"      • Train thêm epochs nếu chưa converge")
        suggestions.append(f"      • Sử dụng early stopping với patience")
        suggestions.append(f"      • Thử different batch sizes")
        
        suggestions.append("\n   📍 Advanced Techniques:")
        suggestions.append(f"      • SMOTE thay vì random oversampling")
        suggestions.append(f"      • Mixup / Cutmix augmentation")
        suggestions.append(f"      • Multi-task learning (nếu có thêm tasks)")
        suggestions.append(f"      • Adversarial training")
        
        # Save suggestions
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
        
        # 2. Confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                   xticklabels=['Positive', 'Negative', 'Neutral'],
                   yticklabels=['Positive', 'Negative', 'Neutral'],
                   ax=ax)
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.output_dir}/confusion_matrix.png")
        
        # 3. Sentiment error rates
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = ['#2ecc71', '#e74c3c', '#f39c12']  # green, red, orange
        
        ax.bar(sentiment_stats_df['sentiment'], sentiment_stats_df['error_rate'] * 100, 
              color=colors)
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Error Rate (%)', fontsize=12)
        ax.set_title('Error Rate by Sentiment Class', fontsize=14, fontweight='bold')
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Target (< 10%)')
        ax.legend()
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(sentiment_stats_df.iterrows()):
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
        
        report_lines = []
        
        report_lines.append("="*70)
        report_lines.append("ERROR ANALYSIS REPORT")
        report_lines.append("="*70)
        report_lines.append("")
        
        # Overall stats
        total = len(self.df)
        correct = self.df['correct'].sum()
        errors = total - correct
        accuracy = correct / total
        
        report_lines.append("📊 OVERALL STATISTICS")
        report_lines.append("-"*70)
        report_lines.append(f"Total samples:     {total:,}")
        report_lines.append(f"Correct:           {correct:,} ({accuracy:.2%})")
        report_lines.append(f"Errors:            {errors:,} ({1-accuracy:.2%})")
        report_lines.append("")
        
        # By aspect
        report_lines.append("📊 ERRORS BY ASPECT")
        report_lines.append("-"*70)
        
        aspect_stats = self.df.groupby('aspect').agg({
            'correct': ['sum', 'count', 'mean']
        }).round(4)
        
        for aspect in aspect_stats.index:
            correct_count = int(aspect_stats.loc[aspect, ('correct', 'sum')])
            total_count = int(aspect_stats.loc[aspect, ('correct', 'count')])
            accuracy = aspect_stats.loc[aspect, ('correct', 'mean')]
            error_count = total_count - correct_count
            
            report_lines.append(f"{aspect:<15} Accuracy: {accuracy:.2%}  Errors: {error_count}/{total_count}")
        
        report_lines.append("")
        
        # By sentiment
        report_lines.append("📊 ERRORS BY SENTIMENT")
        report_lines.append("-"*70)
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sent_df = self.df[self.df['sentiment'] == sentiment]
            if len(sent_df) > 0:
                acc = sent_df['correct'].mean()
                errors = (~sent_df['correct']).sum()
                total = len(sent_df)
                report_lines.append(f"{sentiment:<12} Accuracy: {acc:.2%}  Errors: {errors}/{total}")
        
        report_lines.append("")
        
        # Save report
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
        
        # Load data
        self.load_data()
        
        # Run analyses
        aspect_stats_df = self.analyze_by_aspect()
        sentiment_stats_df, cm = self.analyze_by_sentiment()
        confusion_counts = self.analyze_confusion_patterns()
        hard_cases_df = self.find_hard_cases(top_n=50)
        
        # Save ALL errors for detailed analysis
        all_errors_df = self.save_all_errors()
        
        # Generate suggestions
        suggestions = self.generate_improvement_suggestions(aspect_stats_df, sentiment_stats_df)
        
        # Create visualizations
        self.create_visualizations(aspect_stats_df, sentiment_stats_df, cm)
        
        # Generate report
        report = self.generate_report()
        
        print(f"\n{'='*70}")
        print("✅ ERROR ANALYSIS HOÀN TẤT!")
        print(f"{'='*70}")
        print(f"\n📁 Kết quả được lưu tại: {self.output_dir}/")
        print(f"\nFiles:")
        print(f"   • aspect_error_analysis.csv")
        print(f"   • sentiment_error_analysis.csv")
        print(f"   • confusion_patterns.csv")
        print(f"   • hard_cases.csv (top 5 errors per aspect)")
        print(f"   • all_errors_detailed.csv (TẤT CẢ {len(all_errors_df) if 'all_errors_df' in locals() else ''} errors)")
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
        predictions_file='multi_label/results/test_predictions_multi.csv'
    )
    
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
