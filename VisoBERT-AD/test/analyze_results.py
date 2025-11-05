"""
Script Phân Tích Kết Quả Chi Tiết Theo Từng Aspect
==================================================
Tạo các biểu đồ và báo cáo chi tiết cho từng khía cạnh

Output:
    - Confusion matrices cho từng aspect
    - Bar charts về accuracy, precision, recall, F1
    - Heatmap tổng hợp
    - Báo cáo chi tiết dạng text và PNG
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import os
from pathlib import Path

# Set style cho matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Tạo thư mục cho results
RESULTS_DIR = "multi_label/analysis_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def normalize_value(value):
    """Normalize a value to lowercase string sentiment label"""
    if isinstance(value, str):
        value = value.strip().lower()
        return value if value else None
    if pd.isna(value):
        return None
    # If it's a numeric value, convert using id2label
    if isinstance(value, (int, float)) and not pd.isna(value):
        id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
        return id2label.get(int(value), None)
    value = str(value).strip().lower()
    return value if value else None


def load_predictions(predictions_file='multi_label/models/multilabel_focal_contrastive/test_predictions_detailed.csv',
                     test_file='multi_label/data/test_multilabel.csv'):
    """
    Load predictions từ CSV và convert sang long format for analysis
    IMPORTANT: Sử dụng test data làm ground truth để phân biệt neutral thật vs unlabeled placeholder
    """
    print(f"\n{'='*70}")
    print(f"Đang tải predictions và test data...")
    print(f"{'='*70}")
    
    # Load test data (ground truth source)
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Không tìm thấy file: {test_file}")
    test_wide = pd.read_csv(test_file, encoding='utf-8-sig')
    print(f"Loaded test data: {len(test_wide)} sentences")
    
    # Load predictions file
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Không tìm thấy file: {predictions_file}")
    pred_raw = pd.read_csv(predictions_file, encoding='utf-8-sig')
    print(f"Loaded predictions file: {len(pred_raw)} rows")
    
    # Check format: numeric (test_predictions_detailed.csv) or string
    if 'sample_id' in pred_raw.columns and '_pred' in str(pred_raw.columns):
        print("Detected numeric format predictions file")
        
        # Ensure alignment
        min_len = min(len(pred_raw), len(test_wide))
        pred_raw = pred_raw.iloc[:min_len]
        test_wide = test_wide.iloc[:min_len]
        
        # Extract aspects
        aspects = sorted(set(
            col.replace('_pred', '') 
            for col in pred_raw.columns 
            if col.endswith('_pred')
        ))
        print(f"Found {len(aspects)} aspects: {', '.join(aspects)}")
        
        # Convert to wide format
        pred_wide = pd.DataFrame()
        pred_wide['data'] = test_wide['data'].values
        
        true_wide = pd.DataFrame()
        true_wide['data'] = test_wide['data'].values
        
        for aspect in aspects:
            pred_col = f"{aspect}_pred"
            
            # Predictions: convert numeric to string
            if pred_col in pred_raw.columns:
                id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
                pred_wide[aspect] = pred_raw[pred_col].map(id2label).fillna('neutral')
            else:
                pred_wide[aspect] = 'neutral'
            
            # IMPORTANT: Always use test data as ground truth
            # This distinguishes real Neutral from unlabeled placeholder (NaN)
            if aspect in test_wide.columns:
                true_wide[aspect] = test_wide[aspect].apply(
                    lambda x: normalize_value(x) if pd.notna(x) and str(x).strip() != '' else None
                )
            else:
                true_wide[aspect] = None
    else:
        # String format
        print("Detected string format predictions file")
        pred_wide = pred_raw
        true_wide = test_wide.copy()
        aspects = [col for col in pred_wide.columns if col != 'data']
        print(f"Found {len(aspects)} aspects: {', '.join(aspects)}")
    
    # Convert wide format to long format
    # Only include LABELED aspects (positive/negative/neutral)
    # Skip unlabeled aspects (NaN)
    long_data = []
    skipped_unlabeled = 0
    
    for idx in range(len(pred_wide)):
        text = pred_wide.iloc[idx]['data']
        
        for aspect in aspects:
            pred_val = pred_wide.iloc[idx][aspect] if aspect in pred_wide.columns else None
            true_val = true_wide.iloc[idx][aspect] if aspect in true_wide.columns else None
            
            true_sentiment = normalize_value(true_val)
            pred_sentiment = normalize_value(pred_val)
            
            # SKIP unlabeled aspects (NaN in test data = not mentioned)
            if true_sentiment is None:
                skipped_unlabeled += 1
                continue
            
            # Default prediction if None (shouldn't happen, but handle it)
            if pred_sentiment is None:
                pred_sentiment = 'neutral'
            
            # Only include labeled aspects
            long_data.append({
                'text': text,
                'aspect': aspect,
                'true_sentiment': true_sentiment,
                'predicted_sentiment': pred_sentiment
            })
    
    df = pd.DataFrame(long_data)
    
    total_aspects = len(test_wide) * len(aspects)
    sentiment_counts = df['true_sentiment'].value_counts().to_dict()
    
    print(f"Converted to long format: {len(df)} predictions (labeled aspects)")
    print(f"   • Total aspects in dataset: {total_aspects:,}")
    print(f"   • Labeled aspects: {len(df):,} ({len(df)/total_aspects*100:.1f}%)")
    print(f"     - Positive: {sentiment_counts.get('positive', 0):,}")
    print(f"     - Negative: {sentiment_counts.get('negative', 0):,}")
    print(f"     - Neutral: {sentiment_counts.get('neutral', 0):,}")
    print(f"   • Skipped unlabeled: {skipped_unlabeled:,} ({skipped_unlabeled/total_aspects*100:.1f}%)")
    print(f"\nNOTE: Metrics calculated ONLY on labeled aspects (positive/negative/neutral)")
    print(f"   • Unlabeled aspects (NaN) are excluded")
    print(f"   • Neutral ≠ Unlabeled (Neutral is a real label, Unlabeled means not mentioned)")
    
    return df


def create_all_confusion_matrices_grid(confusion_matrices):
    """Tạo grid chứa tất cả confusion matrices"""
    print(f"\n{'='*70}")
    print("Tạo grid confusion matrices cho tất cả aspects...")
    print(f"{'='*70}")
    
    n_aspects = len(confusion_matrices)
    
    # Tính số hàng và cột cho grid
    n_cols = 4  # 4 cột
    n_rows = (n_aspects + n_cols - 1) // n_cols  # Làm tròn lên
    
    # Tạo figure lớn
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    fig.suptitle('Confusion Matrices by Aspect', fontsize=20, fontweight='bold', y=0.995)
    
    # Flatten axes nếu có nhiều hơn 1 hàng
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    # Vẽ từng confusion matrix
    for idx, cm_data in enumerate(confusion_matrices):
        ax = axes_flat[idx]
        
        aspect = cm_data['aspect']
        cm = cm_data['cm']
        labels = cm_data['labels']
        samples = cm_data['samples']
        
        # Vẽ heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax,
            square=True
        )
        
        # Tính accuracy cho aspect này
        accuracy = np.trace(cm) / np.sum(cm) * 100
        
        ax.set_title(f'{aspect}\n(n={samples}, acc={accuracy:.1f}%)', 
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('True', fontsize=10)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.tick_params(labelsize=9)
    
    # Ẩn các subplot thừa
    for idx in range(n_aspects, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, 'confusion_matrices_all_aspects.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Đã lưu: {save_path}")


def analyze_by_aspect(df):
    """Phân tích chi tiết theo từng aspect"""
    print(f"\n{'='*70}")
    print("PHÂN TÍCH THEO TỪNG ASPECT")
    print(f"{'='*70}\n")
    
    aspects = sorted(df['aspect'].unique())
    results = []
    confusion_matrices = []
    
    for aspect in aspects:
        print(f"\n{'─'*70}")
        print(f"Aspect: {aspect}")
        print(f"{'─'*70}")
        
        # Filter data cho aspect này
        aspect_df = df[df['aspect'] == aspect]
        y_true = aspect_df['true_sentiment'].values
        y_pred = aspect_df['predicted_sentiment'].values
        
        # Tính metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Tính confusion matrix cho aspect này
        labels = ['positive', 'negative', 'neutral']
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        confusion_matrices.append({
            'aspect': aspect,
            'cm': cm,
            'labels': labels,
            'samples': len(aspect_df)
        })
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            output_dict=True,
            zero_division=0
        )
        
        # Lưu results
        results.append({
            'aspect': aspect,
            'samples': len(aspect_df),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report
        })
        
        # In kết quả
        print(f"Số mẫu:   {len(aspect_df)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        # In chi tiết theo từng sentiment
        print(f"\nChi tiết theo sentiment:")
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in report and sentiment in ['positive', 'negative', 'neutral']:
                print(f"  {sentiment:>10}: "
                      f"P={report[sentiment]['precision']:.3f} "
                      f"R={report[sentiment]['recall']:.3f} "
                      f"F1={report[sentiment]['f1-score']:.3f} "
                      f"(n={int(report[sentiment]['support'])})")
    
    return results, confusion_matrices


def create_metrics_comparison_plot(results):
    """Tạo biểu đồ so sánh metrics giữa các aspects"""
    print(f"\n{'='*70}")
    print("Tạo biểu đồ so sánh metrics...")
    print(f"{'='*70}")
    
    # Chuẩn bị data
    aspects = [r['aspect'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1_scores = [r['f1'] for r in results]
    
    # 1. Grouped bar chart cho tất cả metrics
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(aspects))
    width = 0.2
    
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, f1_scores, width, label='F1 Score', alpha=0.8)
    
    ax.set_xlabel('Aspect', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Performance Metrics Comparison Across Aspects', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(aspects, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Thêm giá trị lên từng bar
    for i, (acc, prec, rec, f1) in enumerate(zip(accuracies, precisions, recalls, f1_scores)):
        ax.text(i - 1.5*width, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i - 0.5*width, prec + 0.02, f'{prec:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + 0.5*width, rec + 0.02, f'{rec:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + 1.5*width, f1 + 0.02, f'{f1:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {save_path}")
    
    # 2. Individual plots cho từng metric
    metrics_data = {
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    }
    
    for metric_name, metric_values in metrics_data.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(aspects, metric_values, alpha=0.7, edgecolor='black')
        
        # Màu sắc theo giá trị
        colors = plt.cm.RdYlGn(np.array(metric_values))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel('Aspect', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name} by Aspect', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(aspects, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        # Thêm giá trị
        for i, (aspect, value) in enumerate(zip(aspects, metric_values)):
            ax.text(i, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Thêm đường mean
        mean_val = np.mean(metric_values)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean: {mean_val:.3f}')
        ax.legend()
        
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f'{metric_name.lower().replace(" ", "_")}_by_aspect.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Đã lưu: {save_path}")


def create_sample_distribution_plot(results):
    """Tạo biểu đồ phân bố số mẫu"""
    print(f"\n{'='*70}")
    print("Tạo biểu đồ phân bố mẫu...")
    print(f"{'='*70}")
    
    aspects = [r['aspect'] for r in results]
    samples = [r['samples'] for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(aspects, samples, alpha=0.7, edgecolor='black', color='steelblue')
    
    ax.set_xlabel('Aspect', fontweight='bold')
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Sample Distribution Across Aspects', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticklabels(aspects, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Thêm giá trị
    max_sample = max(samples)
    for i, (aspect, sample) in enumerate(zip(aspects, samples)):
        ax.text(i, sample + max_sample * 0.01, str(sample), ha='center', va='bottom', fontweight='bold')
    
    # Tăng ylim để text không bị cắt
    ax.set_ylim([0, max_sample * 1.1])
    
    # Thêm tổng
    total = sum(samples)
    ax.text(0.98, 0.98, f'Total: {total} samples', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'sample_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {save_path}")


def create_heatmap_metrics(results):
    """Tạo heatmap cho metrics của tất cả aspects"""
    print(f"\n{'='*70}")
    print("Tạo heatmap metrics...")
    print(f"{'='*70}")
    
    aspects = [r['aspect'] for r in results]
    metrics_matrix = np.array([
        [r['accuracy'] for r in results],
        [r['precision'] for r in results],
        [r['recall'] for r in results],
        [r['f1'] for r in results]
    ])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(
        metrics_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlGnBu',
        xticklabels=aspects,
        yticklabels=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        cbar_kws={'label': 'Score'},
        vmin=0.0,
        vmax=1.0
    )
    
    plt.title('Performance Metrics Heatmap by Aspect', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Aspect', fontweight='bold')
    plt.ylabel('Metric', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, 'metrics_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {save_path}")


def create_overall_confusion_matrix(df):
    """Tạo confusion matrix tổng thể"""
    print(f"\n{'='*70}")
    print("Tạo confusion matrix tổng thể...")
    print(f"{'='*70}")
    
    y_true = df['true_sentiment'].values
    y_pred = df['predicted_sentiment'].values
    labels = ['positive', 'negative', 'neutral']
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    
    # Thêm percentage
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                          ha="center", va="center", color="red", fontsize=10)
    
    plt.title('Overall Confusion Matrix\n(All Aspects Combined)', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, 'confusion_matrix_overall.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {save_path}")


def save_detailed_report(results, df):
    """Lưu báo cáo chi tiết dạng text"""
    print(f"\n{'='*70}")
    print("Đang lưu báo cáo chi tiết...")
    print(f"{'='*70}")
    
    report_path = os.path.join(RESULTS_DIR, 'detailed_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("BÁO CÁO PHÂN TÍCH CHI TIẾT THEO TỪNG ASPECT\n")
        f.write("="*70 + "\n\n")
        
        # Overall stats
        f.write("TỔNG QUAN:\n")
        f.write(f"  Tổng số mẫu test: {len(df)}\n")
        f.write(f"  Số aspects: {df['aspect'].nunique()}\n")
        
        overall_acc = accuracy_score(df['true_sentiment'], df['predicted_sentiment'])
        f.write(f"  Overall Accuracy: {overall_acc:.4f}\n\n")
        
        # Per-aspect details
        f.write("="*70 + "\n")
        f.write("CHI TIẾT THEO TỪNG ASPECT:\n")
        f.write("="*70 + "\n\n")
        
        for result in results:
            f.write(f"\n{'─'*70}\n")
            f.write(f"Aspect: {result['aspect']}\n")
            f.write(f"{'─'*70}\n")
            f.write(f"Số mẫu:     {result['samples']}\n")
            f.write(f"Accuracy:   {result['accuracy']:.4f}\n")
            f.write(f"Precision:  {result['precision']:.4f}\n")
            f.write(f"Recall:     {result['recall']:.4f}\n")
            f.write(f"F1 Score:   {result['f1']:.4f}\n\n")
            
            f.write("Chi tiết theo sentiment:\n")
            report = result['report']
            for sentiment in ['positive', 'negative', 'neutral']:
                if sentiment in report:
                    f.write(f"  {sentiment:>10}:\n")
                    f.write(f"    Precision: {report[sentiment]['precision']:.4f}\n")
                    f.write(f"    Recall:    {report[sentiment]['recall']:.4f}\n")
                    f.write(f"    F1-score:  {report[sentiment]['f1-score']:.4f}\n")
                    f.write(f"    Support:   {int(report[sentiment]['support'])}\n")
        
        # Summary statistics
        f.write(f"\n{'='*70}\n")
        f.write("THỐNG KÊ TỔNG HỢP:\n")
        f.write(f"{'='*70}\n\n")
        
        accuracies = [r['accuracy'] for r in results]
        f1_scores = [r['f1'] for r in results]
        
        f.write(f"Accuracy trung bình:  {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})\n")
        f.write(f"F1 Score trung bình:  {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})\n")
        f.write(f"Accuracy cao nhất:    {max(accuracies):.4f} ({results[accuracies.index(max(accuracies))]['aspect']})\n")
        f.write(f"Accuracy thấp nhất:   {min(accuracies):.4f} ({results[accuracies.index(min(accuracies))]['aspect']})\n")
        f.write(f"F1 Score cao nhất:    {max(f1_scores):.4f} ({results[f1_scores.index(max(f1_scores))]['aspect']})\n")
        f.write(f"F1 Score thấp nhất:   {min(f1_scores):.4f} ({results[f1_scores.index(min(f1_scores))]['aspect']})\n")
    
    print(f"Đã lưu: {report_path}")


def create_summary_table_image(results):
    """Tạo bảng tổng hợp dạng ảnh"""
    print(f"\n{'='*70}")
    print("Tạo bảng tổng hợp...")
    print(f"{'='*70}")
    
    # Chuẩn bị data
    table_data = []
    for r in results:
        table_data.append([
            r['aspect'],
            r['samples'],
            f"{r['accuracy']:.3f}",
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1']:.3f}"
        ])
    
    # Thêm dòng trung bình
    accuracies = [r['accuracy'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1_scores = [r['f1'] for r in results]
    total_samples = sum([r['samples'] for r in results])
    
    table_data.append([
        'MEAN',
        total_samples,
        f"{np.mean(accuracies):.3f}",
        f"{np.mean(precisions):.3f}",
        f"{np.mean(recalls):.3f}",
        f"{np.mean(f1_scores):.3f}"
    ])
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=(12, len(results) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Aspect', 'Samples', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.12]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style dòng mean
    for i in range(6):
        table[(len(results) + 1, i)].set_facecolor('#FFC107')
        table[(len(results) + 1, i)].set_text_props(weight='bold')
    
    # Màu xen kẽ cho các dòng
    for i in range(1, len(results) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(6):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, 'summary_table.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {save_path}")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("PHÂN TÍCH CHI TIẾT KẾT QUẢ ABSA")
    print("="*70)
    
    # =====================================================================
    # 1. VISUALIZE OVERSAMPLING (nếu có)
    # =====================================================================
    oversampling_info_path = os.path.join(RESULTS_DIR, 'oversampling_info.json')
    if os.path.exists(oversampling_info_path):
        print(f"\n{'='*70}")
        print("VISUALIZE OVERSAMPLING")
        print(f"{'='*70}")
        
        try:
            import visualize_oversampling
            print("\nĐang tạo visualization cho oversampling...")
            visualize_oversampling.main()
        except Exception as e:
            print(f"\nWARNING: Không thể tạo oversampling visualization: {str(e)}")
    else:
        print(f"\nWARNING: Không tìm thấy oversampling info, bỏ qua visualization oversampling")
    
    # =====================================================================
    # 2. PHÂN TÍCH KẾT QUẢ TEST
    # =====================================================================
    print(f"\n{'='*70}")
    print("PHÂN TÍCH KẾT QUẢ TEST")
    print(f"{'='*70}")
    
    # Load predictions
    df = load_predictions(
        predictions_file='multi_label/models/multilabel_focal_contrastive/test_predictions_detailed.csv',
        test_file='multi_label/data/test_multilabel.csv'
    )
    
    # Analyze by aspect
    results, confusion_matrices = analyze_by_aspect(df)
    
    # Create visualizations
    create_overall_confusion_matrix(df)
    create_all_confusion_matrices_grid(confusion_matrices)  # Grid confusion matrices
    create_metrics_comparison_plot(results)
    create_sample_distribution_plot(results)
    create_heatmap_metrics(results)
    create_summary_table_image(results)
    
    # Save detailed report
    save_detailed_report(results, df)
    
    # Summary
    print(f"\n{'='*70}")
    print("HOÀN TẤT PHÂN TÍCH!")
    print(f"{'='*70}")
    print(f"\nTất cả kết quả đã được lưu vào: {os.path.abspath(RESULTS_DIR)}/")
    print(f"\nCác file đã tạo:")
    
    files = sorted(os.listdir(RESULTS_DIR))
    for file in files:
        print(f"   • {file}")
    
    print(f"\nTổng số file: {len(files)}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
