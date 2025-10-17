"""
Script Visualize Oversampling Before/After
==========================================
Tạo các biểu đồ so sánh phân bố dữ liệu trước và sau khi oversampling
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RESULTS_DIR = "analysis_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_oversampling_info(file_path='analysis_results/oversampling_info.json'):
    """Load oversampling information từ JSON"""
    print(f"\n{'='*70}")
    print(f"📂 Đang tải thông tin oversampling...")
    print(f"{'='*70}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Không tìm thấy file: {file_path}\n"
            f"Vui lòng chạy train.py trước để tạo file này."
        )
    
    with open(file_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    print(f"✓ Đã tải thông tin oversampling")
    print(f"   Timestamp: {info['timestamp']}")
    print(f"   Strategy: {info['strategy']}")
    print(f"   Before: {info['before']['total_samples']:,} samples")
    print(f"   After:  {info['after']['total_samples']:,} samples")
    print(f"   Increase: +{info['after']['total_samples'] - info['before']['total_samples']:,} samples")
    
    return info


def plot_overall_comparison(info):
    """Vẽ biểu đồ so sánh tổng thể"""
    print(f"\n{'='*70}")
    print("📊 Tạo biểu đồ so sánh tổng thể...")
    print(f"{'='*70}")
    
    sentiments = ['positive', 'negative', 'neutral']
    
    before_counts = [info['before']['sentiment_distribution'].get(s, 0) for s in sentiments]
    after_counts = [info['after']['sentiment_distribution'].get(s, 0) for s in sentiments]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart comparison
    ax1 = axes[0]
    x = np.arange(len(sentiments))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, before_counts, width, label='Before', alpha=0.8, color='steelblue')
    bars2 = ax1.bar(x + width/2, after_counts, width, label='After', alpha=0.8, color='coral')
    
    ax1.set_xlabel('Sentiment', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
    ax1.set_title('Overall Sentiment Distribution: Before vs After Oversampling', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sentiments)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Pie charts (Before and After side by side)
    ax2 = axes[1]
    ax2.axis('off')
    
    # Create 2 subplots inside ax2
    fig2 = plt.figure(figsize=(12, 6))
    
    # Before pie
    ax_before = fig2.add_subplot(121)
    colors_before = ['#66b3ff', '#ff9999', '#99ff99']
    wedges, texts, autotexts = ax_before.pie(
        before_counts, 
        labels=sentiments, 
        autopct='%1.1f%%',
        colors=colors_before,
        startangle=90,
        textprops={'fontsize': 11}
    )
    ax_before.set_title(f'BEFORE\n({info["before"]["total_samples"]:,} samples)', 
                        fontsize=13, fontweight='bold', pad=10)
    
    # After pie
    ax_after = fig2.add_subplot(122)
    colors_after = ['#3399ff', '#ff6666', '#66ff66']
    wedges, texts, autotexts = ax_after.pie(
        after_counts, 
        labels=sentiments, 
        autopct='%1.1f%%',
        colors=colors_after,
        startangle=90,
        textprops={'fontsize': 11}
    )
    ax_after.set_title(f'AFTER\n({info["after"]["total_samples"]:,} samples)', 
                       fontsize=13, fontweight='bold', pad=10)
    
    plt.suptitle('Sentiment Distribution Comparison (Overall)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path2 = os.path.join(RESULTS_DIR, 'oversampling_comparison_pie.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Đã lưu: {save_path2}")
    
    # Save first plot
    plt.figure(fig.number)
    plt.tight_layout()
    save_path1 = os.path.join(RESULTS_DIR, 'oversampling_comparison_overall.png')
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Đã lưu: {save_path1}")


def plot_aspect_wise_comparison(info):
    """Vẽ biểu đồ so sánh theo từng aspect"""
    print(f"\n{'='*70}")
    print("📊 Tạo biểu đồ so sánh theo từng aspect...")
    print(f"{'='*70}")
    
    aspects = sorted(info['before']['aspects'].keys())
    sentiments = ['positive', 'negative', 'neutral']
    
    # Calculate grid size
    n_aspects = len(aspects)
    n_cols = 3
    n_rows = (n_aspects + n_cols - 1) // n_cols
    
    # Create large figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig.suptitle('Aspect-Wise Sentiment Distribution: Before vs After Oversampling', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Flatten axes
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    # Plot each aspect
    for idx, aspect in enumerate(aspects):
        ax = axes_flat[idx]
        
        before_dist = info['before']['aspects'][aspect]
        after_dist = info['after']['aspects'][aspect]
        
        before_counts = [before_dist.get(s, 0) for s in sentiments]
        after_counts = [after_dist.get(s, 0) for s in sentiments]
        
        x = np.arange(len(sentiments))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, before_counts, width, label='Before', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, after_counts, width, label='After', alpha=0.8, color='coral')
        
        # Calculate total samples
        total_before = sum(before_counts)
        total_after = sum(after_counts)
        increase = total_after - total_before
        
        ax.set_title(f'{aspect}\n({total_before:,} → {total_after:,}, +{increase:,})', 
                     fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sentiments, fontsize=9)
        ax.set_ylabel('Samples', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_aspects, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'oversampling_comparison_by_aspect.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Đã lưu: {save_path}")


def plot_heatmap_comparison(info):
    """Vẽ heatmap so sánh phân bố theo aspect và sentiment"""
    print(f"\n{'='*70}")
    print("🔥 Tạo heatmap so sánh...")
    print(f"{'='*70}")
    
    aspects = sorted(info['before']['aspects'].keys())
    sentiments = ['positive', 'negative', 'neutral']
    
    # Create matrices for before and after
    before_matrix = []
    after_matrix = []
    
    for aspect in aspects:
        before_row = [info['before']['aspects'][aspect].get(s, 0) for s in sentiments]
        after_row = [info['after']['aspects'][aspect].get(s, 0) for s in sentiments]
        before_matrix.append(before_row)
        after_matrix.append(after_row)
    
    before_matrix = np.array(before_matrix)
    after_matrix = np.array(after_matrix)
    
    # Create figure with 3 subplots (Before, After, Difference)
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Heatmap 1: Before
    sns.heatmap(before_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sentiments, yticklabels=aspects,
                cbar_kws={'label': 'Samples'}, ax=axes[0])
    axes[0].set_title('BEFORE Oversampling', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Sentiment', fontweight='bold')
    axes[0].set_ylabel('Aspect', fontweight='bold')
    
    # Heatmap 2: After
    sns.heatmap(after_matrix, annot=True, fmt='d', cmap='Oranges',
                xticklabels=sentiments, yticklabels=aspects,
                cbar_kws={'label': 'Samples'}, ax=axes[1])
    axes[1].set_title('AFTER Oversampling', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Sentiment', fontweight='bold')
    axes[1].set_ylabel('Aspect', fontweight='bold')
    
    # Heatmap 3: Difference (Increase)
    diff_matrix = after_matrix - before_matrix
    sns.heatmap(diff_matrix, annot=True, fmt='d', cmap='Greens',
                xticklabels=sentiments, yticklabels=aspects,
                cbar_kws={'label': 'Added Samples'}, ax=axes[2])
    axes[2].set_title('INCREASE (After - Before)', fontsize=14, fontweight='bold', pad=10)
    axes[2].set_xlabel('Sentiment', fontweight='bold')
    axes[2].set_ylabel('Aspect', fontweight='bold')
    
    plt.suptitle('Aspect-Wise Sentiment Distribution Heatmap Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, 'oversampling_heatmap_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Đã lưu: {save_path}")


def plot_balance_improvement(info):
    """Vẽ biểu đồ cải thiện độ cân bằng cho từng aspect"""
    print(f"\n{'='*70}")
    print("📈 Tạo biểu đồ cải thiện độ cân bằng...")
    print(f"{'='*70}")
    
    aspects = sorted(info['before']['aspects'].keys())
    
    imbalance_before = []
    imbalance_after = []
    
    for aspect in aspects:
        before_counts = list(info['before']['aspects'][aspect].values())
        after_counts = list(info['after']['aspects'][aspect].values())
        
        # Calculate imbalance ratio (max/min)
        if min(before_counts) > 0:
            imb_before = max(before_counts) / min(before_counts)
        else:
            imb_before = float('inf')
        
        if min(after_counts) > 0:
            imb_after = max(after_counts) / min(after_counts)
        else:
            imb_after = float('inf')
        
        imbalance_before.append(imb_before)
        imbalance_after.append(imb_after)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(aspects))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, imbalance_before, width, label='Before', alpha=0.8, color='tomato')
    bars2 = ax.bar(x + width/2, imbalance_after, width, label='After', alpha=0.8, color='mediumseagreen')
    
    ax.set_xlabel('Aspect', fontweight='bold', fontsize=12)
    ax.set_ylabel('Imbalance Ratio (max/min)', fontweight='bold', fontsize=12)
    ax.set_title('Class Balance Improvement by Aspect\n(Lower is Better)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(aspects, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at 1.0 (perfect balance)
    ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Balance')
    ax.legend(fontsize=11)
    
    # Add value labels
    for i, (before, after) in enumerate(zip(imbalance_before, imbalance_after)):
        if before != float('inf'):
            ax.text(i - width/2, before + 0.1, f'{before:.2f}', 
                   ha='center', va='bottom', fontsize=9, color='darkred')
        if after != float('inf'):
            ax.text(i + width/2, after + 0.1, f'{after:.2f}', 
                   ha='center', va='bottom', fontsize=9, color='darkgreen')
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'oversampling_balance_improvement.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Đã lưu: {save_path}")


def create_summary_report(info):
    """Tạo báo cáo tổng hợp dạng text"""
    print(f"\n{'='*70}")
    print("💾 Tạo báo cáo tổng hợp...")
    print(f"{'='*70}")
    
    report_path = os.path.join(RESULTS_DIR, 'oversampling_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("BÁO CÁO OVERSAMPLING CHI TIẾT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Timestamp: {info['timestamp']}\n")
        f.write(f"Strategy: {info['strategy']}\n")
        f.write(f"Description: {info['description']}\n\n")
        
        f.write("="*70 + "\n")
        f.write("TỔNG QUAN\n")
        f.write("="*70 + "\n\n")
        
        before_total = info['before']['total_samples']
        after_total = info['after']['total_samples']
        increase = after_total - before_total
        increase_pct = (increase / before_total) * 100
        
        f.write(f"Tổng số samples BEFORE: {before_total:,}\n")
        f.write(f"Tổng số samples AFTER:  {after_total:,}\n")
        f.write(f"Tăng thêm:              +{increase:,} samples (+{increase_pct:.1f}%)\n\n")
        
        f.write("Phân bố sentiment tổng thể:\n")
        f.write(f"{'Sentiment':<12} {'Before':<15} {'After':<15} {'Increase':<15}\n")
        f.write("-"*60 + "\n")
        
        for sentiment in ['positive', 'negative', 'neutral']:
            before = info['before']['sentiment_distribution'].get(sentiment, 0)
            after = info['after']['sentiment_distribution'].get(sentiment, 0)
            inc = after - before
            f.write(f"{sentiment:<12} {before:<15,} {after:<15,} +{inc:<14,}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("CHI TIẾT THEO TỪNG ASPECT\n")
        f.write("="*70 + "\n")
        
        aspects = sorted(info['before']['aspects'].keys())
        
        for aspect in aspects:
            f.write(f"\n{'-'*70}\n")
            f.write(f"Aspect: {aspect}\n")
            f.write(f"{'-'*70}\n")
            
            before_aspect = info['before']['aspects'][aspect]
            after_aspect = info['after']['aspects'][aspect]
            
            before_total = sum(before_aspect.values())
            after_total = sum(after_aspect.values())
            
            f.write(f"Tổng: {before_total:,} → {after_total:,} (+{after_total - before_total:,})\n\n")
            
            f.write(f"{'Sentiment':<12} {'Before':<10} {'After':<10} {'Increase':<10}\n")
            f.write("-"*45 + "\n")
            
            for sentiment in ['positive', 'negative', 'neutral']:
                before = before_aspect.get(sentiment, 0)
                after = after_aspect.get(sentiment, 0)
                inc = after - before
                f.write(f"{sentiment:<12} {before:<10,} {after:<10,} +{inc:<9,}\n")
            
            # Calculate balance improvement
            before_counts = list(before_aspect.values())
            after_counts = list(after_aspect.values())
            
            if min(before_counts) > 0 and min(after_counts) > 0:
                imb_before = max(before_counts) / min(before_counts)
                imb_after = max(after_counts) / min(after_counts)
                f.write(f"\nImbalance ratio: {imb_before:.2f} → {imb_after:.2f}\n")
                if imb_after == 1.0:
                    f.write("✅ Perfect balance achieved!\n")
    
    print(f"✓ Đã lưu: {report_path}")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("📊 VISUALIZE OVERSAMPLING: BEFORE vs AFTER")
    print("="*70)
    
    try:
        # Load oversampling info
        info = load_oversampling_info()
        
        # Create visualizations
        plot_overall_comparison(info)
        plot_aspect_wise_comparison(info)
        plot_heatmap_comparison(info)
        plot_balance_improvement(info)
        
        # Create summary report
        create_summary_report(info)
        
        # Summary
        print(f"\n{'='*70}")
        print("✅ HOÀN TẤT VISUALIZATION!")
        print(f"{'='*70}")
        print(f"\n✓ Tất cả kết quả đã được lưu vào: {os.path.abspath(RESULTS_DIR)}/")
        print(f"\n✓ Các file visualization:")
        
        viz_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('oversampling_')]
        for file in sorted(viz_files):
            print(f"   • {file}")
        
        print(f"\n✓ Tổng số file: {len(viz_files)}")
        print("\n" + "="*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ LỖI: {str(e)}")
        print(f"\n💡 Gợi ý: Chạy 'python train.py' trước để tạo file oversampling_info.json")
    except Exception as e:
        print(f"\n❌ LỖI: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()






