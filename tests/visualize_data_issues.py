# -*- coding: utf-8 -*-
"""
Visualize data issues for ABSA dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read data
print('Reading data...')
train = pd.read_csv('data/train.csv', encoding='utf-8')

# Create figure
fig = plt.figure(figsize=(20, 12))

# ============================================================================
# 1. Aspect Distribution
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
aspect_counts = train['aspect'].value_counts()
aspect_counts.plot(kind='barh', ax=ax1, color='skyblue')
ax1.set_title('Aspect Distribution (Imbalance)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Count')
ax1.set_ylabel('Aspect')
ax1.grid(True, alpha=0.3)

# Add value labels
for i, v in enumerate(aspect_counts):
    ax1.text(v + 20, i, str(v), va='center')

# ============================================================================
# 2. Sentiment Distribution
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
sentiment_counts = train['sentiment'].value_counts()
colors = ['#90EE90', '#FFB6C1', '#87CEEB']
sentiment_counts.plot(kind='bar', ax=ax2, color=colors)
ax2.set_title('Sentiment Distribution (Imbalance)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Sentiment')
ax2.set_ylabel('Count')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, v in enumerate(sentiment_counts):
    ax2.text(i, v + 100, str(v), ha='center', fontweight='bold')

# ============================================================================
# 3. Sentence Length Distribution
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
train['length'] = train['sentence'].str.len()
ax3.hist(train['length'], bins=50, color='coral', edgecolor='black', alpha=0.7)
ax3.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Min threshold (10)')
ax3.axvline(x=500, color='red', linestyle='--', linewidth=2, label='Max threshold (500)')
ax3.set_title('Sentence Length Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Length (characters)')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ============================================================================
# 4. Aspect-Sentiment Heatmap
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
aspect_sentiment = pd.crosstab(train['aspect'], train['sentiment'])
sns.heatmap(aspect_sentiment, annot=True, fmt='d', cmap='YlOrRd', ax=ax4, 
            cbar_kws={'label': 'Count'})
ax4.set_title('Aspect-Sentiment Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Sentiment')
ax4.set_ylabel('Aspect')

# ============================================================================
# 5. Imbalance Ratio by Aspect
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
aspect_max_min = []
for aspect in train['aspect'].unique():
    aspect_df = train[train['aspect'] == aspect]
    sentiment_counts = aspect_df['sentiment'].value_counts()
    if len(sentiment_counts) > 1:
        ratio = sentiment_counts.max() / sentiment_counts.min()
        aspect_max_min.append({'aspect': aspect, 'ratio': ratio})

imbalance_df = pd.DataFrame(aspect_max_min).sort_values('ratio', ascending=False)
imbalance_df.plot(x='aspect', y='ratio', kind='barh', ax=ax5, 
                  color='orange', legend=False)
ax5.axvline(x=2, color='red', linestyle='--', linewidth=2, label='Threshold (2:1)')
ax5.set_title('Sentiment Imbalance Ratio per Aspect', fontsize=14, fontweight='bold')
ax5.set_xlabel('Max/Min Ratio')
ax5.set_ylabel('Aspect')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ============================================================================
# 6. Data Quality Issues Summary
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Count issues
short_sentences = len(train[train['length'] < 10])
long_sentences = len(train[train['length'] > 500])
duplicates = train.duplicated().sum()

# Emoji check
import re
emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    "]+", flags=re.UNICODE)
emoji_sentences = len(train[train['sentence'].str.contains(emoji_pattern, regex=True, na=False)])

# Imbalance ratios
max_aspect = aspect_counts.max()
min_aspect = aspect_counts.min()
aspect_ratio = max_aspect / min_aspect

max_sent = sentiment_counts.max()
min_sent = sentiment_counts.min()
sent_ratio = max_sent / min_sent

# Create summary text
summary_text = f"""
DATA QUALITY ISSUES SUMMARY

Dataset Size: {len(train):,} samples

IMBALANCE ISSUES:
• Aspect imbalance: {aspect_ratio:.2f}:1
  - Max: {max_aspect} ({aspect_counts.index[0]})
  - Min: {min_aspect} ({aspect_counts.index[-1]})

• Sentiment imbalance: {sent_ratio:.2f}:1
  - Max: {max_sent} ({sentiment_counts.index[0]})
  - Min: {min_sent} ({sentiment_counts.index[-1]})

TEXT QUALITY ISSUES:
• Short sentences (<10): {short_sentences} ({short_sentences/len(train)*100:.2f}%)
• Long sentences (>500): {long_sentences} ({long_sentences/len(train)*100:.2f}%)
• Sentences with emoji: {emoji_sentences} ({emoji_sentences/len(train)*100:.2f}%)
• Duplicate samples: {duplicates}

SENTENCE LENGTH STATS:
• Mean: {train['length'].mean():.2f} chars
• Median: {train['length'].median():.0f} chars
• Min: {train['length'].min()} chars
• Max: {train['length'].max()} chars

RECOMMENDATION:
✓ Apply aspect-wise oversampling
✓ Clean emoji & special characters
✓ Remove short sentences
✓ Truncate/remove long sentences
✓ Remove duplicates
"""

ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         family='monospace')

# ============================================================================
# Save figure
# ============================================================================
plt.tight_layout()
output_file = 'data_quality_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'\n✓ Saved visualization to: {output_file}')

plt.show()

print('\n' + '='*70)
print('VISUALIZATION COMPLETE!')
print('='*70)
