"""
Error Analysis for Sentiment Classification
============================================
Analyzes only labeled aspects (filters out NaN/unlabeled)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

SENTIMENT_MAP = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
ASPECTS = ['Battery', 'Camera', 'Design', 'Display', 'General', 
           'Packaging', 'Performance', 'Price', 'Shipping', 'Shop_Service']

def main():
    print("="*80)
    print("ERROR ANALYSIS - Sentiment Classification")
    print("="*80)
    
    # Paths
    test_file = 'VisoBERT-STL/data/test_multilabel.csv'
    pred_file = 'VisoBERT-STL/models/sentiment_classification/test_predictions_detailed.csv'
    output_dir = 'VisoBERT-STL/error_analysis'
    
    os.makedirs(output_dir, exist_ok=True)
    
    test_df = pd.read_csv(test_file, encoding='utf-8-sig')
    pred_df = pd.read_csv(pred_file, encoding='utf-8-sig')
    
    print(f"\nLoaded {len(test_df)} test samples")
    print(f"Loaded {len(pred_df)} predictions")
    
    # Count labeled vs unlabeled
    total_pairs = len(pred_df) * len(ASPECTS)
    labeled_count = sum(pred_df[f'{asp}_true'].notna().sum() for asp in ASPECTS)
    unlabeled_count = total_pairs - labeled_count
    
    print(f"\nDataset statistics:")
    print(f"  Total aspect-sentiment pairs: {total_pairs:,}")
    print(f"  Labeled: {labeled_count:,} ({labeled_count/total_pairs*100:.1f}%)")
    print(f"  Unlabeled (NaN): {unlabeled_count:,} ({unlabeled_count/total_pairs*100:.1f}%)")
    
    # Step 1: Confusion matrices per aspect (ONLY labeled)
    print("\n" + "="*80)
    print("STEP 1: CONFUSION MATRICES (Only Labeled Aspects)")
    print("="*80)
    
    all_errors = []
    
    for aspect in ASPECTS:
        true_col = f'{aspect}_true'
        pred_col = f'{aspect}_pred'
        
        # Filter ONLY labeled aspects
        mask = pred_df[true_col].notna()
        y_true = pred_df.loc[mask, true_col].astype(int).tolist()
        y_pred = pred_df.loc[mask, pred_col].astype(int).tolist()
        
        if len(y_true) == 0:
            print(f"{aspect}: No labeled samples")
            continue
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        # Find errors
        for true_idx in range(3):
            for pred_idx in range(3):
                if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                    all_errors.append({
                        'aspect': aspect,
                        'true': SENTIMENT_MAP[true_idx],
                        'predicted': SENTIMENT_MAP[pred_idx],
                        'count': cm[true_idx, pred_idx],
                        'error_type': f"{SENTIMENT_MAP[true_idx]} -> {SENTIMENT_MAP[pred_idx]}"
                    })
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                   xticklabels=['Pos', 'Neg', 'Neu'],
                   yticklabels=['Pos', 'Neg', 'Neu'],
                   ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{aspect} Confusion Matrix (Labeled Only)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cm_{aspect}.png', dpi=150)
        plt.close()
        
        correct = sum(np.array(y_true) == np.array(y_pred))
        print(f"{aspect:15} Labeled: {len(y_true):4d}, Correct: {correct:4d} ({correct/len(y_true)*100:.1f}%)")
    
    # Save error patterns
    error_df = pd.DataFrame(all_errors).sort_values('count', ascending=False)
    error_df.to_csv(f'{output_dir}/error_patterns.csv', index=False, encoding='utf-8-sig')
    print(f"\nSaved error patterns")
    
    # Step 2: Extract errors (ONLY labeled)
    print("\n" + "="*80)
    print("STEP 2: EXTRACT ERRORS (Only Labeled Aspects)")
    print("="*80)
    
    errors = []
    for idx, row in pred_df.iterrows():
        sample_id = row['sample_id']
        text = test_df.loc[sample_id, 'data'] if sample_id < len(test_df) else ""
        
        for aspect in ASPECTS:
            true_col = f'{aspect}_true'
            pred_col = f'{aspect}_pred'
            
            true_val = row[true_col]
            pred_val = row[pred_col]
            
            # ONLY count labeled aspects with errors
            if pd.notna(true_val) and true_val != pred_val:
                errors.append({
                    'sample_id': sample_id,
                    'aspect': aspect,
                    'text': text,
                    'true_sentiment': SENTIMENT_MAP[int(true_val)],
                    'pred_sentiment': SENTIMENT_MAP[int(pred_val)],
                    'error_type': f"{SENTIMENT_MAP[int(true_val)]} -> {SENTIMENT_MAP[int(pred_val)]}"
                })
    
    print(f"Found {len(errors)} TRUE errors (labeled aspects only)")
    
    # Sort by frequency
    error_counts = Counter([e['error_type'] for e in errors])
    for err in errors:
        err['frequency'] = error_counts[err['error_type']]
    # Sort by aspect (alphabetical), then frequency (descending), then sample_id (descending)
    aspect_order = {asp: idx for idx, asp in enumerate(ASPECTS)}
    errors_sorted = sorted(errors, key=lambda x: (aspect_order.get(x['aspect'], 999), -x['frequency'], -x['sample_id']))
    
    # Save top 200
    top_errors = pd.DataFrame(errors_sorted[:200])
    top_errors.to_csv(f'{output_dir}/high_confidence_errors.csv', index=False, encoding='utf-8-sig')
    print(f"Saved top 200 errors")
    
    # Top error types
    print(f"\nTop Error Types:")
    for i, (error_type, count) in enumerate(error_counts.most_common(10), 1):
        print(f"  {i}. {error_type:<30} {count:4d} errors")
    
    # Step 3: Samples with multiple errors
    print("\n" + "="*80)
    print("STEP 3: SAMPLES WITH MULTIPLE ERRORS (Labeled Only)")
    print("="*80)
    
    error_df_full = pd.DataFrame(errors)
    sample_error_counts = error_df_full.groupby('sample_id').size()
    problematic = sample_error_counts[sample_error_counts >= 3].sort_values(ascending=False)
    
    print(f"Found {len(problematic)} samples with 3+ labeled aspect errors")
    
    recommendations = []
    for sample_id, count in problematic.head(30).items():
        sample_errors = error_df_full[error_df_full['sample_id'] == sample_id]
        text = sample_errors.iloc[0]['text']
        aspects = ', '.join(sample_errors['aspect'].tolist())
        
        recommendations.append({
            'sample_id': sample_id,
            'num_errors': count,
            'aspects': aspects,
            'text_preview': text[:150] + '...' if len(text) > 150 else text
        })
        
        print(f"  Sample {sample_id}: {count} errors in {aspects}")
    
    rec_df = pd.DataFrame(recommendations)
    rec_df.to_csv(f'{output_dir}/relabel_recommendations.csv', index=False, encoding='utf-8-sig')
    
    # Summary report
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nDataset:")
    print(f"  Total samples: {len(pred_df):,}")
    print(f"  Labeled aspect-sentiment pairs: {labeled_count:,} (17.0%)")
    print(f"  Unlabeled pairs: {unlabeled_count:,} (83.0%)")
    print(f"\nErrors (Labeled aspects only):")
    print(f"  Total errors: {len(errors):,}")
    print(f"  Error rate: {len(errors)/labeled_count*100:.1f}%")
    print(f"  Accuracy: {(labeled_count - len(errors))/labeled_count*100:.1f}%")
    print(f"\nAll outputs saved to: {output_dir}/")
    
    # Create review template
    template_lines = [
        "="*80,
        "ERROR CLASSIFICATION FOR HUMAN REVIEW",
        "="*80,
        "",
        f"Note: This analysis ONLY includes LABELED aspects (not NaN/unlabeled)",
        f"Total errors to review: {min(100, len(errors_sorted))}",
        "",
        "CLASSIFICATION CODES:",
        "  [A] Annotation Mistake - Label is wrong, model correct",
        "  [B] Ambiguous - Both reasonable",
        "  [C] Model Lacks Context",
        "  [D] Tokenization Issue",
        "  [E] Model Error - Model wrong",
        "",
        "="*80,
        ""
    ]
    
    for i, err in enumerate(errors_sorted[:100], 1):
        template_lines.extend([
            f"--- ERROR {i} ---",
            f"Sample ID: {err['sample_id']}",
            f"Aspect: {err['aspect']}",
            f"True: {err['true_sentiment']} | Predicted: {err['pred_sentiment']}",
            f"Error Type: {err['error_type']} (appears {err['frequency']}x)",
            "",
            "Text:",
            err['text'][:300] + ('...' if len(err['text']) > 300 else ''),
            "",
            "ERROR_CLASS: _____ (A/B/C/D/E)",
            "RELABEL_TO: _____",
            "NOTES: ________________________________________________",
            "",
            "-"*80,
            ""
        ])
    
    with open(f'{output_dir}/error_classification_template.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(template_lines))
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")

if __name__ == '__main__':
    main()
