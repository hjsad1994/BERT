"""
Filter Long Sequences - Remove rows with >256 tokens
=====================================================
This script removes sequences exceeding 256 tokens from dataset.csv
to ensure compatibility with the model's max_length setting.
"""

import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
import json
from datetime import datetime

def filter_long_sequences(
    input_file: str = "dataset.csv",
    output_file: str = "dataset_filtered.csv",
    backup_file: str = "backups/dataset_before_filter.csv",
    max_length: int = 256,
    model_name: str = "5CD-AI/Vietnamese-Sentiment-visobert"
):
    """
    Filter out sequences exceeding max_length tokens.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        backup_file: Path to backup original file
        max_length: Maximum token length allowed
        model_name: Model name for tokenizer
    """
    
    print(f"\n{'='*70}")
    print(f"FILTER LONG SEQUENCES - Max Length: {max_length} tokens")
    print(f"{'='*70}\n")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}\n")
    
    # Read dataset
    print(f"Reading dataset: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    original_count = len(df)
    print(f"Original rows: {original_count:,}\n")
    
    # Create backup
    backup_path = Path(backup_file)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(backup_file, index=False, encoding='utf-8-sig')
    print(f"Backup created: {backup_file}\n")
    
    # Check token lengths
    print("Analyzing token lengths...")
    token_lengths = []
    long_sequences = []
    
    for idx, row in df.iterrows():
        text = str(row['data']) if pd.notna(row['data']) else ""
        
        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_count = len(tokens)
        token_lengths.append(token_count)
        
        # Track long sequences
        if token_count > max_length:
            long_sequences.append({
                'index': idx,
                'token_count': token_count,
                'text': text[:100] + "..." if len(text) > 100 else text
            })
        
        # Progress
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1:,}/{original_count:,} rows...")
    
    print(f"\nToken length analysis:")
    print(f"  Min tokens: {min(token_lengths)}")
    print(f"  Max tokens: {max(token_lengths)}")
    print(f"  Mean tokens: {sum(token_lengths) / len(token_lengths):.2f}")
    print(f"  Median tokens: {sorted(token_lengths)[len(token_lengths) // 2]}")
    
    # Count sequences by length range
    ranges = [(0, 128), (129, 256), (257, 512), (513, 1000), (1001, float('inf'))]
    print(f"\nToken distribution:")
    for start, end in ranges:
        count = sum(1 for x in token_lengths if start <= x <= end)
        pct = (count / original_count) * 100
        end_str = f"{end}" if end != float('inf') else "+"
        print(f"  {start:4d}-{end_str:4s} tokens: {count:6,} ({pct:5.2f}%)")
    
    # Filter out long sequences
    print(f"\nFiltering sequences > {max_length} tokens...")
    df_filtered = df[df.index.isin([i for i in range(len(df)) if token_lengths[i] <= max_length])]
    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count
    
    print(f"\nResults:")
    print(f"  Original rows: {original_count:,}")
    print(f"  Filtered rows: {filtered_count:,}")
    print(f"  Removed rows: {removed_count:,} ({(removed_count/original_count)*100:.2f}%)")
    
    # Save filtered dataset
    df_filtered.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nFiltered dataset saved: {output_file}")
    
    # Save statistics
    stats = {
        'timestamp': datetime.now().isoformat(),
        'input_file': input_file,
        'output_file': output_file,
        'backup_file': backup_file,
        'max_length': max_length,
        'original_count': int(original_count),
        'filtered_count': int(filtered_count),
        'removed_count': int(removed_count),
        'removal_percentage': float((removed_count/original_count)*100),
        'token_stats': {
            'min': int(min(token_lengths)),
            'max': int(max(token_lengths)),
            'mean': float(sum(token_lengths) / len(token_lengths)),
            'median': int(sorted(token_lengths)[len(token_lengths) // 2])
        },
        'long_sequences_sample': long_sequences[:10]  # First 10 examples
    }
    
    stats_file = 'analysis_results/filter_long_sequences_stats.json'
    Path(stats_file).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Statistics saved: {stats_file}")
    
    # Show examples of removed sequences
    if long_sequences:
        print(f"\nExamples of removed sequences (first 5):")
        for i, seq in enumerate(long_sequences[:5], 1):
            print(f"\n  {i}. Index {seq['index']}: {seq['token_count']} tokens")
            try:
                print(f"     Text: {seq['text']}")
            except UnicodeEncodeError:
                print(f"     Text: [Vietnamese text - cannot display in console]")
    
    print(f"\n{'='*70}")
    print("FILTERING COMPLETED")
    print(f"{'='*70}\n")
    
    return df_filtered, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter long sequences from dataset')
    parser.add_argument('--input', default='dataset.csv', help='Input CSV file')
    parser.add_argument('--output', default='dataset_filtered.csv', help='Output CSV file')
    parser.add_argument('--backup', default='backups/dataset_before_filter.csv', help='Backup file path')
    parser.add_argument('--max-length', type=int, default=256, help='Maximum token length')
    parser.add_argument('--model', default='5CD-AI/Vietnamese-Sentiment-visobert', help='Model name for tokenizer')
    parser.add_argument('--inplace', action='store_true', help='Replace original dataset.csv')
    
    args = parser.parse_args()
    
    # If inplace, output to dataset.csv
    output_file = 'dataset.csv' if args.inplace else args.output
    
    # Filter sequences
    df_filtered, stats = filter_long_sequences(
        input_file=args.input,
        output_file=output_file,
        backup_file=args.backup,
        max_length=args.max_length,
        model_name=args.model
    )
    
    if args.inplace:
        print(f"\nWARNING: Original dataset.csv has been replaced!")
        print(f"Backup available at: {args.backup}")
