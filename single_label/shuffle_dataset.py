import pandas as pd
import argparse
from pathlib import Path

def shuffle_dataset(input_file, output_file=None, seed=None):
    """
    Shuffle rows in dataset.csv while preserving header.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (default: overwrites input)
        seed: Random seed for reproducibility (default: None for random)
    """
    print(f"Reading dataset from: {input_file}")
    
    # Read with UTF-8-sig to handle BOM
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Total rows: {len(df)}")
    
    # Shuffle rows
    if seed is not None:
        print(f"Shuffling with random seed: {seed}")
        df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        print("Shuffling with random seed (no fixed seed)")
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
    
    # Determine output file
    if output_file is None:
        output_file = input_file
        print(f"Overwriting original file: {output_file}")
    else:
        print(f"Saving to new file: {output_file}")
    
    # Save with UTF-8-sig to preserve BOM
    df_shuffled.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"Shuffled dataset saved successfully!")
    print(f"Output shape: {df_shuffled.shape}")
    
    # Show first few rows after shuffle
    print("\nFirst 3 rows after shuffle:")
    print(df_shuffled.head(3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle dataset.csv rows")
    parser.add_argument(
        "--input", 
        type=str, 
        default="dataset.csv",
        help="Input CSV file (default: dataset.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file (default: overwrite input file)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    
    args = parser.parse_args()
    
    shuffle_dataset(args.input, args.output, args.seed)
