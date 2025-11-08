"""
Script to remove samples with no data from dataset.csv
Removes rows that are completely empty (only commas: ,,,,,,,,,,,)
"""

import pandas as pd
import sys

def remove_empty_samples(input_file='dataset.csv', output_file='dataset.csv'):
    """
    Remove samples that have no data (rows with only commas)
    Processes the file directly to ensure all empty rows are removed
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (default: overwrites input)
    """
    print(f"Reading {input_file}...")
    
    # First, read the raw file to count empty lines
    # Use utf-8-sig to handle BOM if present
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    
    # Count empty lines (lines with only commas)
    # A line with only commas: after stripping, it should be only commas (11 commas for 11 columns)
    empty_line_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip header line
        if i == 0:
            continue
        # Check if line has only commas (no other characters except commas and newline)
        if stripped and stripped.replace(',', '').strip() == '' and stripped.count(',') >= 10:
            empty_line_indices.append(i)
    
    print(f"Found {len(empty_line_indices)} empty lines (rows with only commas)")
    
    if len(empty_line_indices) > 0:
        # Show examples
        print("\nExamples of empty lines (first 5):")
        for idx in empty_line_indices[:5]:
            print(f"  Line {idx + 1}: {repr(lines[idx].strip())}")
        
        # Read CSV with pandas, skipping empty lines
        # We'll filter them out after reading
        # Use utf-8-sig to handle BOM if present
        df = pd.read_csv(input_file, keep_default_na=False, encoding='utf-8-sig')
        
        # Also check for empty rows in the dataframe
        all_columns = df.columns.tolist()
        
        def is_row_empty(row):
            for col in all_columns:
                value = str(row[col]).strip()
                if value and value.lower() != 'nan':
                    return False
            return True
        
        empty_mask = df.apply(is_row_empty, axis=1)
        num_empty_in_df = empty_mask.sum()
        
        print(f"\nEmpty rows detected in dataframe: {num_empty_in_df}")
        
        # Remove empty samples
        df_cleaned = df[~empty_mask].copy()
        
        print(f"\nCleaned dataset size: {len(df_cleaned)} rows")
        print(f"Removed {num_empty_in_df} empty samples")
        
        # Save cleaned dataset with UTF-8-SIG encoding (with BOM)
        df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nSaved cleaned dataset to {output_file} (UTF-8-SIG format)")
        
        return df_cleaned
    else:
        print("No empty samples found. Dataset is already clean.")
        # Still save with UTF-8-SIG encoding to ensure consistent format
        df = pd.read_csv(input_file, encoding='utf-8-sig', keep_default_na=False)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nSaved dataset to {output_file} (UTF-8-SIG format)")
        return df

if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'dataset.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'dataset.csv'
    
    remove_empty_samples(input_file, output_file)

