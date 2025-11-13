"""
Remove empty samples from a CSV dataset.

Behavior:
- Read input CSV with UTF-8-SIG (handles BOM if present)
- Identify and remove rows that are entirely empty (all columns empty/whitespace/NaN)
- Save output CSV with UTF-8-SIG (ensures BOM for Excel compatibility)

Usage:
    python remove_empty_samples.py input.csv output.csv
    # If output.csv omitted, input.csv will be overwritten
"""

import sys
from typing import Tuple

import pandas as pd


def detect_empty_rows(df: pd.DataFrame) -> pd.Series:
    """
    A row is considered empty if ALL columns are empty after stripping whitespace,
    or are NaN. This also catches raw blank CSV lines (',,,,,,,,,,,') that pandas
    parses into all-NaN rows.
    """
    # Treat NaN as empty
    is_na = df.isna()

    # Treat empty strings/whitespace as empty
    as_str = df.astype(str)
    is_ws_empty = as_str.apply(lambda col: col.str.strip() == "", axis=0)

    # A cell is empty if it's NaN OR whitespace-empty
    is_cell_empty = is_na | is_ws_empty

    # A row is empty if ALL cells in that row are empty
    return is_cell_empty.all(axis=1)


def remove_empty_samples(input_file: str = "dataset.csv", output_file: str = "dataset.csv") -> Tuple[pd.DataFrame, int]:
    print(f"Reading {input_file} (UTF-8-SIG)...")
    # Keep default NaN handling; we convert empty strings to empty via strip
    df = pd.read_csv(input_file, encoding="utf-8-sig", keep_default_na=True)

    print(f"Original dataset size: {len(df)} rows")

    empty_mask = detect_empty_rows(df)
    num_empty = int(empty_mask.sum())

    if num_empty > 0:
        print(f"Found {num_empty} empty rows to remove")
        # Show first few indices for transparency
        print("First empty row indices:", df.index[empty_mask][:10].tolist())
        df_cleaned = df[~empty_mask].copy()
    else:
        print("No empty rows found. Dataset is already clean.")
        df_cleaned = df

    print(f"Saving cleaned dataset to {output_file} (UTF-8-SIG)...")
    df_cleaned.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Done. Cleaned dataset size: {len(df_cleaned)} rows")

    return df_cleaned, num_empty


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path
    remove_empty_samples(input_path, output_path)

