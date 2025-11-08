#!/bin/bash

# Script to run multi-label data preparation pipeline
# This script runs:
#   1. prepare_data_multilabel.py - Split dataset into train/val/test
#   2. augment_multilabel_balanced.py - Create balanced training data

set -e  # Exit on error

echo "================================================================================"
echo "Multi-Label Data Preparation Pipeline"
echo "================================================================================"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Prepare data (split into train/val/test)
echo "================================================================================"
echo "Step 1: Preparing data - Splitting dataset into train/val/test"
echo "================================================================================"
echo ""

python prepare_data_multilabel.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: prepare_data_multilabel.py failed!"
    exit 1
fi

echo ""
echo "Step 1 completed successfully!"
echo ""

# Step 2: Augment data (create balanced training data)
echo "================================================================================"
echo "Step 2: Augmenting data - Creating balanced training data"
echo "================================================================================"
echo ""

python augment_multilabel_balanced.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: augment_multilabel_balanced.py failed!"
    exit 1
fi

echo ""
echo "Step 2 completed successfully!"
echo ""

# Final summary
echo "================================================================================"
echo "Pipeline Completed Successfully!"
echo "================================================================================"
echo ""
echo "Data files have been created in the following directories:"
echo "  - BILSTM-MTL/data/"
echo "  - BILSTM-STL/data/"
echo "  - VisoBERT-MTL/data/"
echo "  - VisoBERT-STL/data/"
echo ""
echo "Each directory contains:"
echo "  - train_multilabel.csv"
echo "  - train_multilabel_balanced.csv"
echo "  - validation_multilabel.csv"
echo "  - test_multilabel.csv"
echo "  - multilabel_metadata.json"
echo ""
echo "================================================================================"

