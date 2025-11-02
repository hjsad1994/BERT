#!/bin/bash

################################################################################
# Multi-Label Analysis Runner
################################################################################
# Chạy analyze_results.py và error_analysis.py cho multi-label ABSA
# 
# Usage:
#   From D:\BERT\:
#     bash multi_label/test/run_analysis.sh
#
# Requirements:
#   - multi_label/models/multilabel_focal_contrastive/test_predictions_detailed.csv (from training)
#   - multi_label/data/test_multilabel.csv (ground truth)
################################################################################

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "========================================================================"
echo "MULTI-LABEL ANALYSIS RUNNER"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "multi_label/test/analyze_results.py" ]; then
    echo -e "${RED}ERROR: Please run this script from D:\BERT\ directory${NC}"
    echo ""
    echo "Usage:"
    echo "  cd D:\BERT"
    echo "  bash multi_label/test/run_analysis.sh"
    exit 1
fi

# Check if predictions file exists
PREDICTIONS_FILE="multi_label/models/multilabel_focal_contrastive/test_predictions_detailed.csv"
if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo -e "${RED}ERROR: Predictions file not found!${NC}"
    echo ""
    echo "Expected: $PREDICTIONS_FILE"
    echo ""
    echo "Please run training first:"
    echo "  python multi_label/train_multilabel.py"
    exit 1
fi

# Check if test data exists
TEST_FILE="multi_label/data/test_multilabel.csv"
if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED}ERROR: Test data not found!${NC}"
    echo ""
    echo "Expected: $TEST_FILE"
    exit 1
fi

echo -e "${GREEN}✓ All required files found${NC}"
echo ""

################################################################################
# 1. Run analyze_results.py
################################################################################
echo "========================================================================"
echo "STEP 1/2: Running Results Analysis"
echo "========================================================================"
echo ""

python multi_label/test/analyze_results.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Results analysis completed successfully!${NC}"
    echo -e "${BLUE}Output: multi_label/analysis_results/${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}ERROR: Results analysis failed!${NC}"
    echo ""
    exit 1
fi

################################################################################
# 2. Run error_analysis.py
################################################################################
echo "========================================================================"
echo "STEP 2/2: Running Error Analysis"
echo "========================================================================"
echo ""

python multi_label/test/error_analysis.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Error analysis completed successfully!${NC}"
    echo -e "${BLUE}Output: multi_label/error_analysis_results/${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}ERROR: Error analysis failed!${NC}"
    echo ""
    exit 1
fi

################################################################################
# Summary
################################################################################
echo "========================================================================"
echo "ANALYSIS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "   • multi_label/analysis_results/"
echo "   • multi_label/error_analysis_results/"
echo ""
echo "Key files:"
echo "   • detailed_analysis_report.txt"
echo "   • error_analysis_report.txt"
echo "   • confusion_matrices_all_aspects.png"
echo "   • all_errors_detailed.csv"
echo "   • improvement_suggestions.txt"
echo ""
echo "Note:"
echo "   • Metrics calculated ONLY on labeled aspects (positive/negative/neutral)"
echo "   • Unlabeled aspects (NaN) are excluded from analysis"
echo ""
echo "========================================================================"
echo ""
