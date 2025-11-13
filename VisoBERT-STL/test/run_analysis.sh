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
#   - multi_label/models/multilabel_focal/test_predictions_detailed.csv (from training)
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
echo "🔬 MULTI-LABEL ANALYSIS RUNNER"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "VisoBERT-STL/test/analyze_results.py" ] && [ ! -f "multi_label/test/analyze_results.py" ]; then
    echo -e "${RED}ERROR: Error: Please run this script from project root (E:\\BERT\\) directory${NC}"
    echo ""
    echo "Usage:"
    echo "  cd E:\\BERT"
    echo "  bash VisoBERT-STL/test/run_analysis.sh"
    exit 1
fi

# Check if predictions file exists (try multiple possible locations)
PREDICTIONS_FOUND=0
PREDICTIONS_PATHS=(
    "VisoBERT-STL/models/sentiment_classification/test_predictions_detailed.csv"
    "VisoBERT-STL/models-o/sentiment_classification/test_predictions_detailed.csv"
    "multi_label/models/multilabel_focal/test_predictions_detailed.csv"
)

for PRED_FILE in "${PREDICTIONS_PATHS[@]}"; do
    if [ -f "$PRED_FILE" ]; then
        echo -e "${GREEN}✓ Found predictions file: $PRED_FILE${NC}"
        PREDICTIONS_FOUND=1
        break
    fi
done

if [ $PREDICTIONS_FOUND -eq 0 ]; then
    echo -e "${YELLOW}WARNING:  Predictions file not found in expected locations:${NC}"
    for PRED_FILE in "${PREDICTIONS_PATHS[@]}"; do
        echo "    • $PRED_FILE"
    done
    echo "  Scripts will auto-detect paths, but please ensure predictions file exists."
    echo ""
fi

# Check if test data exists (try multiple possible locations)
TEST_FOUND=0
TEST_PATHS=(
    "VisoBERT-STL/data/test_multilabel.csv"
    "VisoBERT-STL/data-o/test_multilabel.csv"
    "multi_label/data/test_multilabel.csv"
)

for TEST_FILE in "${TEST_PATHS[@]}"; do
    if [ -f "$TEST_FILE" ]; then
        echo -e "${GREEN}✓ Found test data: $TEST_FILE${NC}"
        TEST_FOUND=1
        break
    fi
done

if [ $TEST_FOUND -eq 0 ]; then
    echo -e "${YELLOW}WARNING:  Test data not found in expected locations:${NC}"
    for TEST_FILE in "${TEST_PATHS[@]}"; do
        echo "    • $TEST_FILE"
    done
    echo "  Scripts will auto-detect paths, but please ensure test data exists."
    echo ""
fi

if [ $PREDICTIONS_FOUND -eq 1 ] && [ $TEST_FOUND -eq 1 ]; then
    echo -e "${GREEN}✓ All required files found${NC}"
else
    echo -e "${YELLOW}⚠ Some files missing, but scripts will attempt auto-detection${NC}"
fi
echo ""

################################################################################
# 1. Run analyze_results.py
################################################################################
echo "========================================================================"
echo " STEP 1/2: Running Results Analysis"
echo "========================================================================"
echo ""

python VisoBERT-STL/test/analyze_results.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN} Results analysis completed successfully!${NC}"
    echo -e "${BLUE} Output: multi_label/analysis_results/${NC}"
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
echo " STEP 2/2: Running Error Analysis"
echo "========================================================================"
echo ""

python VisoBERT-STL/test/error_analysis.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN} Error analysis completed successfully!${NC}"
    echo -e "${BLUE} Output: multi_label/error_analysis_results/${NC}"
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
echo " ANALYSIS COMPLETE!"
echo "========================================================================"
echo ""
echo " Results saved to:"
echo "   • VisoBERT-STL/analysis_results/ (or multi_label/analysis_results/)"
echo "   • VisoBERT-STL/error_analysis_results/ (or multi_label/error_analysis_results/)"
echo ""
echo " Key files:"
echo "   • detailed_analysis_report.txt"
echo "   • error_analysis_report.txt"
echo "   • confusion_matrices_all_aspects.png"
echo "   • all_errors_detailed.csv"
echo "   • improvement_suggestions.txt"
echo ""
echo "NOTE:  Note:"
echo "   • Metrics calculated ONLY on labeled aspects (positive/negative/neutral)"
echo "   • Unlabeled aspects (NaN) are excluded from analysis"
echo ""
echo "========================================================================"
echo ""
