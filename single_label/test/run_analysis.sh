#!/bin/bash

################################################################################
# Single-Label Analysis Runner
################################################################################
# Chạy analyze_results.py và error_analysis.py cho single-label ABSA
# 
# Usage:
#   From D:\BERT\:
#     bash single_label/test/run_analysis.sh
#
# Requirements:
#   - single_label/results/test_predictions_single.csv (from training)
#   - single_label/data/test.csv (ground truth)
################################################################################

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "========================================================================"
echo "🔬 SINGLE-LABEL ANALYSIS RUNNER"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "single_label/test/analyze_results.py" ]; then
    echo -e "${RED}❌ Error: Please run this script from D:\BERT\ directory${NC}"
    echo ""
    echo "Usage:"
    echo "  cd D:\BERT"
    echo "  bash single_label/test/run_analysis.sh"
    exit 1
fi

# Check if predictions file exists
if [ ! -f "single_label/results/test_predictions_single.csv" ]; then
    echo -e "${RED}❌ Error: Predictions file not found!${NC}"
    echo ""
    echo "Expected: single_label/results/test_predictions_single.csv"
    echo ""
    echo "Please run training first:"
    echo "  python single_label\\train.py --config single_label\\config_single.yaml"
    exit 1
fi

# Check if test data exists
if [ ! -f "single_label/data/test.csv" ]; then
    echo -e "${RED}❌ Error: Test data not found!${NC}"
    echo ""
    echo "Expected: single_label/data/test.csv"
    exit 1
fi

echo -e "${GREEN}✓ All required files found${NC}"
echo ""

################################################################################
# 1. Run analyze_results.py
################################################################################
echo "========================================================================"
echo "📊 STEP 1/2: Running Results Analysis"
echo "========================================================================"
echo ""

python single_label/test/analyze_results.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Results analysis completed successfully!${NC}"
    echo -e "${BLUE}📁 Output: single_label/analysis_results/${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Results analysis failed!${NC}"
    echo ""
    exit 1
fi

################################################################################
# 2. Run error_analysis.py
################################################################################
echo "========================================================================"
echo "🔍 STEP 2/2: Running Error Analysis"
echo "========================================================================"
echo ""

python single_label/test/error_analysis.py

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Error analysis completed successfully!${NC}"
    echo -e "${BLUE}📁 Output: single_label/error_analysis_results/${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Error analysis failed!${NC}"
    echo ""
    exit 1
fi

################################################################################
# Summary
################################################################################
echo "========================================================================"
echo "✅ ANALYSIS COMPLETE!"
echo "========================================================================"
echo ""
echo "📊 Results saved to:"
echo "   • single_label/analysis_results/"
echo "   • single_label/error_analysis_results/"
echo ""
echo "📁 Key files:"
echo "   • detailed_analysis_report.txt"
echo "   • error_analysis_report.txt"
echo "   • confusion_matrices_all_aspects.png"
echo "   • all_errors_detailed.csv"
echo "   • improvement_suggestions.txt"
echo ""
echo "========================================================================"
echo ""
