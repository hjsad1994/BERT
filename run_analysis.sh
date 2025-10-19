#!/bin/bash
# Script to run both analyze_results.py and error_analysis.py
# For Linux/Mac/Git Bash on Windows

set -e  # Exit on error

echo "========================================================================"
echo "🚀 STARTING COMPLETE ANALYSIS PIPELINE"
echo "========================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if required files exist
echo -e "${BLUE}📋 Checking requirements...${NC}"
if [ ! -f "test_predictions.csv" ]; then
    echo -e "${RED}❌ Error: test_predictions.csv not found${NC}"
    echo "Please run: python generate_test_predictions.py"
    exit 1
fi

if [ ! -f "data/test.csv" ]; then
    echo -e "${RED}❌ Error: data/test.csv not found${NC}"
    echo "Please run: python prepare_data.py"
    exit 1
fi

echo -e "${GREEN}✓ All required files found${NC}"
echo ""

# ========================================================================
# 1. Run analyze_results.py
# ========================================================================
echo "========================================================================"
echo -e "${BLUE}📊 [1/2] Running analyze_results.py${NC}"
echo "========================================================================"
echo ""

python analyze_results.py
ANALYZE_EXIT_CODE=$?

if [ $ANALYZE_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}❌ analyze_results.py failed with exit code $ANALYZE_EXIT_CODE${NC}"
    exit $ANALYZE_EXIT_CODE
else
    echo ""
    echo -e "${GREEN}✓ analyze_results.py completed successfully${NC}"
fi

echo ""
echo ""

# ========================================================================
# 2. Run error_analysis.py
# ========================================================================
echo "========================================================================"
echo -e "${BLUE}🔍 [2/2] Running error_analysis.py${NC}"
echo "========================================================================"
echo ""

python tests/error_analysis.py
ERROR_EXIT_CODE=$?

if [ $ERROR_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}❌ error_analysis.py failed with exit code $ERROR_EXIT_CODE${NC}"
    exit $ERROR_EXIT_CODE
else
    echo ""
    echo -e "${GREEN}✓ error_analysis.py completed successfully${NC}"
fi

# ========================================================================
# Summary
# ========================================================================
echo ""
echo "========================================================================"
echo -e "${GREEN}✅ ALL ANALYSIS COMPLETED SUCCESSFULLY!${NC}"
echo "========================================================================"
echo ""
echo -e "${YELLOW}📁 Results saved to:${NC}"
echo "   • analysis_results/          (analyze_results.py output)"
echo "   • error_analysis_results/    (error_analysis.py output)"
echo ""
echo -e "${YELLOW}📊 Generated files:${NC}"

# Count files in analysis_results
if [ -d "analysis_results" ]; then
    ANALYSIS_COUNT=$(ls -1 analysis_results | wc -l)
    echo "   • analysis_results/: $ANALYSIS_COUNT files"
fi

# Count files in error_analysis_results
if [ -d "error_analysis_results" ]; then
    ERROR_COUNT=$(ls -1 error_analysis_results | wc -l)
    echo "   • error_analysis_results/: $ERROR_COUNT files"
fi

echo ""
echo "========================================================================"
