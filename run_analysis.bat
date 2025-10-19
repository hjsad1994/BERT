@echo off
REM Script to run both analyze_results.py and error_analysis.py
REM For Windows Command Prompt

setlocal enabledelayedexpansion

echo ========================================================================
echo 🚀 STARTING COMPLETE ANALYSIS PIPELINE
echo ========================================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if required files exist
echo 📋 Checking requirements...
if not exist "test_predictions.csv" (
    echo ❌ Error: test_predictions.csv not found
    echo Please run: python generate_test_predictions.py
    exit /b 1
)

if not exist "data\test.csv" (
    echo ❌ Error: data\test.csv not found
    echo Please run: python prepare_data.py
    exit /b 1
)

echo ✓ All required files found
echo.

REM ========================================================================
REM 1. Run analyze_results.py
REM ========================================================================
echo ========================================================================
echo 📊 [1/2] Running analyze_results.py
echo ========================================================================
echo.

python analyze_results.py
if errorlevel 1 (
    echo.
    echo ❌ analyze_results.py failed with exit code !errorlevel!
    exit /b !errorlevel!
)

echo.
echo ✓ analyze_results.py completed successfully
echo.
echo.

REM ========================================================================
REM 2. Run error_analysis.py
REM ========================================================================
echo ========================================================================
echo 🔍 [2/2] Running error_analysis.py
echo ========================================================================
echo.

python tests\error_analysis.py
if errorlevel 1 (
    echo.
    echo ❌ error_analysis.py failed with exit code !errorlevel!
    exit /b !errorlevel!
)

echo.
echo ✓ error_analysis.py completed successfully
echo.
echo 📄 NEW: Now exports ALL errors to CSV files!
echo    • all_errors_detailed.csv - ALL prediction errors with full details
echo    • errors_summary_by_aspect.csv - Error summary grouped by aspect

REM ========================================================================
REM Summary
REM ========================================================================
echo.
echo ========================================================================
echo ✅ ALL ANALYSIS COMPLETED SUCCESSFULLY!
echo ========================================================================
echo.
echo 📁 Results saved to:
echo    • analysis_results\          (analyze_results.py output)
echo    • error_analysis_results\    (error_analysis.py output)
echo.
echo 📊 Generated files:

REM Count files in analysis_results
if exist "analysis_results" (
    for /f %%A in ('dir /b /a-d "analysis_results" 2^>nul ^| find /c /v ""') do set ANALYSIS_COUNT=%%A
    echo    • analysis_results\: !ANALYSIS_COUNT! files
)

REM Count files in error_analysis_results
if exist "error_analysis_results" (
    for /f %%A in ('dir /b /a-d "error_analysis_results" 2^>nul ^| find /c /v ""') do set ERROR_COUNT=%%A
    echo    • error_analysis_results\: !ERROR_COUNT! files
)

echo.
echo 🆕 KEY FILES FOR DETAILED ERROR ANALYSIS:
echo    • all_errors_detailed.csv - Contains ALL errors for deep analysis
echo    • errors_summary_by_aspect.csv - Error patterns by aspect
echo    • hard_cases.csv - Top 5 difficult cases per aspect
echo.
echo 💡 TIP: Open all_errors_detailed.csv in Excel to analyze every single error!
echo.
echo ========================================================================
echo.
pause
