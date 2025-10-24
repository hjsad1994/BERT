@echo off
echo ======================================================================
echo  COMPREHENSIVE ERROR ANALYSIS
echo ======================================================================
echo.

echo [Step 1/2] Generating test predictions from trained model...
echo.
python multi_label/generate_predictions.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to generate predictions!
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo [Step 2/2] Running error analysis...
echo.
python multi_label/test/error_analysis.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to run error analysis!
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo SUCCESS! Error analysis complete.
echo.
echo Results saved to: multi_label/error_analysis_results/
echo.
echo Key files:
echo   - all_errors_detailed.csv        (ALL 240 errors)
echo   - hard_cases.csv                 (Top 5 per aspect)
echo   - improvement_suggestions.txt    (Actionable recommendations)
echo   - error_analysis_report.txt      (Summary statistics)
echo   - *.png                          (Visualizations)
echo.
echo ======================================================================
pause
