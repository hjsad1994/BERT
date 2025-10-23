@echo off
REM Full Pipeline with Reproducible Seeds
REM ======================================
REM This runs the complete pipeline: prepare -> oversample -> train

cd /d D:\BERT\single_label

echo.
echo ========================================================================
echo REPRODUCIBLE ABSA PIPELINE - SINGLE LABEL
echo ========================================================================
echo.
echo This will run:
echo   1. Data preparation (split with seed=42)
echo   2. Aspect-wise oversampling (seed=42)
echo   3. Model training (seed=42)
echo.
pause

python run_full_pipeline.py --config config_single.yaml

pause
