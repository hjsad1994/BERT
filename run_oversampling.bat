@echo off
REM Aspect-Wise Oversampling Script
REM ================================

echo =============================================
echo  ASPECT-WISE OVERSAMPLING
echo =============================================
echo.

echo Step 1: Analyzing current distribution...
python aspect_wise_oversampling.py

echo.
echo Step 2: Update config.yaml
echo   Change: train_file: "data/train_oversampled_aspect_wise.csv"
echo.
echo Press any key to open config.yaml...
pause
notepad config.yaml

echo.
echo Step 3: Train with oversampled data
echo   This will take 3.5-4 hours (50%% longer than normal)
echo.
set /p train="Start training now? (y/n): "
if /i "%train%"=="y" (
    python train.py
)

echo.
echo =============================================
echo  DONE!
echo =============================================
echo.
echo Next: Compare results with/without oversampling
pause
