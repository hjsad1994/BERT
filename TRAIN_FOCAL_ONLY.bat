@echo off
REM Train with Focal Loss ONLY (no contrastive) - Pure Baseline

echo ================================================================================
echo Focal Loss ONLY Training (No Contrastive)
echo ================================================================================
echo.

echo Step 1: Verify config...
echo.
python test_focal_only.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Config verification failed!
    echo Please check multi_label\config_focal_only.yaml
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Config verified! Focal Loss ONLY (no contrastive)
echo ================================================================================
echo.

echo Settings:
echo   - Loss: Focal 100%%
echo   - Contrastive: DISABLED
echo   - Gamma: 2.0
echo   - Output: multi_label\models\multilabel_focal_only\
echo.

echo This is the BASELINE for comparison!
echo.

echo Start training? (15 epochs, ~75 minutes)
set /p choice="Type 'y' to start, 'n' to exit: "

if /i "%choice%"=="y" (
    echo.
    echo Starting baseline training...
    echo.
    python multi_label\train_multilabel_focal_contrastive.py --config multi_label/config_focal_only.yaml --epochs 15
    
    echo.
    echo ================================================================================
    echo Training Complete!
    echo ================================================================================
    echo.
    echo Check results:
    echo   type multi_label\models\multilabel_focal_only\test_results_focal_contrastive.json
    echo.
    echo This is your BASELINE result.
    echo Compare with:
    echo   - Focal+Contrastive: 95.99%% F1
    echo   - GHM only: 96.0-96.5%% F1
    echo   - GHM+Contrastive: 96.5-97%% F1
    echo.
) else (
    echo.
    echo Training cancelled.
    echo To train later: 
    echo   python multi_label\train_multilabel_focal_contrastive.py --config multi_label/config_focal_only.yaml --epochs 15
    echo.
)

pause
