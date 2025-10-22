@echo off
REM Train with GHM-C Loss ONLY (no contrastive)

echo ================================================================================
echo GHM-C Loss ONLY Training (No Contrastive)
echo ================================================================================
echo.

echo Step 1: Verify config...
echo.
python test_ghm_only.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Config verification failed!
    echo Please check multi_label\config_ghm.yaml
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Config verified! GHM-C Loss ONLY (no contrastive)
echo ================================================================================
echo.

echo Settings:
echo   - Loss: GHM-C 100%%
echo   - Contrastive: DISABLED
echo   - Output: multi_label\models\multilabel_ghm_only\
echo.

echo Start training? (15 epochs, ~75 minutes)
set /p choice="Type 'y' to start, 'n' to exit: "

if /i "%choice%"=="y" (
    echo.
    echo Starting training...
    echo.
    python multi_label\train_multilabel_ghm_contrastive.py --epochs 15
    
    echo.
    echo ================================================================================
    echo Training Complete!
    echo ================================================================================
    echo.
    echo Check results:
    echo   type multi_label\models\multilabel_ghm_only\test_results_ghm_contrastive.json
    echo.
    echo Compare with Focal Loss:
    echo   Focal:  95.99%% F1
    echo   GHM-C:  ? (check results)
    echo.
) else (
    echo.
    echo Training cancelled.
    echo To train later: python multi_label\train_multilabel_ghm_contrastive.py --epochs 15
    echo.
)

pause
