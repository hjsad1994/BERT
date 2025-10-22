@echo off
REM Quick batch script to test GHM Loss
echo ================================================================================
echo GHM-C Loss Testing Suite
echo ================================================================================
echo.

echo Step 1: Testing GHM Loss Implementation...
echo.
python test_ghm_quick.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Test failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Test PASSED! GHM-C Loss is working.
echo ================================================================================
echo.

echo Do you want to start full training? (15 epochs, ~90 minutes)
echo.
set /p choice="Type 'y' to start training, or 'n' to exit: "

if /i "%choice%"=="y" (
    echo.
    echo Starting training with GHM-C Loss...
    echo Output: multi_label\models\multilabel_ghm_contrastive\
    echo Logs: multi_label\models\multilabel_ghm_contrastive\training_logs\
    echo.
    python multi_label\train_multilabel_ghm_contrastive.py --epochs 15
    
    echo.
    echo ================================================================================
    echo Training Complete!
    echo ================================================================================
    echo.
    echo Check results:
    echo   type multi_label\models\multilabel_ghm_contrastive\test_results_ghm_contrastive.json
    echo.
) else (
    echo.
    echo Training skipped. To train later, run:
    echo   python multi_label\train_multilabel_ghm_contrastive.py --epochs 15
    echo.
)

echo ================================================================================
echo All Done!
echo ================================================================================
pause
