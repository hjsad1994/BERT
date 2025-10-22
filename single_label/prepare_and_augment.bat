@echo off
REM ============================================================================
REM Single-Label Data Preparation Script
REM ============================================================================
REM Automatically prepare and augment data for single-label ABSA
REM
REM Usage: Run from D:\BERT\
REM   single_label\prepare_and_augment.bat
REM ============================================================================

echo.
echo ============================================================================
echo 🚀 SINGLE-LABEL DATA PREPARATION
echo ============================================================================
echo.

REM Check if we're in the right directory
if not exist "dataset.csv" (
    echo ❌ Error: Please run this script from D:\BERT\ directory
    echo.
    echo Usage:
    echo   cd D:\BERT
    echo   single_label\prepare_and_augment.bat
    exit /b 1
)

echo ✓ Found dataset.csv
echo.

REM Step 1: Prepare data
echo ============================================================================
echo 📊 STEP 1/2: Preparing data (convert to single-label format)
echo ============================================================================
echo.

python single_label\prepare_data.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ Data preparation failed!
    exit /b 1
)

echo.
echo ✓ Data preparation completed!
echo.

REM Step 2: Augment data
echo ============================================================================
echo 🔄 STEP 2/2: Augmenting data (Neutral + 'nhưng' oversampling)
echo ============================================================================
echo.

python single_label\augment_neutral_and_nhung.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ Data augmentation failed!
    exit /b 1
)

echo.
echo ============================================================================
echo ✅ DATA PREPARATION COMPLETE!
echo ============================================================================
echo.
echo 📁 Data files created in: single_label/data/
echo.
echo Files:
echo   • train.csv (original)
echo   • validation.csv
echo   • test.csv
echo   • train_augmented_neutral_nhung.csv (augmented)
echo   • data_metadata.json
echo.
echo 🎯 NEXT STEPS:
echo.
echo 1. Update config (already configured):
echo    single_label/config_single.yaml
echo.
echo 2. Train model:
echo    python single_label\train.py --config single_label\config_single.yaml
echo.
echo ============================================================================
echo.

pause
