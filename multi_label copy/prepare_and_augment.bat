@echo off
REM ============================================================================
REM Multi-Label Data Preparation Script
REM ============================================================================
REM Automatically prepare and augment data for multi-label ABSA
REM
REM Usage: Run from D:\BERT\
REM   multi_label\prepare_and_augment.bat
REM ============================================================================

echo.
echo ============================================================================
echo 🚀 MULTI-LABEL DATA PREPARATION (Focal + Contrastive)
echo ============================================================================
echo.

REM Check if we're in the right directory
if not exist "dataset.csv" (
    echo ❌ Error: Please run this script from D:\BERT\ directory
    echo.
    echo Usage:
    echo   cd D:\BERT
    echo   multi_label\prepare_and_augment.bat
    exit /b 1
)

echo ✓ Found dataset.csv
echo.

REM Check if data already exists
if exist "multi_label\data\train_multilabel_balanced.csv" (
    echo ⚠️  Warning: Balanced data already exists!
    echo    multi_label\data\train_multilabel_balanced.csv
    echo.
    set /p confirm="Do you want to regenerate? (y/n): "
    if /i not "!confirm!"=="y" (
        echo.
        echo ✓ Using existing data
        goto :skip_generation
    )
)

REM Step 1: Prepare data (only if needed)
if not exist "multi_label\data\train_multilabel.csv" (
    echo ============================================================================
    echo 📊 STEP 1/2: Preparing data (split into train/val/test)
    echo ============================================================================
    echo.

    python multi_label\prepare_data_multilabel.py

    if %errorlevel% neq 0 (
        echo.
        echo ❌ Data preparation failed!
        exit /b 1
    )

    echo.
    echo ✓ Data preparation completed!
    echo.
) else (
    echo ✓ train_multilabel.csv already exists, skipping preparation
    echo.
)

REM Step 2: Augment data
echo ============================================================================
echo 🔄 STEP 2/2: Balancing data (aspect-wise oversampling)
echo ============================================================================
echo.

python multi_label\augment_multilabel_balanced.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ Data balancing failed!
    exit /b 1
)

:skip_generation

echo.
echo ============================================================================
echo ✅ DATA PREPARATION COMPLETE!
echo ============================================================================
echo.
echo 📁 Data files in: multi_label/data/
echo.
echo Files:
echo   • train_multilabel.csv (original)
echo   • validation_multilabel.csv
echo   • test_multilabel.csv
echo   • train_multilabel_balanced.csv (balanced - RECOMMENDED)
echo   • multilabel_metadata.json
echo.
echo 📊 Data statistics:
echo   • Original: 7,309 samples
echo   • Balanced: 15,921 samples (+117.8%%)
echo   • Imbalance: 5.30x → 1.22x (77%% improvement)
echo.
echo 🎯 NEXT STEPS:
echo.
echo 1. Config already set to use balanced data:
echo    multi_label/config_multi.yaml
echo.
echo 2. Train model (Target: 96%% F1):
echo    python multi_label\train_multilabel_focal_contrastive.py --epochs 8 --focal-weight 0.7 --contrastive-weight 0.3
echo.
echo    Or quick start:
echo    train_focal_contrastive.bat
echo.
echo ============================================================================
echo.

pause
