@echo off
REM Quick Start: Focal Loss + Contrastive Learning
REM Run from D:\BERT\ (root folder)
REM Now includes automatic logging to training_logs folder

echo ================================================================================
echo FOCAL LOSS + CONTRASTIVE LEARNING TRAINING (WITH LOGGING)
echo ================================================================================
echo Method: 70%% Focal Loss + 30%% Contrastive Learning
echo Config: Can use config.yaml or override with args
echo Expected F1: 96.0-96.5%%
echo Training Time: ~45 minutes on RTX 3070
echo Logs: Saved to multi_label\models\...\training_logs\
echo ================================================================================
echo.

REM Training with logging (override some config values)
python multi_label\train_multilabel_focal_contrastive.py ^
    --epochs 8 ^
    --focal-weight 0.7 ^
    --contrastive-weight 0.3 ^
    --temperature 0.1 ^
    --output-dir multi_label\models\multilabel_focal_contrastive

echo.
echo ================================================================================
echo Training Complete!
echo.
echo Results saved to: multi_label\models\multilabel_focal_contrastive\
echo Check test_results_focal_contrastive.json for detailed metrics
echo ================================================================================
echo.
echo Note: This script uses explicit args. To use config-based training:
echo       Run: train_focal_contrastive_config.bat
pause
