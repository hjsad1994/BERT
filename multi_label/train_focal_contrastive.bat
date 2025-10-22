@echo off
REM Focal Loss + Contrastive Learning (COMBO!)
REM Focal: Handle class imbalance (gamma=2.0)
REM Contrastive: Learn better representations (soft weighting)
REM Expected: 96-96.5%% F1

echo ================================================================================
echo Training FOCAL LOSS + CONTRASTIVE LEARNING (COMBO!)
echo ================================================================================
echo Why this should work:
echo   - Focal Loss: Focus on hard examples, handle class imbalance
echo   - Contrastive: Learn better representations (already working!)
echo   - Balance: 70%% focal (classification) + 30%% contrastive (representations)
echo.
echo Expected losses:
echo   - Focal: 0.2-0.4 (much better than 0.07 CE!)
echo   - Contrastive: 0.3-0.6 (same as before)
echo   - Combined: Better classification + good representations
echo.
echo Expected F1: 96-96.5%% (vs 95.44%% contrastive-only)
echo ================================================================================
echo.

python train_multilabel_focal_contrastive.py ^
    --epochs 8 ^
    --focal-weight 0.7 ^
    --contrastive-weight 0.3 ^
    --temperature 0.1 ^
    --output-dir multilabel_focal_contrastive_model

echo.
echo ================================================================================
echo Training Complete!
echo Check if:
echo   - Focal loss ~0.2-0.4 (better than 0.07!)
echo   - Contrastive loss ~0.3-0.6 (same as before)
echo   - Test F1 > 96%% (target!)
echo ================================================================================
