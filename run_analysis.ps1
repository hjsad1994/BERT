# Script to run both analyze_results.py and error_analysis.py
# For Windows PowerShell

# Set UTF-8 encoding for output
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Change to script directory
Set-Location $PSScriptRoot

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "🚀 STARTING COMPLETE ANALYSIS PIPELINE" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if required files exist
Write-Host "📋 Checking requirements..." -ForegroundColor Blue
if (-not (Test-Path "test_predictions.csv")) {
    Write-Host "❌ Error: test_predictions.csv not found" -ForegroundColor Red
    Write-Host "Please run: python generate_test_predictions.py" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path "data\test.csv")) {
    Write-Host "❌ Error: data\test.csv not found" -ForegroundColor Red
    Write-Host "Please run: python prepare_data.py" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ All required files found" -ForegroundColor Green
Write-Host ""

# ========================================================================
# 1. Run analyze_results.py
# ========================================================================
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "📊 [1/2] Running analyze_results.py" -ForegroundColor Blue
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

python analyze_results.py
$analyzeExitCode = $LASTEXITCODE

if ($analyzeExitCode -ne 0) {
    Write-Host ""
    Write-Host "❌ analyze_results.py failed with exit code $analyzeExitCode" -ForegroundColor Red
    exit $analyzeExitCode
}

Write-Host ""
Write-Host "✓ analyze_results.py completed successfully" -ForegroundColor Green
Write-Host ""
Write-Host ""

# ========================================================================
# 2. Run error_analysis.py
# ========================================================================
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "🔍 [2/2] Running error_analysis.py" -ForegroundColor Blue
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

python tests\error_analysis.py
$errorExitCode = $LASTEXITCODE

if ($errorExitCode -ne 0) {
    Write-Host ""
    Write-Host "❌ error_analysis.py failed with exit code $errorExitCode" -ForegroundColor Red
    exit $errorExitCode
}

Write-Host ""
Write-Host "✓ error_analysis.py completed successfully" -ForegroundColor Green
Write-Host ""
Write-Host "📄 NEW: Now exports ALL errors to CSV files!" -ForegroundColor Cyan
Write-Host "   • all_errors_detailed.csv - ALL prediction errors with full details" -ForegroundColor Yellow
Write-Host "   • errors_summary_by_aspect.csv - Error summary grouped by aspect" -ForegroundColor Yellow

# ========================================================================
# Summary
# ========================================================================
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "✅ ALL ANALYSIS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📁 Results saved to:" -ForegroundColor Yellow
Write-Host "   • analysis_results\          (analyze_results.py output)"
Write-Host "   • error_analysis_results\    (error_analysis.py output)"
Write-Host ""
Write-Host "📊 Generated files:" -ForegroundColor Yellow

# Count files in analysis_results
if (Test-Path "analysis_results") {
    $analysisCount = (Get-ChildItem "analysis_results" -File).Count
    Write-Host "   • analysis_results\: $analysisCount files"
}

# Count files in error_analysis_results
if (Test-Path "error_analysis_results") {
    $errorCount = (Get-ChildItem "error_analysis_results" -File).Count
    Write-Host "   • error_analysis_results\: $errorCount files"
}

Write-Host ""
Write-Host "🆕 KEY FILES FOR DETAILED ERROR ANALYSIS:" -ForegroundColor Magenta
Write-Host "   • all_errors_detailed.csv - Contains ALL $errorCount errors for deep analysis" -ForegroundColor Yellow
Write-Host "   • errors_summary_by_aspect.csv - Error patterns by aspect and confusion type" -ForegroundColor Yellow
Write-Host "   • hard_cases.csv - Top 5 difficult cases per aspect" -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 TIP: Open all_errors_detailed.csv in Excel to analyze every single error!" -ForegroundColor Cyan
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
