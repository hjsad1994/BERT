# GPU Monitor Script for Windows
# Run this while training to monitor GPU usage

while ($true) {
    Clear-Host
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "  GPU MONITOR - RTX 3070" -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Run nvidia-smi
    nvidia-smi
    
    Write-Host ""
    Write-Host "Refreshing every 1 second... (Ctrl+C to stop)" -ForegroundColor Yellow
    
    Start-Sleep -Seconds 1
}
