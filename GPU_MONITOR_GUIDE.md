# GPU Monitoring Guide for Windows

## Method 1: PowerShell Script (Recommended)

### Run the monitor script:
```powershell
.\monitor_gpu.ps1
```

**If you get execution policy error:**
```powershell
PowerShell -ExecutionPolicy Bypass -File .\monitor_gpu.ps1
```

---

## Method 2: One-liner PowerShell

### Copy and paste this into PowerShell:
```powershell
while($true) { Clear-Host; nvidia-smi; Write-Host "`nRefreshing... (Ctrl+C to stop)" -ForegroundColor Yellow; Start-Sleep -Seconds 1 }
```

---

## Method 3: Simple Loop (Easiest)

```powershell
while(1) { cls; nvidia-smi; sleep 1 }
```

---

## Method 4: Compact View (GPU % and Memory only)

```powershell
while($true) { 
    Clear-Host
    Write-Host "GPU Monitor - Press Ctrl+C to stop" -ForegroundColor Cyan
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
    Start-Sleep -Seconds 1
}
```

---

## Method 5: Task Manager (GUI)

1. Open **Task Manager** (Ctrl+Shift+Esc)
2. Go to **Performance** tab
3. Click **GPU 0** (your RTX 3070)
4. Watch:
   - GPU usage %
   - Memory usage
   - Temperature

---

## What to Look For

### During Training:

✅ **GPU Utilization: 99-100%**
```
|  0  NVIDIA GeForce RTX 3070  | 99%  |  ← Should be 99-100%
```

✅ **Memory Usage: 7.5-7.8GB / 8GB**
```
|  7650MiB / 8192MiB           |       ← Should be ~7.5-7.8GB
```

✅ **Temperature: <85°C**
```
|  75C                         |       ← Should be <85°C
```

### If GPU is Under 95%:

❌ **Problem:** Data loading bottleneck
**Solution:** Increase `dataloader_num_workers` to 4

❌ **Problem:** Small batch size
**Solution:** Increase `per_device_train_batch_size` to 96 (already done!)

❌ **Problem:** CPU bottleneck
**Solution:** Close other programs, check Task Manager CPU usage

---

## Training Workflow

### Terminal 1 (Training):
```powershell
cd D:\BERT
python train.py
```

### Terminal 2 (Monitoring):
```powershell
cd D:\BERT
.\monitor_gpu.ps1
```

**OR use one-liner:**
```powershell
while(1) { cls; nvidia-smi; sleep 1 }
```

---

## Alternative: nvidia-smi Commands

### Continuous monitoring (compact):
```powershell
nvidia-smi dmon -s u
```

### Watch specific metrics:
```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader --loop=1
```

### Full details every 2 seconds:
```powershell
nvidia-smi -l 2
```

---

## Expected Output During Training

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.x    |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  N/A |
| 40%   75C    P2   220W / 220W |   7650MiB /  8192MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
                                  ↑                       ↑
                            7.5GB/8GB              99% Utilization
                              ✓                          ✓
```

---

## Quick Commands Reference

### Start Training:
```powershell
python train.py
```

### Monitor (choose one):

**Easiest:**
```powershell
while(1) { cls; nvidia-smi; sleep 1 }
```

**Best (use script):**
```powershell
.\monitor_gpu.ps1
```

**Built-in:**
```powershell
nvidia-smi -l 1
```

**Compact:**
```powershell
nvidia-smi dmon -s u
```

---

## Troubleshooting

### "nvidia-smi is not recognized"

**Add NVIDIA to PATH:**
```powershell
$env:Path += ";C:\Program Files\NVIDIA Corporation\NVSMI"
nvidia-smi
```

**Or use full path:**
```powershell
& "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
```

### Script won't run (Execution Policy)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**OR run with bypass:**
```powershell
PowerShell -ExecutionPolicy Bypass -File .\monitor_gpu.ps1
```

---

## Summary

**Simplest command (copy-paste into PowerShell):**
```powershell
while(1) { cls; nvidia-smi; sleep 1 }
```

**Start training in another terminal:**
```powershell
python train.py
```

**Watch for:**
- GPU: 99-100% ✓
- Memory: 7.5-7.8GB ✓
- Temp: <85°C ✓

**Press Ctrl+C to stop monitoring.**

Ready to train with full GPU power! 🚀
