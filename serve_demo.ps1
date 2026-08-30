# Serve the webapp on a free public URL for the demo / judging window.
#
#   powershell -ExecutionPolicy Bypass -File serve_demo.ps1
#
# Starts the FastAPI app (iter7 checkpoint) and a Cloudflare quick tunnel.
# Prints a https://<random>.trycloudflare.com URL. Ctrl+C stops both.
# No account, no cost. The URL lives only while this script runs.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$PY   = "C:\Users\attil\ttj-venv26\Scripts\python.exe"
$CF   = "C:\Users\attil\tools\cloudflared.exe"
$PORT = 8000

$env:DETECTOR_CHECKPOINT = "runs/iter7/best.pt"
$env:DETECTOR_DEVICE     = "xpu"
$env:HF_HOME             = "C:\Users\attil\ttj-cache\hf"
$env:SYCL_CACHE_PERSISTENT = "1"
$env:SYCL_CACHE_DIR      = "C:\Users\attil\ttj-cache\sycl26"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Write-Host "Starting the detector API (loads the model, ~30s)..." -ForegroundColor Cyan

$apiArgs = @("-m", "uvicorn", "webapp.server:app", "--host", "127.0.0.1", "--port", "$PORT")
$api = Start-Process -FilePath $PY -ArgumentList $apiArgs -PassThru -NoNewWindow -RedirectStandardOutput "$env:TEMP\ttj_api.log" -RedirectStandardError "$env:TEMP\ttj_api.err"

$ready = $false
$h = $null
for ($i = 0; $i -lt 60; $i++) {
  Start-Sleep 2
  try {
    $h = Invoke-RestMethod "http://127.0.0.1:$PORT/health" -TimeoutSec 3
    if ($h.ok) { $ready = $true; break }
  } catch {}
}

if (-not $ready) {
  Write-Host "API did not come up. Check $env:TEMP\ttj_api.err" -ForegroundColor Red
  if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force }
  exit 1
}

$bk = if ($h.branch_kind) { "/" + $h.branch_kind } else { "" }
Write-Host ("API ready: {0}{1} on {2}, threshold {3}" -f $h.model_type, $bk, $h.device, $h.threshold) -ForegroundColor Green
Write-Host ""
Write-Host "Opening the public tunnel. The https://<...>.trycloudflare.com URL appears below." -ForegroundColor Cyan
Write-Host "Share that URL. Press Ctrl+C here to stop everything." -ForegroundColor Cyan
Write-Host ""

try {
  & $CF tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate
} finally {
  if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force }
  Write-Host ""
  Write-Host "Stopped. The public URL is now dead." -ForegroundColor Yellow
}
