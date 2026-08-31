# Serve the detector on a free public URL for the demo / judging window.
#
#   powershell -ExecutionPolicy Bypass -File serve_demo.ps1
#
# Starts the FastAPI app and a Cloudflare quick tunnel, then prints a
# https://<random>.trycloudflare.com URL. No account, no cost. The URL lives
# only while this script runs; Ctrl+C stops both.
#
# Everything below is discovered at runtime -- no machine-specific paths, so
# this works on any Windows box with a checkout of this repo.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$PORT = 8000

# Python: prefer the project venv, else whatever is on PATH.
$PY = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { $PY = "python" }

# cloudflared: use a local copy if present, else fetch it once (~55 MB).
$CF = Join-Path $PSScriptRoot "tools\cloudflared.exe"
if (-not (Test-Path $CF)) {
  Write-Host "Downloading cloudflared (one time, ~55 MB)..." -ForegroundColor Cyan
  New-Item -ItemType Directory -Force (Join-Path $PSScriptRoot "tools") | Out-Null
  $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
  Invoke-WebRequest -Uri $url -OutFile $CF
}

# The webapp already defaults to checkpoints/detector.pt and auto-selects the
# device, so nothing needs to be pinned here.

Write-Host "Starting the detector API (loads the model, ~30s)..." -ForegroundColor Cyan

$apiArgs = @("-m", "uvicorn", "webapp.server:app", "--host", "127.0.0.1", "--port", "$PORT")
$api = Start-Process -FilePath $PY -ArgumentList $apiArgs -PassThru -NoNewWindow `
        -RedirectStandardOutput "$env:TEMP\ttj_api.log" -RedirectStandardError "$env:TEMP\ttj_api.err"

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
  Write-Host "(If the weights are missing, run:  python inference.py demo_images)" -ForegroundColor Yellow
  if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force }
  exit 1
}

Write-Host ("API ready: {0} on {1}, {2:N0} params, threshold {3}" -f `
  $h.model_type, $h.device, $h.parameters, $h.threshold) -ForegroundColor Green
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
