# serve.ps1 - Start BlogBoard frontend + admin API cleanly.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\serve.ps1

$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path $PSScriptRoot -Parent
$web  = Join-Path $root "blogboard\web"
$py   = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "C:\Program Files\Python313\python.exe" }

Write-Host "Stopping stale servers on :8000 / :8001..."
Get-NetTCPConnection -LocalPort 8000,8001 -ErrorAction SilentlyContinue | ForEach-Object {
    Stop-Process -Id $_.OwningProcess -Force
}
Start-Sleep -Seconds 2

Write-Host "Starting frontend on http://localhost:8000 ..."
Push-Location $web
Start-Process -FilePath $py -ArgumentList @("-m", "http.server", "8000") -WindowStyle Minimized
Pop-Location

Write-Host "Starting admin API on http://localhost:8001 ..."
Push-Location $root
Start-Process -FilePath $py -ArgumentList @("-m", "uvicorn", "blogboard.api.app:app", "--port", "8001") -WindowStyle Minimized
Pop-Location

$up = $false
foreach ($i in 1..10) {
    Start-Sleep -Seconds 2
    try {
        $f = Invoke-WebRequest -Uri "http://localhost:8000/index.html" -UseBasicParsing -TimeoutSec 5
        $a = Invoke-WebRequest -Uri "http://localhost:8001/api/health" -UseBasicParsing -TimeoutSec 5
        if ($f.StatusCode -eq 200 -and $a.StatusCode -eq 200) { $up = $true; break }
    } catch { continue }
}

if ($up) {
    Write-Host ""
    Write-Host "  BOTH SERVERS UP" -ForegroundColor Green
    Write-Host "  Site: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "  API : http://localhost:8001/api/stats" -ForegroundColor Cyan
} else {
    Write-Host "  Servers did not respond in 20s. Check the minimized console windows." -ForegroundColor Yellow
}
