$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent

Push-Location "$root\backend"
if (-not (Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt | Out-Host
$back = Start-Process -PassThru -NoNewWindow -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m","uvicorn","app.main:app","--reload","--port","8000"
Pop-Location

Push-Location "$root\frontend"
if (-not (Test-Path "node_modules")) { npm install | Out-Host }
$front = Start-Process -PassThru -NoNewWindow -FilePath "npm" -ArgumentList "run","dev"
Pop-Location

Write-Host "Backend  PID: $($back.Id)  ->  http://127.0.0.1:8000"
Write-Host "Frontend PID: $($front.Id) ->  http://127.0.0.1:3000"
Write-Host "Press Ctrl+C to stop."

try {
    Wait-Process -Id $back.Id, $front.Id
}
finally {
    Stop-Process -Id $back.Id, $front.Id -ErrorAction SilentlyContinue
}
