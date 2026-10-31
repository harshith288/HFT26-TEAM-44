# One-command run. Usage: powershell -ExecutionPolicy Bypass -File src/scripts/run.ps1 [mode]
#   dashboard  main dashboard at http://localhost:8000, opens the browser (default)
#   streamlit  backup Streamlit dashboard
#   test       full test suite
#   pipeline   process train and val images into SQLite
#   once       process new files in data/incoming, then stop
#   watch      keep checking data/incoming
param([ValidateSet('dashboard', 'streamlit', 'test', 'pipeline', 'once', 'watch')][string]$Mode = 'dashboard',
      [switch]$NoBrowser, [int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot)
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw 'No .venv found. Run src/scripts/setup.ps1 first.' }

Set-Location $root
switch ($Mode) {
    'test' { & $py -m pytest tests -v }
    'pipeline' { & $py -m src.pipeline --sample }
    'once' { & $py -m src.pipeline --once }
    'watch' { & $py -m src.pipeline --watch }
    'streamlit' { & $py -m streamlit run src/streamlit_app.py }
    'dashboard' {
        if (-not $NoBrowser) {
            # Opens the page once the server answers, up to a minute from now. This polls
            # 127.0.0.1 because localhost tries IPv6 first, which is slower than the timeout.
            $open = 'for ($i = 0; $i -lt 60; $i++) { try { Invoke-WebRequest http://127.0.0.1:$Port/ -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process http://localhost:$Port; break } catch { Start-Sleep 1 } }'.Replace('$Port', "$Port")
            Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile', '-Command', $open
        }
        & $py -m uvicorn src.server:app --port $Port
    }
}
exit $LASTEXITCODE
