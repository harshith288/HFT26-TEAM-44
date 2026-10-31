# One-command setup: Python 3.11 venv, pinned packages, models, sampled images.
# Usage: powershell -ExecutionPolicy Bypass -File src/scripts/setup.ps1 [-BrowserTests]
#   -BrowserTests  also installs Playwright, for src/scripts/e2e_upload.py and
#                  src/scripts/capture_screenshots.py. They use the installed Edge, so no
#                  browser is downloaded. The app and pytest do not need it.
param([switch]$BrowserTests)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot)
$py = Join-Path $root '.venv\Scripts\python.exe'

# Native tools do not stop the script on failure, so check each exit code.
function Invoke-Step($name, [scriptblock]$step) {
    Write-Host "== $name"
    & $step
    if ($LASTEXITCODE -ne 0) { throw "$name failed with exit code $LASTEXITCODE" }
}

Invoke-Step 'Install uv' { python -m pip install --user --quiet uv }
if (-not (Test-Path $py)) {
    # uv can print a harmless link error after a good install, so test the interpreter instead.
    python -m uv python install 3.11
    Invoke-Step 'Create venv' { python -m uv venv --python 3.11 (Join-Path $root '.venv') }
}
Invoke-Step 'Install PyTorch' { python -m uv pip install --python $py -r (Join-Path $root 'requirements-torch.txt') }
Invoke-Step 'Install packages' { python -m uv pip install --python $py -r (Join-Path $root 'requirements.txt') }
if ($BrowserTests) {
    Invoke-Step 'Install Playwright' { python -m uv pip install --python $py -r (Join-Path $root 'requirements.txt') }
}

foreach ($dir in 'data\incoming', 'data\golden', 'docs\reports') {
    New-Item -ItemType Directory -Force (Join-Path $root $dir) | Out-Null
}

Invoke-Step 'Check CUDA' { & $py -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())" }
Invoke-Step 'Download models' { & $py (Join-Path $root 'src\scripts\download_models.py') }

$raw = Join-Path $root 'data\raw'
New-Item -ItemType Directory -Force $raw | Out-Null
$base = 'https://storage.googleapis.com/public-datasets-lila/caltechcameratraps'
$annotations = Join-Path $raw 'eccv_18_annotations.tar.gz'
if (-not (Test-Path $annotations)) {
    Invoke-WebRequest "$base/eccv_18_annotations.tar.gz" -OutFile $annotations
}
Invoke-Step 'Extract CCT20 metadata' { tar -xzf $annotations -C $raw }
$bboxes = Join-Path $raw 'caltech_bboxes_20200316.json'
if (-not (Test-Path $bboxes)) {
    Invoke-WebRequest "$base/labels/caltech_bboxes_20200316.json" -OutFile $bboxes
}
Invoke-Step 'Sample and download images' { & $py (Join-Path $root 'src\scripts\sample_cct20.py') }
Invoke-Step 'Build golden set' { & $py (Join-Path $root 'src\scripts\make_golden.py') }

Write-Host 'SETUP_DONE'
