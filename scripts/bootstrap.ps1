param(
    [switch]$WithReport
)

$ErrorActionPreference = "Stop"

python -m venv .venv

$extras = ".[dev]"
if ($WithReport) {
    $extras = ".[dev,report]"
}

& .venv\Scripts\python -m pip install -e $extras
& .venv\Scripts\python scripts\generate_proto.py

Write-Host "Bootstrap complete."
Write-Host "Use: .venv\Scripts\python -m pytest -m smoke"
