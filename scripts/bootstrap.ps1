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
Write-Host "Use: .\scripts\test.ps1 -Profile grpc"
Write-Host "Or:  .\scripts\test.ps1 -Profile available"
