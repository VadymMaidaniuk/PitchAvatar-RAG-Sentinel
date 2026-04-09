param(
    [ValidateSet("full", "available", "grpc")]
    [string]$Profile = "",
    [string]$Marker = "",
    [string[]]$Path = @()
)

$ErrorActionPreference = "Stop"

if ($Profile -ne "" -and $Path.Count -gt 0) {
    throw "Use either -Profile or -Path, not both."
}

$targets = switch ($Profile) {
    "grpc" { @("tests\clients", "tests\smoke\test_smoke.py") }
    "available" { @("tests\clients", "tests\smoke", "tests\search", "tests\workflow") }
    "full" { @("tests") }
    default {
        if ($Path.Count -gt 0) {
            $Path
        } else {
            @("tests")
        }
    }
}

$args = @("-m", "pytest") + $targets
if ($Marker -ne "") {
    $args += @("-m", $Marker)
}

& .venv\Scripts\python @args
