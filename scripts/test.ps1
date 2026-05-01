param(
    [ValidateSet("full", "available", "grpc", "offline")]
    [string]$Profile = "",
    [string]$Marker = "",
    [string[]]$Path = @()
)

$ErrorActionPreference = "Stop"

if ($Profile -ne "" -and $Path.Count -gt 0) {
    throw "Use either -Profile or -Path, not both."
}

$targets = switch ($Profile) {
    "offline" { @("tests") }
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

$profileMarker = switch ($Profile) {
    "offline" { "not integration and not destructive" }
    default { "" }
}
$effectiveMarker = if ($Marker -ne "") { $Marker } else { $profileMarker }

$args = @("-m", "pytest") + $targets
if ($effectiveMarker -ne "") {
    $args += @("-m", $effectiveMarker)
}

& .venv\Scripts\python @args
