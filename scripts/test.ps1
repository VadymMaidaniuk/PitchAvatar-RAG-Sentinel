param(
    [string]$Marker = "",
    [string]$Path = "tests"
)

$ErrorActionPreference = "Stop"

$args = @("-m", "pytest", $Path)
if ($Marker -ne "") {
    $args += @("-m", $Marker)
}

& .venv\Scripts\python @args
