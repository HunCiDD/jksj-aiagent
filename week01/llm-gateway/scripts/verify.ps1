param(
    [switch]$Live
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found. Run 'uv sync' from $repoRoot first."
}
Push-Location $repoRoot
try {
    uv run --no-sync --python $python pytest -q
    if ($Live) {
        uv run --no-sync --python $python week01/llm-gateway/scripts/smoke.py
    }
}
finally {
    Pop-Location
}
