$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 scripts\launch_local.py
    exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    python scripts\launch_local.py
    exit $LASTEXITCODE
}

Write-Error "Python 3 no esta instalado o no esta en PATH. Instala Python 3.11+."
exit 1
