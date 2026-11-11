Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
$ErrorActionPreference = "Stop"

if (-not (Get-Command idf.py -ErrorAction SilentlyContinue)) {
    if ($env:IDF_PATH) {
        $exportScript = Join-Path $env:IDF_PATH "export.ps1"
    } else {
        $exportScript = "C:\esp\v6.0.1\esp-idf\export.ps1"
        if (-not $env:IDF_TOOLS_PATH) {
            $env:IDF_TOOLS_PATH = "C:\Espressif"
        }
        if (-not $env:IDF_PYTHON_ENV_PATH) {
            $env:IDF_PYTHON_ENV_PATH = Join-Path $env:IDF_TOOLS_PATH "tools\python\v6.0.1\venv"
        }
    }
    . $exportScript
}

$projectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectDirectory
try {
    idf.py build
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
