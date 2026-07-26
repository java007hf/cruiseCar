Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
$ErrorActionPreference = "Stop"

$env:IDF_TOOLS_PATH = "C:\Espressif"
$env:IDF_PYTHON_ENV_PATH = "C:\Espressif\tools\python\v6.0.1\venv"
. "C:\esp\v6.0.1\esp-idf\export.ps1"

idf.py `
  -D CMAKE_MAKE_PROGRAM=C:\Espressif\tools\ninja\1.12.1\ninja.exe `
  build
