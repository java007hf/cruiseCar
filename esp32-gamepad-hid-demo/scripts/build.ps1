Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
$ErrorActionPreference = "Stop"

$env:IDF_TOOLS_PATH = "C:\Users\Administrator\.espressif"
. "C:\esp\v6.0.1\esp-idf\export.ps1"

idf.py `
  -D CMAKE_MAKE_PROGRAM=C:\Users\Administrator\.espressif\tools\ninja\1.12.1\ninja.exe `
  build
