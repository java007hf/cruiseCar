@echo off
setlocal

set GRADLE_BAT=%USERPROFILE%\.gradle\wrapper\dists\gradle-8.2-bin\bbg7u40eoinfdyxsxr3z4i7ta\gradle-8.2\bin\gradle.bat
if not exist "%GRADLE_BAT%" (
  set GRADLE_BAT=%USERPROFILE%\.gradle\wrapper\dists\gradle-8.11.1-bin\bpt9gzteqjrbo1mjrsomdt32c\gradle-8.11.1\bin\gradle.bat
)
if not exist "%GRADLE_BAT%" (
  echo Gradle was not found in the local wrapper cache.
  exit /b 1
)

call "%GRADLE_BAT%" %*

