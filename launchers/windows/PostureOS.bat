@echo off
setlocal
cd /d "%~dp0\..\.."

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  py -3 scripts\launch_local.py
  exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  python scripts\launch_local.py
  exit /b %ERRORLEVEL%
)

echo Python 3 no esta instalado o no esta en PATH.
echo Instala Python 3.11+ desde https://www.python.org/downloads/windows/
pause
exit /b 1
