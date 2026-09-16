@echo off
REM Double-click this file (or run it from a terminal) to run the screener.
REM First run: creates a private Python environment, installs packages,
REM downloads ~3 GB of SEC data, then screens ~2,000 companies (60-90 min).
REM Later runs reuse everything and take ~30-60 min.
REM
REM Options (run from a terminal):  run_screener.bat --test       quick 5-min check
REM                                 run_screener.bat --refresh-data  re-pull SEC data
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/
    echo IMPORTANT: tick "Add python.exe to PATH" in the installer, then run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment ^(one time^)...
    python -m venv .venv || (echo Could not create the environment. & pause & exit /b 1)
)
call ".venv\Scripts\activate.bat"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt || (echo Package install failed. Check your internet connection. & pause & exit /b 1)

python scripts\run_screener.py %*
set RC=%ERRORLEVEL%
echo.
if %RC% NEQ 0 echo The run stopped with an error. Scroll up for the message, or check output\run_*.log
pause
exit /b %RC%
