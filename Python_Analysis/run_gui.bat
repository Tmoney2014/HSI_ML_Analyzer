@echo off
echo ===================================================
echo   HSI Analysis Environment Launcher
echo ===================================================

cd /d "%~dp0"

IF NOT EXIST "venv" (
    echo [Info] Creating Virtual Environment...
    python -m venv venv
)

echo [Info] Activating Virtual Environment...
call venv\Scripts\activate

echo [Info] Checking/Installing Dependencies...
pip install -r requirements.txt

echo.
echo [Info] Starting GUI Application...
python app.py

pause
