@echo off
echo [HSI_ML_Analyzer] Setting up Python Environment...

:: Create Virtual Environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

:: Activate and Install Requirements
echo Installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo.
echo ========================================================
echo Setup Complete! To activate manually, run:
echo venv\Scripts\activate
echo ========================================================
pause
