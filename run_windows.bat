@echo off
echo ==========================================
echo      Alpha Global Quantitative Platform
echo ==========================================
echo.
echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.11 from https://www.python.org/
    pause
    exit /b
)

for /f "tokens=2" %%I in ('python --version') do set PYTHON_VER=%%I
echo Found Python version: %PYTHON_VER%

echo %PYTHON_VER% | findstr "3.11" >nul
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] ---------------------------------------------------
    echo It is highly recommended to use Python 3.11 for Finlab stability.
    echo Your current version is %PYTHON_VER%.
    echo If you encounter errors, please install Python 3.11.
    echo -------------------------------------------------------------
    echo.
    pause
)

echo [2/3] Installing dependencies...
pip install -r requirements.txt

echo.
echo [3/3] Launching App...
echo.
echo The App will open in your default browser shortly.
echo To close, close the terminal window.
echo.
streamlit run app.py

pause
