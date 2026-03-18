@echo off
echo ==========================================
echo      Alpha Global Quantitative Platform
echo ==========================================
echo.

set TARGET_PY=python

:: Check for Embedded Python
if exist "python_embed\python.exe" (
    echo [INFO] Found Embedded Python. Using portable environment.
    set TARGET_PY=python_embed\python.exe

    :: Ensure pip is installed in embedded python (it requires get-pip.py usually,
    :: but assuming user prepared it or we use ensurepip if available)
    :: Note: Embedded python often needs 'pth' file edit to see site-packages.
    :: We use a helper script to fix this automatically.

    echo [INFO] Patching embedded python configuration...
    "%TARGET_PY%" utils\fix_embed_pth.py

    :: Ensure pip is installed. Embedded python doesn't come with pip by default.
    :: We download get-pip.py if not present.
    "%TARGET_PY%" -m pip --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [INFO] Pip not found. Attempting to install pip...
        if not exist "get-pip.py" (
            echo [INFO] Downloading get-pip.py...
            curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        )
        "%TARGET_PY%" get-pip.py
    )

    goto :INSTALL_DEPS
)

echo [1/3] Checking System Python installation...
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

:INSTALL_DEPS
echo.
echo [2/3] Upgrading pip and installing dependencies...
"%TARGET_PY%" -m pip install --upgrade pip
"%TARGET_PY%" -m pip install --upgrade -r requirements.txt

:: Check if pip install was actually successful
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install required packages!
    echo The "No module named streamlit" error means the installation process above failed.
    echo Please scroll up and read the red error text to see why it failed.
    echo Possible reasons: no internet, missing C++ build tools, or Windows long path issues.
    echo.
    pause
    exit /b
)

echo.
echo [3/3] Launching App...
echo.
echo The App will open in your default browser shortly.
echo To close, close the terminal window.
echo.
"%TARGET_PY%" -m streamlit run app.py

pause
