@echo off
title AI Invest HQ v3.2.0
echo.
echo  ========================================
echo   AI Invest HQ v3.2.0
echo   Alpha Global Quant Trading Platform
echo   Powered by Isaac V3.9 + Will VCP V2.0
echo  ========================================
echo.
cd /d "%~dp0"
set "ROOT=%cd%"
if exist "%ROOT%\python_embed\python.exe" goto USE_EMBED
where python >nul 2>&1
if %errorlevel% equ 0 goto USE_SYSTEM
echo  [ERROR] Python not found!
pause
exit /b 1
:USE_EMBED
echo  [OK] Using embedded Python
set "PY=%ROOT%\python_embed\python.exe"
set "ST=%ROOT%\python_embed\Scripts\streamlit.exe"
if exist "%ROOT%\utils\fix_embed_pth.py" "%PY%" "%ROOT%\utils\fix_embed_pth.py" >nul 2>&1
goto DEPS
:USE_SYSTEM
echo  [OK] Using system Python
set "PY=python"
set "ST=streamlit"
goto DEPS
:DEPS
if not exist "%ROOT%\requirements.txt" goto KEYS
"%PY%" -c "import streamlit" >nul 2>&1
if %errorlevel% equ 0 goto KEYS
echo  [*] Installing dependencies...
"%PY%" -m pip install --upgrade pip >nul 2>&1
"%PY%" -m pip install -r "%ROOT%\requirements.txt"
if %errorlevel% neq 0 (
    echo  [ERROR] Install failed.
    pause
    exit /b 1
)
echo  [OK] Dependencies installed
:KEYS
if exist "%ROOT%\.streamlit\secrets.toml" goto LAUNCH
echo.
echo  [!] First run: API keys not configured
if not exist "%ROOT%\.streamlit" mkdir "%ROOT%\.streamlit"
if exist "%ROOT%\.streamlit\secrets.toml.example" (
    copy /Y "%ROOT%\.streamlit\secrets.toml.example" "%ROOT%\.streamlit\secrets.toml" >nul 2>&1
) else (
    echo GEMINI_API_KEY = ""> "%ROOT%\.streamlit\secrets.toml"
    echo FINLAB_API_KEY = "">> "%ROOT%\.streamlit\secrets.toml"
    echo SINOPAC_API_KEY = "">> "%ROOT%\.streamlit\secrets.toml"
    echo SINOPAC_SECRET_KEY = "">> "%ROOT%\.streamlit\secrets.toml"
    echo TELEGRAM_BOT_TOKEN = "">> "%ROOT%\.streamlit\secrets.toml"
    echo TELEGRAM_CHAT_ID = "">> "%ROOT%\.streamlit\secrets.toml"
)
echo  [OK] Config created
echo  Press any key to edit API keys...
pause >nul
notepad "%ROOT%\.streamlit\secrets.toml"
echo  Press any key to start...
pause >nul
:LAUNCH
if not exist "%ROOT%\data" mkdir "%ROOT%\data"

:: Kill any existing process on port 8501
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
    echo  [*] Killing old process on port 8501 (PID %%p)
    taskkill /PID %%p /F >nul 2>&1
)

:: Start scheduler as background process
echo  [*] Starting scheduler (background)...
start /B "" "%PY%" "%ROOT%\scheduler.py" --apscheduler

echo.
echo  [*] Starting at http://localhost:8501
echo  [*] Close this window to stop the server
echo.
"%ST%" run "%ROOT%\app.py" --server.port 8501 --server.headless false --browser.gatherUsageStats false

:: Cleanup when user closes or Ctrl+C
echo.
echo  [*] Shutting down...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
    taskkill /PID %%p /F >nul 2>&1
)
taskkill /F /IM streamlit.exe >nul 2>&1
